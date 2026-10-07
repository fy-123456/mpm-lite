"""GPU-resident quadrature for the same fixed-reference Q1 potential.

Only evaluation/assembly and storage change: full consistent mass, float64
constitutive law, left clamp and the existing L-BFGS/Armijo remain in use.
Nodal accumulated increments encode exactly the retained compatible history.
The optional consistent_pic mode explicitly reads particle velocities, projects
with the full mass, and measures particle kinetic energy after every update.
This is a material-reference path, not an Eulerian Lite/MLS or APIC remap.
"""
import numpy as np
import warp as wp

from .refinement_gpu import response_kernel
from .refinement import TensorReferenceQ1
from .convergence_reference import state_from_field, sum_fields
from .consistent_transfer import minimize_with_backtracking


@wp.kernel
def trial_gradient(F0: wp.array(dtype=wp.mat33d), ids: wp.array2d(dtype=wp.int32),
                   gradients: wp.array2d(dtype=wp.vec3d), coefficients: wp.array(dtype=wp.vec3d),
                   F: wp.array(dtype=wp.mat33d)):
    q = wp.tid()
    value = F0[q]
    for a in range(8):
        value += wp.outer(coefficients[ids[q,a]],gradients[q,a])
    F[q] = value


@wp.kernel
def jacobian_chunks(F: wp.array(dtype=wp.mat33d), result: wp.array(dtype=wp.float64), n: int):
    k = wp.tid()
    value = wp.float64(1.0e30)
    for q in range(k*32,wp.min((k+1)*32,n)):
        determinant = wp.determinant(F[q])
        if not wp.isfinite(determinant):
            determinant = wp.float64(-1.0)
        value = wp.min(value,determinant)
    result[k] = value


@wp.kernel
def energy_chunks(psi: wp.array(dtype=wp.float64), w: wp.array(dtype=wp.float64),
                  result: wp.array(dtype=wp.float64), n: int):
    k = wp.tid()
    value = wp.float64(0.)
    for q in range(k*32,wp.min((k+1)*32,n)):
        value += w[q]*psi[q]
    result[k] = value


@wp.kernel
def nodal_force(P: wp.array(dtype=wp.mat33d), w: wp.array(dtype=wp.float64),
                rowptr: wp.array(dtype=wp.int32), qids: wp.array(dtype=wp.int32),
                gradients: wp.array(dtype=wp.vec3d), force: wp.array(dtype=wp.vec3d)):
    a = wp.tid()
    value = wp.vec3d()
    for index in range(rowptr[a],rowptr[a+1]):
        q = qids[index]
        value += w[q]*(P[q] @ gradients[index])
    force[a] = value


class DeviceQuadrature:
    def __init__(self,basis,field,sites,device,material=True):
        self.device, self.n = device,len(sites.X)
        mapping = basis.sample(sites.X,sites.weight)
        if not np.all(np.diff(mapping.N.indptr)==8):
            raise ValueError('resident operator requires eight Q1 entries per point')
        for D in mapping.D:
            if not (np.array_equal(D.indices,mapping.N.indices) and np.array_equal(D.indptr,mapping.N.indptr)):
                raise ValueError('Q1 derivative sparsity mismatch')
        self.ids=wp.array(mapping.N.indices.reshape(-1,8).astype(np.int32),dtype=wp.int32,device=device)
        gradient=np.stack([D.data for D in mapping.D],axis=-1).reshape(-1,8,3)
        self.gradients=wp.array(gradient,dtype=wp.vec3d,device=device)
        self.F0=wp.array(field.evaluate(sites.X)[1],dtype=wp.mat33d,device=device)
        self.F=wp.empty(self.n,dtype=wp.mat33d,device=device)
        self.chunks=(self.n+31)//32
        self.determinants=wp.empty(self.chunks,dtype=wp.float64,device=device)
        if material:
            transpose=[D.T.tocsr() for D in mapping.D]
            for D in transpose[1:]:
                if not np.array_equal(D.indices,transpose[0].indices):
                    raise ValueError('transpose derivative sparsity mismatch')
            self.rowptr=wp.array(transpose[0].indptr.astype(np.int32),dtype=wp.int32,device=device)
            self.qids=wp.array(transpose[0].indices.astype(np.int32),dtype=wp.int32,device=device)
            self.tgradients=wp.array(np.stack([D.data for D in transpose],axis=-1),dtype=wp.vec3d,device=device)
            self.A=wp.array(sites.A,dtype=wp.mat33d,device=device)
            self.w=wp.array(sites.weight,dtype=wp.float64,device=device)
            self.psi=wp.empty(self.n,dtype=wp.float64,device=device)
            self.P=wp.empty(self.n,dtype=wp.mat33d,device=device)
            self.energies=wp.empty(self.chunks,dtype=wp.float64,device=device)

    def gradient(self,coefficients):
        wp.launch(trial_gradient,dim=self.n,inputs=[self.F0,self.ids,self.gradients,coefficients,self.F],device=self.device)
        wp.launch(jacobian_chunks,dim=self.chunks,inputs=[self.F,self.determinants,self.n],device=self.device)
        return float(self.determinants.numpy().min())


class ResidentReference:
    def __init__(self,source,initial,particles,quadrature,device='cuda:0',velocity_moments=None,
                 transfer_mode='nodal'):
        if transfer_mode not in ('nodal', 'consistent_pic'):
            raise ValueError('transfer_mode must be nodal or consistent_pic')
        self.transfer_mode = transfer_mode
        self.initial,self.source,self.device=initial,source,device
        state=state_from_field(initial['position'],particles,particles)
        self.base=TensorReferenceQ1(source,state)
        self.coefficients=np.zeros_like(source.X)
        self.device_coefficients=wp.zeros(len(source.X),dtype=wp.vec3d,device=device)
        self.force_device=wp.empty(len(source.X),dtype=wp.vec3d,device=device)
        self.q=DeviceQuadrature(self.base.basis,initial['position'],quadrature,device)
        self.p=DeviceQuadrature(self.base.basis,initial['position'],particles,device,material=False)
        vp=initial['velocity'].evaluate(particles.X)[0]
        self.velocity=self.base.project(vp)
        self.kinetic=.5*float(np.sum(particles.weight[:,None]*vp*vp))
        if velocity_moments is not None:
            self.kinetic,rhs=velocity_moments
            self.velocity=self.base.R @ self.base.mass_solve(self.base.R.T @ rhs)
        self.cache=None
        self.energy=self.elastic(self.coefficients)[0]
        self.last_min_det=min(self.cache[3],self.p.gradient(self.device_coefficients))
        if self.last_min_det <= 0:
            raise ValueError('inadmissible initial history')

    def elastic(self,coefficients):
        if self.cache is not None and np.array_equal(coefficients,self.cache[0]):
            return self.cache[1],self.cache[2]
        self.device_coefficients.assign(coefficients)
        minimum=self.q.gradient(self.device_coefficients)
        if minimum <= 0:
            raise ValueError('non-positive or non-finite material Jacobian')
        q=self.q; p=self.source.params
        wp.launch(response_kernel,dim=q.n,inputs=[q.F,q.A,p.mu,p.lam,p.k_f,q.psi,q.P],device=self.device)
        wp.launch(energy_chunks,dim=q.chunks,inputs=[q.psi,q.w,q.energies,q.n],device=self.device)
        wp.launch(nodal_force,dim=len(self.source.X),inputs=[q.P,q.w,q.rowptr,q.qids,q.tgradients,self.force_device],device=self.device)
        U=float(q.energies.numpy().sum()); force=self.force_device.numpy()
        if not np.isfinite(U) or not np.isfinite(force).all():
            raise ValueError('non-finite potential or force')
        self.cache=(coefficients.copy(),U,force,minimum)
        return U,force

    def step(self,dt,tolerance=1e-9,external=None):
        b=self.base; b._check_dt(dt)
        external=np.zeros_like(self.velocity) if external is None else np.asarray(external)
        if external.shape!=self.velocity.shape or not np.isfinite(external).all():
            raise ValueError('finite nodal reference load required')
        # Use the same Q1 basis for G2P, full-mass P2G and material gradients.
        # No lumping or sign clipping: the mass was independently integrated.
        predictor = (b.project(b.particles.N @ self.velocity) if self.transfer_mode == 'consistent_pic'
                     else self.velocity)
        transfer_error = float(np.linalg.norm(predictor-self.velocity) /
                               max(np.linalg.norm(self.velocity), 1e-30))
        yhat=b.mass_to_y(predictor[b.free])
        Uold,Kold=self.energy,self.kinetic
        def objective(y):
            v=b.R @ b.mass_from_y(y.reshape(-1,3))
            try: U,force=self.elastic(self.coefficients+dt*v)
            except ValueError: return np.inf,np.zeros_like(y)
            dy=y.reshape(-1,3)-yhat
            gradient=dy+dt*b.mass_force_to_y(b.R.T @ (force-external))
            return U+.5*float(np.sum(dy*dy))-dt*float(np.sum(external*v)),gradient.ravel()
        solution,iterations,backtracks=minimize_with_backtracking(objective,yhat.ravel(),tolerance=tolerance)
        value,grad=objective(solution)
        residual=float(np.max(abs(grad)))
        if not np.isfinite(value) or residual>tolerance*1.01:
            raise RuntimeError('resident solve did not converge')
        v=b.R @ b.mass_from_y(solution.reshape(-1,3))
        coefficients=self.coefficients+dt*v
        U,force=self.elastic(coefficients)
        # Explicitly restore accepted coefficients even after a cached evaluation.
        self.device_coefficients.assign(coefficients)
        minimum=min(self.cache[3],self.p.gradient(self.device_coefficients))
        if minimum<=0: raise ValueError('inadmissible particle update')
        nodal_kinetic=.5*float(np.sum(v*(b.M @ v)))
        K=(.5*float(np.sum(b.mass[:,None]*(b.particles.N @ v)**2))
           if self.transfer_mode == 'consistent_pic' else nodal_kinetic)
        dv=v-predictor; work=dt*float(np.sum(external*v))
        budget=dict(projection_delta=.5*float(np.sum(predictor*(b.M @ predictor)))-Kold,
                    inertia_remainder=-.5*float(np.sum(dv*(b.M @ dv))),
                    elastic_remainder=U-Uold-dt*float(np.sum(force*v)),
                    solver_work=float(np.sum(v*(b.M @ dv+dt*(force-external)))),
                    external_work=work,kinetic_readback_delta=K-nodal_kinetic)
        budget['energy_budget_residual']=U+K-Uold-Kold-sum(budget.values())
        self.coefficients,self.velocity,self.energy,self.kinetic=coefficients,v,U,K
        self.last_min_det=minimum
        return dict(elastic=U,kinetic=K,mechanical=U+K,iterations=iterations,backtracks=backtracks,
                    scaled_residual_inf=residual,min_det=minimum,
                    transfer_roundtrip_relative=transfer_error,transfer_mode=self.transfer_mode,
                    clamp_speed=float(np.max(abs(v[b.fixed]))),history_terms=len(self.fields()['position'].terms),**budget)

    def fields(self):
        return dict(position=sum_fields(self.initial['position'],self.base.coefficient_field(self.coefficients)),
                    velocity=self.base.coefficient_field(self.velocity))

    def close(self):
        self.base.close()
