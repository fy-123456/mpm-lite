"""Full-mass midpoint solid + identical discrete volume exchange.

Transport is explicitly a CONSTANT material-reference hydraulic network. It is
not an approximation to an advertised current-geometry permeability law.
Nonlinear Jv includes the determinant chain Hessian; rest K is only a predictor
and block preconditioner. No dense finite-difference Newton Jacobian.
"""
import time
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator,gmres
from .state import State,validate

class Solver:
    def __init__(self,model,geometry,material,*,drained=False,alpha=1.):
        self.model=model;self.geometry=geometry;self.material=material;self.alpha=float(alpha)
        self.M=np.kron(model.r.M,np.eye(3));self.K=model.r.K
        self.free=model.r.ids;self.fixed=(3*model.r.fixed[:,None]+np.arange(3)).ravel()
        self.D=np.array([[1.,1.,0.],[-1.,0.,1.]]) if drained else np.array([[1.],[-1.]])
        self.H=np.eye(self.D.shape[1]);self.L=self.D@self.D.T
        self.C=geometry.reference_volume/100.;self.drained=bool(drained);self.jv_calls=0
        self.identity=dict(schema='formal-material-pressure-midpoint-v1',material_model=model.identity,geometry_order=geometry.order,
            control_volumes=[[.125,.5],[.5,.875]],alpha=self.alpha,capacity=self.C.tolist(),D=self.D.tolist(),H=self.H.tolist(),
            transport='constant reference hydraulic network',pressure_boundary=0.,integrator='midpoint with separate material path error')
    def rest(self):
        q=self.model.rest().q;return State(q,q.copy(),np.zeros(2),rule=self.material.order)
    def residual(self,old,q,p,h):
        mid=(old.q+q)/2;d=q-old.q;v=2*d/h-old.velocity;pm=(old.p+p)/2
        mat=self.material.evaluate(mid);geo=self.geometry.chain(old.q,q)
        z=self.D.T@pm;inertia=self.M@((v-old.velocity).ravel())/h
        pressure=self.alpha*np.einsum('cni,c->ni',geo['G'],pm)
        mech=inertia+mat['force'].ravel()-pressure.ravel()
        mass=self.C*(p-old.p)+self.alpha*(geo['V1']-geo['V0'])+h*self.D@z
        residual=np.r_[mech[self.free],mass/h]
        scale=max(float(la.norm(mat['force'])),float(la.norm(inertia)),float(la.norm(pressure)),1e-8)
        return dict(residual=residual,mech=mech,mass=mass,mat=mat,geo=geo,v=v,pm=pm,z=z,
            tolerance=1e-8+1e-3*scale,mass_tolerance=1e-12+1e-3*max(la.norm(self.C*(p-old.p)),la.norm(geo['V1']-geo['V0'])*abs(self.alpha),h*la.norm(self.D@z),1e-12))
    def jvp(self,old,q,p,h,w):
        self.jv_calls+=1;dq=np.zeros_like(q);dq.ravel()[self.free]=w[:-2];dp=w[-2:]
        mat=self.material.evaluate((old.q+q)/2,dq/2);geo=self.geometry.chain(old.q,q,dq)
        pm=(old.p+p)/2
        force=(2*self.M@dq.ravel()/h**2+mat['tangent_action'].ravel()
            -self.alpha*np.einsum('cni,c->ni',geo['dG'],pm).ravel()
            -.5*self.alpha*np.einsum('cni,c->ni',geo['G'],dp).ravel())
        end=self.geometry.evaluate(q)
        mass=self.C*dp/h+self.alpha*np.einsum('cni,ni->c',end['G'],dq)/h+.5*self.L@dp
        return np.r_[force[self.free],mass]
    def preconditioner(self,old,q,h):
        geo=self.geometry.chain(old.q,q);G=geo['G'].reshape(2,-1)[:,self.free]
        E=self.geometry.evaluate(q)['G'].reshape(2,-1)[:,self.free]
        A=2*self.M/h**2+.5*self.K
        J=np.block([[A[np.ix_(self.free,self.free)],-.5*self.alpha*G.T],
                    [self.alpha*E/h,np.diag(self.C/h)+.5*self.L]])
        return la.lu_factor(J)
    def step(self,old,h,target,*,fault=None):
        validate(self.model,old);digest=old.digest();start=time.perf_counter();calls=self.material.calls
        if not np.isfinite(h) or h<=0:raise ValueError('positive timestep required')
        if np.shape(target)!=(len(self.fixed),) or not np.isfinite(target).all():raise ValueError('finite prescribed target required')
        q=old.q.copy();q.ravel()[self.fixed]=target;p=old.p.copy()
        fac=self.preconditioner(old,q,h);trace=[]
        # One block chord prediction, then actual nonlinear residual acceptance.
        t=self.residual(old,q,p,h);delta=la.lu_solve(fac,-t['residual']);q.ravel()[self.free]+=delta[:-2];p+=delta[-2:]
        for it in range(6):
            t=self.residual(old,q,p,h)
            trace.append(dict(iteration=it,force=float(la.norm(t['mech'][self.free])),mass=float(la.norm(t['mass'])),force_tolerance=t['tolerance'],mass_tolerance=t['mass_tolerance']))
            if trace[-1]['force']<=t['tolerance'] and trace[-1]['mass']<=t['mass_tolerance']:break
            if time.perf_counter()>self.material.deadline:raise RuntimeError('formal coupled wall budget exhausted')
            J=LinearOperator((len(self.free)+2,)*2,matvec=lambda w:self.jvp(old,q,p,h,w),dtype=float)
            P=LinearOperator(J.shape,matvec=lambda w:la.lu_solve(fac,w),dtype=float)
            delta,info=gmres(J,-t['residual'],M=P,rtol=.05,atol=1e-10,restart=20,maxiter=1)
            if info:raise RuntimeError('bounded formal Krylov solve failed')
            for k in range(6):
                trial=q.copy();trial.ravel()[self.free]+=delta[:-2]*.5**k;pp=p+delta[-2:]*.5**k
                try:tt=self.residual(old,trial,pp,h)
                except ValueError:continue
                if la.norm(tt['residual'])<la.norm(t['residual']):q,p=trial,pp;break
            else:raise RuntimeError('formal line search failed')
        else:raise RuntimeError('formal nonlinear iteration budget exhausted')
        initial=self.material.evaluate(old.q);final=self.material.evaluate(q)
        d=q-old.q;v=t['v'];T0=.5*old.velocity.ravel()@self.M@old.velocity.ravel();T1=.5*v.ravel()@self.M@v.ravel()
        Ep0=.5*np.dot(self.C*old.p,old.p);Ep1=.5*np.dot(self.C*p,p)
        work=float(t['mech'][self.fixed]@d.ravel()[self.fixed]);diss=h*float(t['z']@t['z'])
        chain=t['geo']['V1']-t['geo']['V0']-np.einsum('cni,ni->c',t['geo']['G'],d)
        path=float(final['U']-initial['U']-np.sum(t['mat']['force']*d))
        residual_work=float(t['mech'][self.free]@d.ravel()[self.free]+t['pm']@t['mass'])
        exchange=float(self.alpha*t['pm']@chain)
        defect=float(T1+final['U']+Ep1-T0-initial['U']-Ep0+diss-work)
        minJ=min(initial['min_detF'],t['mat']['min_detF'],final['min_detF'])
        scale=max(abs(work),abs(final['U']-initial['U']),abs(Ep1-Ep0),diss,1e-10)
        if minJ<=.1 or not np.isfinite(defect) or abs(defect)>1e-12+.01*scale:raise ValueError('formal physical energy/admissibility budget failed')
        if abs(defect-path-residual_work+exchange)>1e-11:raise ValueError('formal energy bookkeeping failed')
        if fault:raise RuntimeError('injected before formal coupled commit')
        assert old.digest()==digest
        new=State(q.copy(),v.copy(),p.copy(),old.time+h,old.step+1,old.boundary_volume+h*float(np.sum(self.D@t['z'])),
            old.dissipation+diss,old.external_work+work,self.material.order,old.rule_work);validate(self.model,new)
        row=dict(step=new.step,time=new.time,drained=self.drained,alpha=self.alpha,pressure=p,volume=t['geo']['V1'],volume_change=t['geo']['V1']-t['geo']['V0'],
            flux=t['z'],boundary_volume=new.boundary_volume,mass_defect=t['mass'],volume_chain_defect=chain,exchange_work= self.alpha*float(t['pm']@(t['geo']['V1']-t['geo']['V0'])),
            kinetic=T1,material_U=final['material_U'],stabilization_U=final['stabilization_U'],storage_U=Ep1,darcy_dissipation=diss,
            boundary_work=work,energy_defect=defect,path_error=path,residual_work=residual_work,exchange_defect=exchange,
            ledger_closure=defect-path-residual_work+exchange,minJ=minJ,iterations=trace,material_calls=self.material.calls-calls,
            jv_calls=self.jv_calls,reaction_fixed=t['mech'][self.fixed].reshape(-1,3),seconds=time.perf_counter()-start)
        return new,row
