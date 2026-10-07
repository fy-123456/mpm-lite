"""Constant SPD hydraulic resistance via exact flux whitening.

H=Lh Lh^T, E=D Lh^-T, y=Lh^T z. Parent algebra uses y and E:
E y = D z; ||y||^2 = z^T H z. This keeps residual/Jv/preconditioner and
energy identical without duplicating or editing the sealed parent solver.
"""
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.research_formal_pressure_next.solver import Solver as ParentSolver

class Solver(ParentSolver):
    def __init__(self,model,geometry,material,*,drained=False,alpha=1.,resistance=1.,storage_modulus=100.):
        if not np.isfinite(alpha) or alpha<0:raise ValueError('nonnegative finite alpha required')
        if not np.isfinite(storage_modulus) or storage_modulus<=0:raise ValueError('positive storage modulus required')
        super().__init__(model,geometry,material,drained=drained,alpha=alpha)
        self.Dphysical=self.D.copy();n=self.D.shape[1];H=np.asarray(resistance,float)
        if H.ndim==0:H=np.eye(n)*H
        elif H.ndim==1:H=np.diag(H)
        if H.shape!=(n,n) or not np.isfinite(H).all() or not np.allclose(H,H.T,rtol=1e-12,atol=1e-14):raise ValueError('finite symmetric branch resistance required')
        try:self.chol=la.cholesky(H,lower=True)
        except la.LinAlgError as e:raise ValueError('positive definite resistance required') from e
        self.H=H.copy();self.D=la.solve_triangular(self.chol,self.Dphysical.T,lower=True).T
        self.L=self.D@self.D.T;self.C=geometry.reference_volume/storage_modulus
        self.identity.update(schema='material-coupling-budgeted-v1',H=self.H.tolist(),D=self.Dphysical.tolist(),capacity=self.C.tolist(),
            flux_coordinates='reported physical z; internal exact Cholesky-whitened y',pressure_boundary=0.)
    def residual(self,old,q,p,h):
        self.geometry.budget.check();out=super().residual(old,q,p,h)
        # Diagnostic only: preserve registered acceptance tolerances.
        f=out['mat']['force'].ravel()[self.free]
        inertia=(self.M@((out['v']-old.velocity).ravel())/h)[self.free]
        pressure=self.alpha*np.einsum('cni,c->ni',out['geo']['G'],out['pm']).ravel()[self.free]
        out['free_scale']=max(float(la.norm(f)),float(la.norm(inertia)),float(la.norm(pressure)),1e-8)
        return out
    def step(self,old,h,target,*,fault=None):
        before=self.jv_calls;self.material.deadline=self.material.budget.deadline
        new,row=super().step(old,h,target,fault=fault)
        y=np.asarray(row['flux']);z=la.solve_triangular(self.chol.T,y,lower=False)
        row.update(flux_whitened=y,flux=z,jv_calls_step=self.jv_calls-before,jv_calls_cumulative=self.jv_calls,
            hydraulic_resistance=self.H,physical_dissipation_check=h*float(z@self.H@z))
        if not np.isclose(row['darcy_dissipation'],row['physical_dissipation_check'],rtol=1e-8,atol=1e-14):raise ValueError('hydraulic dissipation mismatch')
        self.material.budget.check();return new,row
