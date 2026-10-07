"""CPU midpoint dry-solid qualification in the original material coordinates.

Full nonlinear potential is evaluated by the injected authenticated operator.
Rest stiffness is a chord search matrix only. No pressure coupling or APIC.
"""
import numpy as np
import scipy.linalg as la
from .state import State

class DryMidpoint:
    def __init__(self,model,evaluate):
        self.model=model;self.evaluate=evaluate
        self.M=np.kron(model.r.M,np.eye(3));self.K=model.r.K
        self.free=model.r.ids
        self.fixed=(3*model.r.fixed[:,None]+np.arange(3)).ravel()
    def step(self,state,dt,target_fixed,*,force_atol=1e-8,force_rtol=1e-3,max_checks=3,fault=None):
        m=self.model;m.validate(state);digest=state.digest()
        if not np.isfinite(dt) or dt<=0:raise ValueError('positive timestep required')
        target=np.asarray(target_fixed,float)
        if target.shape!=(len(self.fixed),) or not np.isfinite(target).all():raise ValueError('invalid boundary target')
        q=state.q.ravel();v=state.velocity.ravel();d=np.zeros_like(q);d[self.fixed]=target-q[self.fixed]
        A=2*self.M/dt**2+.5*self.K;factor=la.cho_factor(A[np.ix_(self.free,self.free)])
        rhs=2*self.M@v/dt-self.K@q
        d[self.free]=la.cho_solve(factor,rhs[self.free]-A[np.ix_(self.free,self.fixed)]@d[self.fixed])
        for check in range(max_checks):
            mid=self.evaluate((q+.5*d).reshape(state.q.shape));f=mid['force'].ravel()
            v1=2*d/dt-v;res=self.M@(v1-v)/dt+f
            scale=max(float(np.linalg.norm(f)),float(np.linalg.norm(self.M@(v1-v)/dt)),1e-8)
            tolerance=force_atol+force_rtol*scale
            if np.linalg.norm(res[self.free])<=tolerance:break
            if check==max_checks-1:raise RuntimeError('nonlinear dry midpoint budget exhausted')
            d[self.free]+=la.cho_solve(factor,-res[self.free])
        initial=self.evaluate(state.q);final=self.evaluate((q+d).reshape(state.q.shape))
        if min(mid['min_detF'],initial['min_detF'],final['min_detF'])<=.1:raise ValueError('inadmissible nonlinear path')
        T0=.5*v@self.M@v;T1=.5*v1@self.M@v1
        boundary_work=float(res[self.fixed]@d[self.fixed]);residual_work=float(res[self.free]@d[self.free])
        path_error=float(final['U']-initial['U']-f@d)
        defect=float(T1+final['U']-T0-initial['U']-boundary_work)
        if abs(defect)>1e-12+.01*max(abs(boundary_work),abs(final['U']-initial['U']),1e-10):raise ValueError('dry physical energy budget exceeded')
        if abs(defect-path_error-residual_work)>1e-11:raise ValueError('dry ledger identity failed')
        candidate=State((q+d).reshape(state.q.shape),v1.reshape(state.velocity.shape),state.time+dt,state.step+1)
        m.validate(candidate)
        if fault:raise RuntimeError('injected before dry commit')
        assert state.digest()==digest
        return candidate,dict(time=candidate.time,step=candidate.step,checks=check+1,true_free_force_residual=float(np.linalg.norm(res[self.free])),
            force_tolerance=tolerance,minJ=final['min_detF'],kinetic=T1,material_U=final.get('material_U'),stabilization_U=final.get('stabilization_U'),
            potential=final['U'],boundary_work=boundary_work,path_error=path_error,residual_work=residual_work,energy_defect=defect,
            ledger_closure=defect-path_error-residual_work,reaction_fixed=res[self.fixed].reshape(-1,3).tolist(),pressure_enabled=False)
