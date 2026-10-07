"""General mixed solve with original residual verification, without SPD claims."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, gmres


def solve_mixed(matrix, rhs, *, method='direct', tolerance=1e-8):
    A = sp.csc_matrix(matrix)
    b = np.asarray(rhs, dtype=float)
    if A.shape != (len(b), len(b)) or not np.isfinite(A.data).all() or not np.isfinite(b).all():
        raise ValueError('finite square mixed system required')
    if method == 'direct':
        x = splu(A).solve(b)
    elif method == 'gmres':
        x, flag = gmres(A, b, atol=1e-12, rtol=tolerance, maxiter=500)
        if flag:raise ValueError('mixed GMRES did not converge')
    else:
        raise ValueError('mixed system requires an explicitly supported general solver')
    residual = float(np.linalg.norm(A @ x-b)/max(np.linalg.norm(b), 1e-12))
    if not np.isfinite(x).all() or residual > tolerance:
        raise ValueError('mixed true residual failed')
    return x, dict(method=method, residual=residual, spd_assumed=False)


from ..research_e.poro import Biot, PoroState, PoroStep


class GeneralBiot(Biot):
    """Existing E equations and ledger using the new general mixed backend."""
    def step(self, old, dt, load, boundary, source=0., method='monolithic', **kwargs):
        if method != 'monolithic':
            return super().step(old,dt,load,boundary,source,method=method,**kwargs)
        M,b,free,frhs,src,load=self._system(old,dt,load,boundary,source)
        x,info=solve_mixed(M,b)
        s,g=self.solid,self.flow.grid;nu=len(s.free);nc=g.nc
        u=np.zeros(s.ndof);u[s.free]=x[:nu]
        p=x[nu:nu+nc].copy();q=np.zeros(g.nf);q[free]=x[nu+nc:]
        state=PoroState(u,p,old.time+dt);du=u-old.u;dp=p-old.p
        mass=s.G @ du+self.C @ dp+dt*self.flow.B @ q-dt*g.volume*src
        loss=dt*float(q @ (self.flow.H @ q))
        numerical=.5*float(du @ (s.A @ du)+dp @ (self.C @ dp))
        work=float(load @ du+dt*(p @ (src*g.volume)+q @ frhs))
        balance=self.energy(state)-self.energy(old)+loss+numerical-work
        mech=s.A @ u-s.G.T @ p-load
        reactions=np.zeros(s.ndof);reactions[s.fixed]=mech[s.fixed]
        metrics=dict(true_residual=info['residual'],mass_defect=float(abs(mass.sum())),
            local_mass_defect=float(np.max(abs(mass))),energy_residual=balance,
            physical_dissipation=loss,numerical_loss=numerical,external_work=work,
            elastic_energy=.5*float(u @ (s.A @ u)),storage_energy=.5*float(p @ (self.C @ p)),
            kinetic_energy=0.,momentum_balance=float(np.max(abs((load+reactions).reshape(-1,g.dim).sum(axis=0)))),
            iterations=1,boundary_outflow_volume=float(dt*np.sum(self.flow.B @ q)),
            fluid_content_change=float(np.sum(s.G @ du+self.C @ dp)))
        return PoroStep(state,q,metrics,bool(loss>=-1e-10 and np.isfinite(balance)))
