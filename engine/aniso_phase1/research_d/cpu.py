"""Fixed-preconditioner PCG with independently recomputed stopping residuals."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import time
import numpy as np


@dataclass
class SolveInfo:
    status: str
    iterations: int
    true_residual: float
    target: float
    relative_residual: float
    recursive_residuals: list = field(default_factory=list)
    true_residuals: list = field(default_factory=list)
    matvec_calls: int = 0
    preconditioner_calls: int = 0
    negative_curvature: bool = False
    seconds: float = 0.
    diagnostics: dict = field(default_factory=dict)

    @property
    def converged(self):
        return self.status == 'converged' and self.true_residual <= self.target

    def record(self):
        return dict(asdict(self), converged=self.converged)


def pcg(apply, b, precondition=None, *, rtol=1e-7, atol=1e-12, maxiter=2000,
        x0=None, check_every=20):
    """No physical shift; a non-SPD search direction fails with a reason.

    Recompute b-Ax at convergence and periodically. Residual replacement also
    restarts conjugacy, so recursive drift never authorizes a false success.
    The preconditioner MUST remain fixed, linear and SPD during this call.
    """
    start = time.perf_counter()
    b = np.asarray(b, dtype=np.float64)
    if b.ndim != 1 or not np.isfinite(b).all(): raise ValueError('finite vector required')
    if not np.isfinite([rtol,atol]).all() or rtol < 0 or atol < 0 or max(rtol,atol) == 0:
        raise ValueError('positive stopping tolerance required')
    if maxiter < 0 or check_every < 1: raise ValueError('invalid iteration budget')
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float, copy=True)
    if x.shape != b.shape or not np.isfinite(x).all(): raise ValueError('invalid initial guess')
    M = (lambda v:v.copy()) if precondition is None else precondition
    normb = float(np.linalg.norm(b)); target = max(atol,rtol*normb)
    calls, mcalls, its = 1, 0, 0
    r = b-apply(x); error = float(np.linalg.norm(r))
    trace = [error]; true = [dict(iteration=0, norm=error)]
    status = 'iteration_limit'; negative = False
    if error <= target: status = 'converged'
    elif not np.isfinite(error): status = 'nonfinite_residual'
    else:
        z = M(r); mcalls += 1; rz = float(r@z); p = z.copy()
        for i in range(1,maxiter+1):
            if not np.isfinite(rz) or rz <= 0:
                status = 'invalid_preconditioner'; break
            Ap = apply(p); calls += 1; curvature = float(p@Ap)
            if not np.isfinite(curvature): status = 'nonfinite_curvature'; break
            if curvature <= 0:
                status = 'nonpositive_curvature'; negative = True; break
            alpha = rz/curvature; x += alpha*p; r -= alpha*Ap; its = i
            error = float(np.linalg.norm(r)); trace.append(error)
            if not np.isfinite(error): status = 'nonfinite_residual'; break
            restart = False
            if error <= target or i % check_every == 0:
                exact = b-apply(x); calls += 1; error = float(np.linalg.norm(exact))
                true.append(dict(iteration=i,norm=error))
                if error <= target: status = 'converged'; break
                # Periodic observations need not destroy PCG's conjugacy.
                if np.linalg.norm(exact-r) > .1*max(error,target):
                    r = exact; restart = True
            z = M(r); mcalls += 1; rz_new = float(r@z)
            p = z.copy() if restart else z+(rz_new/rz)*p
            rz = rz_new
    exact = b-apply(x); calls += 1; error = float(np.linalg.norm(exact))
    true.append(dict(iteration=its,norm=error))
    if not np.isfinite(error): status = 'nonfinite_residual'
    elif status == 'converged' and error > target: status = 'true_residual_failure'
    return x, SolveInfo(status,its,error,target,error/max(normb,atol,np.finfo(float).tiny),
                       trace,true,calls,mcalls,negative,time.perf_counter()-start)
