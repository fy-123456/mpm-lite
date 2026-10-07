"""Full-tensor, zero-source fixed-skeleton reference; explicit vector initial data."""
import numpy as np
import scipy.linalg as la
from benchmarks.research_transverse_next.static import cpu_rest_H
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology
from engine.aniso_phase1.research_pressure_startup_next.theta import schedule, step


def initial(p, cells):
    p = np.array(p, dtype=float, copy=True)
    if p.shape != (cells,) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('finite nonnegative cell-pressure vector required')
    return p


def algebra(cuts, mobility, storage, reservoir, *, source=0.):
    if source != 0 or not np.isfinite(storage) or storage <= 0:
        raise ValueError('positive storage and explicit zero source required')
    top = ReferenceTopology(cuts)
    H = cpu_rest_H(top, mobility)
    symmetry = np.linalg.norm(H-H.T)/np.linalg.norm(H)
    if symmetry > 1e-12:
        raise ValueError('nonsymmetric mobility assembly')
    factor = la.cho_factor(H)
    Z = la.cho_solve(factor, top.B.T)
    gb = top.boundary_term(reservoir)
    z0 = -la.cho_solve(factor, gb)
    C = storage*top.V0
    L = top.B@Z
    rhs = -top.B@z0
    D = np.sqrt(C)
    A = L/D[:, None]/D[None, :]
    asym = np.linalg.norm(A-A.T)/np.linalg.norm(A)
    if asym > 1e-12:
        raise ValueError('nonsymmetric pressure generator')
    lam, Q = la.eigh((A+A.T)*.5)
    if lam[0] <= 0:
        raise ValueError('drained generator has nonpositive spectrum')
    pe = la.solve(A, rhs/D, assume_a='pos')/D
    return dict(top=top, H=H, Z=Z, z0=z0, gb=gb, C=C, L=L, rhs=rhs,
                D=D, A=A, lam=lam, Q=Q, pe=pe, symmetry=asym)


def exact(a, p0, times):
    p0 = initial(p0, len(a['C']))
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(t < 0) or np.any(np.diff(t) <= 0):
        raise ValueError('increasing nonnegative reference times required')
    coeff = a['Q'].T@(a['D']*(p0-a['pe']))
    exponent = t[:, None]*a['lam']
    p = a['pe']+(np.exp(-exponent)*coeff)@a['Q'].T/a['D']
    # expm1 remains accurate as lambda*t approaches zero; eigenvalues are positive.
    integral = t[:, None]*a['pe']+(-np.expm1(-exponent)/a['lam']*coeff)@a['Q'].T/a['D']
    cumulative = integral@a['Z'].T+t[:, None]*a['z0']
    return dict(pressure=p, cumulative=cumulative,
                flux=np.diff(cumulative, axis=0)/np.diff(t)[:, None])


def integrate(a, p0, times):
    t = np.asarray(times, dtype=float)
    theta = schedule(t, 'startup')
    p = initial(p0, len(a['C']))
    cumulative = np.zeros(a['top'].nflux)
    ps, qs, zs, rows = [p.copy()], [cumulative.copy()], [], []
    for h, th in zip(np.diff(t), theta):
        p, z, row = step(a, p, h, float(th))
        cumulative += h*z
        ps.append(p.copy()); qs.append(cumulative.copy()); zs.append(z); rows.append(row)
    return dict(pressure=np.array(ps), cumulative=np.array(qs), flux=np.array(zs), ledger=rows, theta=theta)


def mode_increment_check(actual, reference):
    a = np.asarray(actual)-actual[0]
    b = np.asarray(reference)-reference[0]
    signal = float(np.max(abs(b)))
    error = float(np.max(abs(a-b)))
    budget = 1e-5+.1*signal
    return dict(error_Pa=error, signal_Pa=signal, budget_Pa=budget,
                passed=error<=budget, signal_resolved=signal>5e-5)
