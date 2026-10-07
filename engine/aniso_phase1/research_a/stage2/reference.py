"""Bounded conforming reference solves and regional engineering diagnostics."""
import time
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, cg
from engine.aniso_phase1.tensor_reference import TensorElastic, coordinates
from .metrics import compare_fields


def estimate(edges, degree, scalar_columns=0):
    nodes = int(np.prod([degree*(len(e)-1)+1 for e in edges]))
    active = np.count_nonzero((np.asarray(edges[0]) >= .25) & (np.asarray(edges[0]) <= .75))-1
    free = int(3*(degree*active-1)*np.prod([degree*(len(e)-1)+1 for e in edges[1:]]))
    peak = nodes*(3*8*40+8*scalar_columns*5)
    return dict(nodes=nodes, free_vector_dofs=free, vector_bytes=3*nodes*8,
                assembled_nnz_upper=9*nodes*(2*degree+1)**3, assembly_used=False,
                predicted_peak_bytes=peak, output_bytes=nodes*3*8,
                within_budget=bool(nodes <= 14000000 and peak <= 24*1024**3))


def pilot_edges(case):
    return case.map_edges([np.linspace(.125, .875, 25), np.linspace(.375, .625, 9), np.linspace(.375, .625, 9)])


def solve(case, edges, degree, rtol=2e-8, max_seconds=3600):
    resource = estimate(edges, degree)
    if not resource['within_budget']: raise RuntimeError('Reference exceeds memory/node budget')
    start = time.monotonic()
    active = [np.asarray(edges[0])[(np.asarray(edges[0]) >= .25) & (np.asarray(edges[0]) <= .75)], *edges[1:]]
    if len(active[0]) < 2 or active[0][0] != .25 or active[0][-1] != .75:
        raise ValueError('Both original grip transitions must exist')
    op = TensorElastic(active, degree, case.H)
    X = np.stack(np.meshgrid(*coordinates(active, degree), indexing='ij'), axis=-1).reshape(-1, 3)
    u = case.boundary(X).reshape(*op.shape, 3).transpose(3, 0, 1, 2).copy()
    rhs = -op.apply(u).reshape(3, *op.shape)[:, 1:-1].ravel()
    A = LinearOperator((len(rhs),)*2, matvec=op.free_apply, dtype=float)
    M = LinearOperator(A.shape, matvec=op.precondition, dtype=float)
    iterations = 0
    def callback(_):
        nonlocal iterations
        iterations += 1
        if time.monotonic()-start > max_seconds: raise TimeoutError('Reference wall-time limit')
    v, info = cg(A, rhs, M=M, rtol=rtol, atol=1e-13, maxiter=2000, callback=callback)
    u[:, 1:-1] = v.reshape(3, *op.free_shape)
    force = op.apply(u).reshape(3, *op.shape)
    residual = float(la.norm(force[:, 1:-1])/max(la.norm(rhs), 1e-12))
    axes = coordinates(edges, degree)
    full_X = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    full = case.boundary(full_X).reshape(*map(len, axes), 3)
    where = (axes[0] >= .25) & (axes[0] <= .75)
    full[where] = u.transpose(1, 2, 3, 0)
    reaction = force[:, -1].sum(axis=(1, 2))
    tipX = X.reshape(*op.shape, 3)[-1].reshape(-1, 3)
    moment = np.cross(tipX-case.rotation_center, force[:, -1].reshape(3, -1).T).sum(0)
    energy = .5*float(np.sum(u*force))
    work = float(np.sum(case.boundary(X).reshape(*op.shape, 3).transpose(3, 0, 1, 2)*force))
    passed = bool(info == 0 and residual < 1e-6 and np.isfinite(full).all())
    if not passed: raise RuntimeError(f'Reference failed: info={info}, residual={residual}')
    # Evaluate reactions over every constrained row, including the rigid volumes.
    # Exact finite grip rotation has a small symmetric part in this linear model.
    from engine.aniso_phase1.high_order_space import BoxElastic
    full_op = BoxElastic(edges, degree, case.H)
    full_force = full_op.apply(full.reshape(-1,3).T.ravel()).reshape(3,-1).T
    boundary = case.boundary(full_X)
    right = full_X[:,0] >= .75-1e-12
    reaction = full_force[right].sum(0)
    moment = np.cross(full_X[right]-case.rotation_center, full_force[right]).sum(0)
    full_energy = .5*float(np.sum(full.reshape(-1,3)*full_force))
    full_work = float(np.sum(boundary*full_force))
    rigid_energy = full_energy-energy
    return full.reshape(-1, 3), dict(case_sha256=case.signature, degree=degree,
        iterations=iterations, relative_residual=residual, energy_J=full_energy,
        reaction_vector_N=reaction.tolist(), reaction_moment_Nm=moment.tolist(),
        free_span_work_identity_relative=abs(2*energy-work)/max(abs(2*energy), 1e-12),
        rigid_volume_constant_energy_J=rigid_energy,
        complete_domain_work_identity_relative=abs(2*full_energy-full_work)/max(abs(2*full_energy),1e-12),
        seconds=time.monotonic()-start, resource=resource, solve_passed=passed,
        stabilization='full FE physical limit; no carrier patch or artificial stiffness',
        continuum_certified=False, boundary='original rigid volumes restored over the complete physical domain')


def compare(case, a, b):
    result = compare_fields(a, b, case.H, case.fiber)
    for r in result['regions'].values():
        # Positive scales are predeclared physical scales; always retain absolute errors.
        r['stress_engineering_passed'] = r['stress_absolute_rms_Pa'] <= .005*max(r['stress_reference_rms_Pa'] or 0., 1e-3)
        r['fiber_engineering_passed'] = r['fiber_strain_absolute_rms'] <= .005*max(r['fiber_reference_rms'] or 0., 1e-5)
    result['all_regions_engineering_passed'] = all(r['stress_engineering_passed'] and r['fiber_engineering_passed'] for r in result['regions'].values())
    return result
