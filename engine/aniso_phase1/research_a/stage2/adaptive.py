"""Bounded local residual enrichment, using the original v22 potential."""
from __future__ import annotations
import copy
from time import monotonic
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from engine.aniso_phase1.high_order_space import BoxElastic, local_box
from ..estimators import correction_score, ranked_indices
from ..support_family import deterministic_order


from ..adaptive import extend_basis


def enrich(problem, definitions, estimator="stress-correction", rounds=6,
           patches_per_round=8, budget=144, callback=None, fixed_sequence=None):
    if rounds <= 0 or patches_per_round <= 0 or budget <= 0:
        raise ValueError("Positive rounds, patches per round, and budget required")
    if estimator not in ("stress-correction", "energy", "residual", "geometric"):
        raise ValueError("Only reference-free estimators are accepted here")
    n = problem.n
    H = problem.case.H
    op = BoxElastic(problem.edges, problem.degree, H)
    local_ops = []
    started = monotonic()
    for definition in definitions:
        idx, local, meta = local_box(problem.edges, problem.degree, definition, H)
        squared = copy.copy(local)
        squared.H = (H.T@H).reshape(3, 3, 3, 3)
        ids = np.concatenate([idx+j*n for j in range(3)])
        local_ops.append((idx, ids, local, squared, meta))
    setup_seconds = monotonic()-started
    W = np.empty((n, 0))
    raw_all = sp.csr_matrix((n, 0))
    transform = np.empty((0, 0))
    u = problem.initial_u.copy()
    previous_energy = np.inf
    records = []
    for step in range(rounds):
        t = monotonic()
        force = op.apply(u.T.ravel())
        scores, corrections, checks = [], [], []
        # Geometry control uses the same local solve set, making the observed
        # solve costs comparable. Selection never uses these scores.
        for idx, ids, local, squared, meta in local_ops:
            rhs = -force[ids]
            w, check = local.correction(rhs, rtol=2e-8)
            scores.append(correction_score(rhs, w, "energy" if estimator == "geometric" else estimator,
                                           stress_action=squared.free_apply))
            corrections.append(w)
            checks.append(check)
        local_seconds = monotonic()-t
        order = deterministic_order(definitions, step) if estimator == "geometric" else ranked_indices(scores)
        chosen = np.asarray(fixed_sequence[step], dtype=int) if fixed_sequence is not None else order[:patches_per_round]
        if len(chosen) != patches_per_round or np.any(chosen < 0) or np.any(chosen >= len(local_ops)):
            raise ValueError("Invalid frozen support sequence")
        columns = []
        for k in chosen:
            idx = local_ops[k][0]
            w = corrections[k].reshape(3, -1).T
            U, singular, _ = la.svd(w, full_matrices=False)
            rank = int(np.sum(singular > max(singular[0]*1e-10, 1e-18))) if len(singular) else 0
            for j in range(rank):
                col = np.zeros(n)
                col[idx] = U[:, j]
                columns.append(col)
        if not columns:
            raise RuntimeError("Local residuals produced no independent functions")
        t = monotonic()
        W, raw_all, transform, added = extend_basis(W, raw_all, transform, np.column_stack(columns), budget-W.shape[1])
        if added == 0:
            raise RuntimeError("No independent functions added before the prescribed final round")
        basis_seconds = monotonic()-t
        t = monotonic()
        field, equilibrium = problem.equilibrate(W)
        global_seconds = monotonic()-t
        u = field["u"]
        invariant = problem.invariant_errors(W)
        if invariant["local_fixed_grip_value"] > 1e-10 or invariant["orthogonality_error"] > 1e-7:
            raise RuntimeError("Invalid support trace or linearly dependent basis")
        energy = equilibrium["energy_J"]
        if energy > previous_energy+max(abs(previous_energy)*1e-7, 1e-12):
            raise RuntimeError("Nested-space energy increased after re-equilibration")
        previous_energy = energy
        record = dict(round=step+1, estimator=estimator, scalar_local_dofs=W.shape[1], new_scalar_dofs=added,
                      target_scalar_budget=budget, chosen_patches=chosen.tolist(), scores=scores,
                      local_checks=checks, equilibrium=equilibrium, invariants=invariant,
                      case_sha256=problem.case.signature, fixed_sequence=fixed_sequence is not None,
                      max_local_vector_dofs=max(len(item[1]) for item in local_ops),
                      total_local_vector_dofs=sum(len(item[1]) for item in local_ops),
                      sparse_raw_nnz=raw_all.nnz, dense_basis_bytes=W.nbytes,
                      cost=dict(setup_seconds=setup_seconds if step == 0 else 0., local_solve_seconds=local_seconds,
                                basis_seconds=basis_seconds, global_balance_seconds=global_seconds),
                      reference_field_used=False)
        records.append(record)
        if callback:
            callback(record, field, raw_all, transform)
    if W.shape[1] != budget:
        raise RuntimeError(f"Prescribed scalar budget not reached: {W.shape[1]} != {budget}")
    reconstruction = float(la.norm(raw_all@transform-W)/max(la.norm(W), 1e-30))
    if reconstruction > 1e-7:
        raise RuntimeError("Basis archive cannot reproduce the selected space")
    return dict(W=W, raw=raw_all, transform=transform, field=field, records=records,
                basis_reconstruction_relative=reconstruction)
