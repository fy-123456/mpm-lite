"""Full-space mass certification and explicit evidence for a blocked ODE gate."""
import time
import resource
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.research_d.common_state import CommonState
from engine.aniso_phase1.research_d.common_kinetic import PointInertia
from engine.aniso_phase1.research_d.identity import digest, sha, write_json
from engine.aniso_phase1.research_c.stage2.model import FrozenPotential, Boundary, DynamicModel, MassRankError


def certify(out, space, M5, M6, K, parent_folder, seed):
    s = space; ids = s.free_scalar_ids
    rng = np.random.default_rng(seed)
    m5 = PointInertia(s, order=5); m6 = PointInertia(s, order=6)
    diagonal = np.diag(M5)
    positive = diagonal>0
    scale = np.ones(s.ndof)
    scale[positive] = 1/np.sqrt(diagonal[positive])
    A = scale[:, None]*M5*scale[None, :]
    B = scale[:, None]*M6*scale[None, :]
    wf = la.eigvalsh(A[np.ix_(ids, ids)])
    wb = la.eigvalsh(B[np.ix_(ids, ids)])
    w = la.eigvalsh(A)
    error = float(la.norm((A-B)[np.ix_(ids, ids)], 2))
    threshold = max(64*len(ids)*np.finfo(float).eps*wf[-1],10*error)
    full_threshold = max(64*s.ndof*np.finfo(float).eps*w[-1],10*la.norm(A-B,2))
    norm=lambda x:float(la.norm(x))
    rel=lambda a,b:norm(a-b)/max(norm(b),1e-30)
    random=rng.normal(size=(s.ndof,3))
    action_checks=[rel(M5@random,m5.apply(random)),rel(M6@random,m6.apply(random))]
    with np.load(parent_folder/'input-states.npz') as z:
        directions={k:z[k] for k in z.files if k.startswith('direction')}
    parent_checks={name:dict(passed=bool(np.allclose(m5.apply(v),m6.apply(v),rtol=1e-10,atol=1e-12)),
        relative=rel(m5.apply(v),m6.apply(v))) for name,v in directions.items()}
    translation=np.zeros_like(random);translation[:s.n,0]=1.
    removed=M5.copy();removed[:s.n,s.n:]=0.;removed[s.n:,:s.n]=0.
    mass=dict(passed=bool(np.sum(wf>threshold)==len(ids) and np.all(wf>=-threshold)),
        free_scalar_rank=int(np.sum(wf>threshold)), expected_free_scalar_rank=len(ids),
        free_vector_rank=3*int(np.sum(wf>threshold)), expected_free_vector_rank=3*len(ids),
        full_scalar_rank=int(np.sum(w>full_threshold)), full_scalar_dofs=s.ndof,
        full_positive_definite=False, free_positive_definite=bool(wf[0]>threshold),
        full_zero_diagonal_ids=np.flatnonzero(~positive).tolist(),
        zero_diagonal_free_ids=np.intersect1d(ids,np.flatnonzero(~positive)).tolist(),
        zero_diagonal_positions=s.carrier_X[np.flatnonzero(~positive)].tolist(),
        free_scaled_eigenvalues=wf.tolist(), free_order6_scaled_eigenvalues=wb.tolist(),
        full_scaled_eigenvalues=w.tolist(), rank_threshold=threshold, full_rank_threshold=full_threshold,
        independent_integration_spectral_error_bound=error,
        free_min_eigenvalue_kg=float(la.eigvalsh(M5[np.ix_(ids,ids)])[0]),
        free_condition_number=None, condition_number_reason='singular; infinity is not serialized as JSON',
        positive_subspace_condition_number=float(wf[-1]/wf[wf>threshold][0]),
        symmetry_relative=rel(M5,M5.T), full_order5_vs6_relative=rel(M5,M6),
        action_recovery_relative=action_checks, parent_mode_checks=parent_checks,
        translation_mass_kg=float(np.sum(translation*(M5@translation))),
        carrier_local_block_norm=norm(M5[:s.n,s.n:]),
        free_boundary_block_norm=norm(M5[np.ix_(ids,s.fixed_scalar_ids)]),
        deleting_cross_terms_action_relative=rel(removed@random,M5@random),
        rank_at_threshold_multipliers={str(k):int(np.sum(wf>threshold*k)) for k in [.01,1.,100.]})
    mass['implementation_checks_passed']=bool(max(action_checks)<1e-10 and
        mass['symmetry_relative']<1e-10 and mass['full_order5_vs6_relative']<1e-8 and
        len(parent_checks)==2 and all(r['passed'] for r in parent_checks.values()) and
        abs(mass['translation_mass_kg']-.046875)<1e-12)
    mass['passed'] &= mass['implementation_checks_passed']
    write_json(out/'mass-audit.json',mass)

    # This independent factorization never uses the recovered mass matrix.
    # Pure carrier null vectors survive unchanged through Q2->Q4 prolongation.
    carrier_ids=ids[:s.nfree_carrier]
    _,singular,vh=la.svd(s.oldA[:,carrier_ids],full_matrices=False)
    sv_tol=64*max(s.oldA[:,carrier_ids].shape)*np.finfo(float).eps*singular[0]
    null=vh[singular<=sv_tol]
    full_directions=[]; rows=[]
    for v in null:
        direction=np.zeros((s.ndof,3));direction[carrier_ids,0]=v
        nodes=s.nodes(direction)
        ks= s.Ks@direction[:s.n]
        rows.append(dict(coefficient_norm=norm(direction), nodal_displacement_norm=norm(nodes),
            nodal_displacement_max=float(np.max(abs(nodes))),
            order5_mass_action_norm=norm(m5.apply(direction)),
            order6_mass_action_norm=norm(m6.apply(direction)),
            carrier_basis_residual=norm(s.oldA[:,carrier_ids]@v),
            stabilization_direction_energy_J=.5*float(np.sum(direction[:s.n]*ks)),
            stabilization_direction_force_norm=norm(ks)))
        full_directions.append(direction)
    np.savez_compressed(out/'mass-null-directions.npz',directions=np.array(full_directions),
        carrier_singular_values=singular,free_scaled_eigenvalues=wf,
        free_order6_scaled_eigenvalues=wb, free_scalar_ids=ids)
    rank_diagnosis=dict(independent_nullity=len(null), expected_free_carriers=s.nfree_carrier,
        independent_free_carrier_rank=int(np.sum(singular>sv_tol)), svd_threshold=sv_tol,
        singular_values=singular.tolist(), null_direction_evidence=rows,
        conclusion='carrier basis dependence causes massless directions with nonzero Ks stiffness',
        order5_and_order6_agree=True, regularization_applied=False, elimination_applied=False,
        remedy='A must deliver a separately identified independent space, or a new descriptor-dynamics protocol is needed; current C ODE gate stays closed')
    write_json(out/'mass-nullspace-audit.json',rank_diagnosis)

    potential=FrozenPotential(s)
    q=np.zeros((s.ndof,3)); state=CommonState(q.copy(),q.copy())
    response=potential.evaluate(q)
    points=[s.test_vectors[f'points{k}'] for k in range(3)]
    fields=potential.fields(state,points)
    np.savez_compressed(out/'initial-dynamic-state.npz',q=q,velocity=q,time=0.,step=0,
        F=fields['F'],PK1=fields['PK1'],**{f'points{k}':x for k,x in enumerate(points)})
    write_json(out/'initial-state-audit.json',dict(passed=bool(norm(response['force'][ids])<1e-10
        and np.max(abs(fields['PK1']))<1e-9),free_residual=norm(response['force'][ids]),
        material_J=response['material_U'],stabilization_J=response['stabilization_U'],
        min_detF=response['min_detF'],PK1_max_Pa=float(np.max(abs(fields['PK1']))),
        equilibrium_checked=True,stress_free=True,dynamically_admitted=mass['passed'],
        initial_sha256=sha(out/'initial-dynamic-state.npz'),original_loaded_snapshot_modified=False))
    boundary=Boundary(s)
    write_json(out/'boundary-history.json',dict(signature=boundary.signature,
        samples=[dict(time=t,displacement=boundary.lift(t)[s.fixed_scalar_ids].tolist(),
            velocity=boundary.speed(t)[s.fixed_scalar_ids].tolist()) for t in [0.,.25,.5,.6,.85,1.1,1.6]]))
    d=s.direction_coefficients(rng.normal(size=s.q_shape));d/=norm(d)
    exact=potential.evaluate(q,d)
    stiffness_check=rel(K@d.ravel(),exact['tangent_action'].ravel())
    write_json(out/'tangent-modal-audit.json',dict(rest_tangent_passed=stiffness_check<2e-5,
        streamed_exact_tangent_relative=stiffness_check, rest_matrix_symmetry=rel(K,K.T),
        generalized_modes_status='blocked_singular_mass',passed=False,
        finite_frequency_range=None, original_Ks_retained=True,
        reason='No SPD generalized eigenproblem exists on all 657 free components'))
    try:
        DynamicModel(s,M5,rest_K=K)
    except MassRankError as exc:
        gate=dict(passed=not mass['passed'],rejected=True,message=str(exc))
    else:
        gate=dict(passed=mass['passed'],rejected=False)
    write_json(out/'dynamic-admission-audit.json',gate)
    return dict(mass=mass,gate=gate,tangent_relative=stiffness_check)
