"""Freeze and test the explicitly selected A input; no dynamic solve is run."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from engine.aniso_phase1.research_d.identity import ROOT, BASELINE, sha, write_json, baseline_audit
from engine.aniso_phase1.research_d.common_space import CommonSpace
from engine.aniso_phase1.research_d.common_kinetic import PointInertia
from engine.aniso_phase1.research_d.common_state import CommonState, StateTransaction, kinetic_state_contract
from .material_reference import audit_material_reference

PIN = "7422b2b099127bebe9c144ea6cac94409bde861aadb94693b7fc650b397c0332"


def source_manifest():
    """Bind all executed engine/benchmark Python, plus targeted test sources."""
    paths = []
    for d in ("engine", "benchmarks", "tests/research_d", "tests/research_a", "tests/research_b", "tests/research_c"):
        paths.extend((ROOT/d).rglob("*.py"))
    paths.extend(ROOT/n for n in ("pyproject.toml", "uv.lock"))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths)) if p.is_file()}


def flat_ids(rows):
    return (np.asarray(rows)[:, None]*3+np.arange(3)).ravel().tolist()


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b),1e-30))


def write_npz(path, **arrays):
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **arrays)


def prepare(out, s):
    v = s.test_vectors
    direction = v["direction"].copy()
    rng = np.random.default_rng(22093001)
    local = np.zeros(s.q_shape)
    local[s.nfree_carrier:] = rng.normal(size=local[s.nfree_carrier:].shape)
    local /= np.linalg.norm(local)
    q0 = s.expand(s.q0)
    q1 = s.expand(s.q0 + 1e-4*direction)
    states = {"archived_static":q0, "fixed_perturbation_1e-4_m":q1}
    directions = {"archive_direction":s.direction_coefficients(direction),
                  "local_only_direction":s.direction_coefficients(local)}
    left = np.flatnonzero(s.carrier_X[:,0] <= s.boundary["left_x"])
    right = np.flatnonzero(s.carrier_X[:,0] >= s.boundary["right_x"])
    material = dict(schema_version=1, orders=dict(wiring=5,candidate=6,check=7),
        dof_groups=dict(free=flat_ids(s.free_scalar_ids), fixed=flat_ids(s.fixed_scalar_ids),
                        grip_left=flat_ids(left),grip_right=flat_ids(right)),
        budgets=dict(energy=dict(atol=1e-10,rtol=1e-4,scale=.01),
                     force=dict(atol=1e-8,rtol=1e-4,scale=1.),
                     tangent=dict(atol=1e-7,rtol=1e-3,scale=1.)))
    protocol = dict(schema_version=1,stage="first_stage_common_input_freeze",
        baseline_sha256=BASELINE,space_manifest_sha256=s.signature, seed=22093001,
        model="frozen A online-overlap-144 space; original material and stabilization",
        material_reference=material, mass=dict(density_kg_m3=1.,candidate_order=5,check_order=6,
            relative_tolerance=1e-10, absolute_tolerance=1e-12),
        mapping=dict(absolute_tolerance=1e-7,adjoint_relative_tolerance=1e-10),
        archive_wiring=s.metadata["fixed_test_gates"],
        state_generation="q0 from sealed A test-vectors; q1=q0+1e-4m*sealed unit direction",
        initial_state="loaded static snapshot, zero velocity, local clock 0; not a loading protocol",
        runtime=dict(python=sys.version,threads={k:os.environ.get(k) for k in
                     ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")}),
        scope=dict(dynamic=False,continuum_spatial_accuracy=False,cuda=False,
                   material_reference="only two frozen states and two tangent directions",
                   baseline_mass_rank_certification=False),
        code_sha256=source_manifest())
    # Protocol and all inputs are published before computing acceptance.
    write_json(out/"protocol.json",protocol)
    write_json(out/"maps.json",s.maps_info())
    write_npz(out/"input-states.npz",**{f"state_{k}":a for k,a in states.items()},
              **{f"direction_{k}":a for k,a in directions.items()})
    write_npz(out/"initial-state.npz",q=q0,velocity=np.zeros_like(q0),time=np.array(0.),
              step=np.array(0),free_q=s.q0,free_scalar_ids=s.free_scalar_ids,
              fixed_scalar_ids=s.fixed_scalar_ids,lift=s.lift)
    contract = kinetic_state_contract(s.signature,(s.ndof,3),"Q4-full-reference-mass-Gauss5-v1")
    contract["mass_operator"] = PointInertia(s).contract()
    contract["boundary"] = s.metadata["boundary_conditions"]
    write_json(out/"kinetic-state-contract.json",contract)
    return protocol,states,directions


def mapping_audit(s):
    v=s.test_vectors; points=[v[f"points{k}"] for k in range(3)]
    x,F=s.evaluate(s.q0,points);dx,dF=s.jvp(v["direction"],points)
    errors={k:float(np.max(abs(a-v[k]))) for k,a in (("x",x),("F",F),("dx",dx),("dF",dF))}
    rng=np.random.default_rng(22093002)
    fx=rng.normal(size=x.shape);ff=rng.normal(size=F.shape)
    lhs=float(np.sum(dx*fx)+np.sum(dF*ff))
    rhs=float(np.sum(v["direction"]*s.vjp(fx,ff,points)))
    adj=abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-30)
    full=s.expand(s.q0)
    mapping=bool(np.array_equal(full[s.free_scalar_ids],s.q0) and
                 np.array_equal(full[s.fixed_scalar_ids],s.lift[s.fixed_scalar_ids]))
    return dict(passed=bool(max(errors.values())<=1e-7 and adj<=1e-10 and mapping),
                archived_probe_absolute_errors=errors,adjoint_relative_error=adj,
                full_free_mapping_passed=mapping,min_probe_detF=float(np.linalg.det(F).min()),
                force_semantics="positive potential gradient; constrained rows retained")


def kinetic_audit(s,directions):
    m=PointInertia(s,order=5); check=PointInertia(s,order=6)
    vs=list(directions.values())
    a,b=vs
    ma,mb=m.apply(a),m.apply(b)
    comparisons=[relative(ma,check.apply(a)),relative(mb,check.apply(b))]
    sym=abs(float(np.sum(a*mb)-np.sum(b*ma)))/max(abs(float(np.sum(a*mb))),abs(float(np.sum(b*ma))),1e-30)
    translation=np.zeros((s.ndof,3));translation[:s.n,0]=1.
    mass=2*m.energy(translation)
    # Nonzero response on carriers to a purely local velocity witnesses retention
    # of the cross block; this is not an exhaustive mass-rank certification.
    cross=float(np.linalg.norm(mb[:s.n]))
    energies=[.5*float(np.sum(v*f)) for v,f in zip(vs,(ma,mb))]
    return dict(passed=bool(max(comparisons)<1e-10 and sym<1e-10 and
                abs(mass-m.volume)<1e-12 and min(energies)>0 and cross>0),
                order5_vs6_relative_actions=comparisons,symmetry_relative_error=sym,
                translation_mass_kg=mass,expected_mass_kg=m.volume,
                mode_kinetic_energies_J=energies,carrier_local_cross_action_norm=cross,
                complete_mass_rank_certified=False,contract=m.contract())


def transaction_audit(s,initial):
    points=[s.test_vectors[f"points{k}"] for k in range(3)]
    def physical(state):
        if not np.array_equal(state.q[s.fixed_scalar_ids],s.lift[s.fixed_scalar_ids]):
            raise ValueError("changed fixed boundary")
        if np.any(state.velocity[s.fixed_scalar_ids]!=0):
            raise ValueError("static fixture has zero fixed velocity")
        _,F=s.evaluate(state.q[s.free_scalar_ids],points)
        if np.any(np.linalg.det(F)<=0):raise ValueError("invalid probe detF")
    state=CommonState(initial.copy(),np.zeros_like(initial),child_states={"material_rule":"full6","space":s.signature})
    tx=StateTransaction(state,validator=physical);before=tx.snapshot().digest()
    trial=tx.begin_trial();trial.state.q[0,0]+=1.;trial.state.time=.001;trial.state.step=1
    refused=False
    try:tx.commit(trial)
    except ValueError:refused=True
    unchanged=tx.snapshot().digest()==before
    stale=tx.begin_trial();good=tx.begin_trial()
    good.state.time=.001;good.state.step=1
    good.state.predictor=np.zeros_like(initial);good.state.child_states["audit"]="accepted-value-copy"
    tx.commit(good)
    rejected=False
    try:tx.commit(stale)
    except ValueError:rejected=True
    copied=tx.snapshot();copied.q[:]=123
    isolation=not np.array_equal(copied.q,tx.snapshot().q)
    return dict(passed=bool(refused and unchanged and rejected and isolation),
                invalid_boundary_rejected=refused,rollback_unchanged=unchanged,
                stale_trial_rejected=rejected,snapshot_copy_isolated=isolation,
                dynamic_step_performed=False,scope="value transaction; physical probe callback only")


def run(out):
    out=Path(out);start=time.monotonic()
    s=CommonSpace(out/"inputs",expected_manifest_sha256=PIN)
    print("LOAD",s.q_shape,(s.ndof,3),flush=True)
    protocol,states,directions=prepare(out,s)
    print("PROTOCOL FROZEN",sha(out/"protocol.json"),flush=True)
    mapping=mapping_audit(s);write_json(out/"mapping-audit.json",mapping)
    kinetic=kinetic_audit(s,directions);write_json(out/"kinetic-audit.json",kinetic)
    transaction=transaction_audit(s,states["archived_static"]);write_json(out/"transaction-audit.json",transaction)
    print("MAPPING MASS TRANSACTION",mapping["passed"],kinetic["passed"],transaction["passed"],flush=True)
    # Exact archived A response from its independent dense HighOrderPotential.
    v=s.test_vectors; wiring=s.response(s.q0,v["direction"],order=5)
    diffs={k:relative(wiring[k],v[k]) for k in ("energy_J","force","tangent_action")}
    gates=protocol["archive_wiring"]
    wp=diffs["energy_J"]<=gates["energy_relative"] and diffs["force"]<=gates["force_relative"] and diffs["tangent_action"]<=gates["tangent_relative"]
    write_json(out/"archive-wiring-audit.json",dict(passed=bool(wp),relative_errors=diffs,
                 min_detF=wiring["min_detF"],scope="A archived original potential versus B streaming adapter, same A basis"))
    print("ARCHIVE WIRING",wp,diffs,flush=True)
    material=audit_material_reference(s,states,directions,protocol["material_reference"])
    write_json(out/"material-reference-audit.json",material)
    print("MATERIAL REFERENCE",material["passed"],flush=True)
    baseline=baseline_audit();write_json(out/"baseline-audit.json",baseline)
    source_unchanged=source_manifest()==protocol["code_sha256"]
    accepted=all((mapping["passed"],kinetic["passed"],transaction["passed"],wp,
                  material["passed"],baseline["passed"],source_unchanged))
    result=dict(common_input_freeze_passed=bool(accepted),protocol_sha256=sha(out/"protocol.json"),
        space_manifest_sha256=s.signature,source_unchanged_during_audit=source_unchanged,
        spatial_operator_consistency_passed=bool(mapping["passed"] and wp),
        material_order6_bounded_reference_passed=material["passed"],mass_order5_action_checks_passed=kinetic["passed"],
        state_interface_passed=transaction["passed"],dynamic_certified=False,
        continuum_spatial_certified=False,cuda_certified=False,default_changed=False,
        seconds=time.monotonic()-start,scope="first-stage inputs and interfaces only",
        departments=dict(A="frozen selected basis; new A bases require a new bundle",
            B="complete six-point material reference, no compression or rule switch",
            C="reference-point inertia and copied state transaction, no time integrator",
            D="real adapter and provenance closure",E="physics parameters retained; coupling not enabled"))
    write_json(out/"acceptance.json",result)
    print("ACCEPTANCE",json.dumps(result),flush=True)
    if not accepted:raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--output",required=True)
    run(p.parse_args().output)

if __name__=="__main__":main()
