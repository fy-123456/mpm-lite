"""Append-only B1-B8 CPU experiment; run with --output NEW_DIRECTORY.

The archived nonlinear material space is static. Prescribed load/hold/unload
snapshots and the small quasi-static feedback scene are NOT C dynamics.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from .scenes import source_rule, operator, states, directions, errors
from engine.aniso_phase1.research_b.rules import compress, moment_audit, local_fallback
from engine.aniso_phase1.research_b.operator import MaterialSession
from engine.aniso_phase1.research_b.tensor import V22Space, TensorRule, TensorMaterialOperator

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    with open(path, 'x') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False,
                  default=lambda a: a.tolist() if isinstance(a, np.ndarray) else str(a))
        f.write('\n')


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p) for pattern in
            ('engine/aniso_phase1/research_b/*.py', 'benchmarks/research_b/*.py', 'tests/research_b/*.py')
            for p in sorted(ROOT.glob(pattern))}


def verify_baseline():
    path = ROOT/'docs/results/lite-aniso-mainline/v22/source-delivered-sha256.json'
    mapping = json.loads(path.read_text())
    differences = [name for name, digest in mapping.items() if not (ROOT/name).is_file() or sha(ROOT/name) != digest]
    if differences:
        raise RuntimeError(('current baseline changed; update protocol explicitly', differences))
    apath = ROOT/'docs/results/lite-aniso-mainline/v22/artifact-sha256.json'
    artifacts = json.loads(apath.read_text())
    adiff = [name for name, digest in artifacts.items() if not (ROOT/name).is_file() or sha(ROOT/name) != digest]
    if adiff:
        raise RuntimeError(('archive input mismatch', adiff))
    return dict(version='v22 archived snapshot (no Git metadata)', source_files_verified=len(mapping),
                artifact_files_verified=len(artifacts), source_manifest_sha256=sha(path),
                archive_manifest_sha256=sha(apath),
                source_zip_sha256=sha(ROOT/'docs/results/lite-aniso-mainline/v22/source-delivered.zip'))


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    baseline = verify_baseline()
    names = ['docs/results/lite-aniso-mainline/v22/multiscale/space/q4/'+f
             for f in ('round6.npz', 'basis-transform.npz', 'basis-raw.npz')]
    names += ['docs/results/lite-aniso-mainline/v19/space/reconstruction32.npz']
    reuse = ['engine/aniso_phase1/'+f for f in ('high_order_space.py', 'tensor_metrics.py',
             'tensor_reference.py', 'consistent_transfer.py', 'history_increment.py',
             'joint_sampling.py', 'direction_moments.py', 'grouped_quadrature.py', 'material_snapshot.py')]
    stat = os.statvfs(ROOT)
    protocol = dict(schema_version=1, producer='research_b/v1', created_utc=datetime.now(timezone.utc).isoformat(),
        baseline=baseline, source_sha256=source_hashes(), input_sha256={n:sha(ROOT/n) for n in names},
        reused_source_sha256={n:sha(ROOT/n) for n in reuse}, dtype='float64', device='CPU', random_seed=220930,
        units=dict(position='m', reference_volume='m^3', energy='J', PK1='Pa', time='s'),
        coordinates='reference Cartesian; x=X+u(q), F=I+grad(u); q row=scalar basis, column=world vector component',
        material=dict(mu=10., lam=20., k_f=200., model='Hencky + bilateral quadratic fibers'),
        reference_volume=.75*.25*.25, geometry=[[.125,.875],[.375,.625],[.375,.625]],
        boundary=dict(left_rigid_to=.25, right_rigid_from=.75, prescribed_right_displacement=.005),
        small_budgets=[4,8,16], small_methods=['conditional','representatives'],
        tensor_candidate_orders=[3,4,5], tensor_reference_order=6, tensor_certification_order=7,
        gates=dict(energy=.01, assembled_force=.01, stress_weak_moments=.01, tangent_action=.02,
                   reference_fraction=.1, derivative_energy_absolute=2e-7, derivative_tangent_relative=2e-4),
        absolute_denominator_floors=dict(U_J=1e-8, coefficient_force=1e-5, weak_stress_moment=1e-6, tangent_action=1e-4),
        rule_change_energy_budgets_J=dict(single=1e-5, cumulative_absolute=5e-5),
        normalization='norm(candidate-reference)/max(norm(reference), dimensioned frozen floor); absolute errors also retained',
        split=dict(train=['axial45','shear0'], development=['bend30','mixed60'],
                   hidden='independent reviewer chooses unseen whole families only after candidate freeze; no tuning after opening'),
        tensor_cases=['archive_load','archive_hold','archive_unload','local_mode','finite_rotation','end_hold'],
        exclusions=dict(stress_reconstruction=False, C_dynamics=False, CUDA=False, production_default_changed=False),
        environment=dict(python=platform.python_version(), numpy=np.__version__, machine=platform.machine(),
            cpu=platform.processor(), threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
            initial_load_average=os.getloadavg(), system_available_bytes=stat.f_bavail*stat.f_frsize,
            memory_limit_bytes=96636764160, cache='B CPU NumPy path; no Warp compilation',
            gpu_used=False, timing='CPU diagnostic only; exclusive host access not asserted'))
    write(output/'protocol.json', protocol)
    return protocol


def matched(op, q, d):
    result = op.evaluate(q, d)
    result['weak_moments'] = op.weak_moments(q)
    return result


def small_scan(output, protocol):
    rows, raw = [], {}
    construction = {}
    selected = None
    state_list = list(states('train'))+list(states('development'))
    write(output/'snapshot-manifest.json', [dict(**{k:v for k,v in s.items() if k!='q'},
          split='train' if i<8 else 'development', q_sha256=hashlib.sha256(s['q'].tobytes()).hexdigest())
          for i,s in enumerate(state_list)])
    np.savez_compressed(output/'snapshots.npz', **{str(i):s['q'] for i,s in enumerate(state_list)})
    for method in protocol['small_methods']:
        for budget in protocol['small_budgets']:
            passed = True
            for family in sorted({s['family'] for s in state_list}):
                family_states = [s for s in state_list if s['family']==family]
                scene = family_states[0]
                full_rule = source_rule(scene['angle'], scene['mixture'])
                start = time.perf_counter(); rule = compress(full_rule, budget, method)
                construction[f'{method}/{budget}/{family}'] = time.perf_counter()-start
                full, compact = operator(full_rule), operator(rule)
                rule.save(output/f'rule-{method}-{budget}-{family}.npz')
                for state in family_states:
                    per_direction = {}
                    for label, d in directions().items():
                        reference = matched(full, state['q'], d)
                        candidate = matched(compact, state['q'], d)
                        error = errors(candidate, reference)
                        per_direction[label] = error
                        if family in ('bend30','mixed60'):
                            passed &= all(v['passed'] for v in error.values())
                        key = f'{method}-{budget}-{state["id"].replace("/","-")}-{label}'
                        for k in ('U','force','tangent_action','weak_moments'):
                            raw[key+'-full-'+k] = reference[k]
                            raw[key+'-compact-'+k] = candidate[k]
                    rows.append(dict(method=method,budget=budget,snapshot=state['id'],
                        split='development' if family in ('bend30','mixed60') else 'train',
                        samples=len(rule.weight),full_samples=len(full_rule.weight),
                        moment_audit=moment_audit(full_rule,rule),directions=per_direction))
            if passed and selected is None:
                selected = dict(method=method,groups_per_partition=budget)
            print('SMALL', method, budget, 'development_passed', passed, flush=True)
    # Conservative policy: if geometric compression misses a budget, retain
    # full integration. B7 is a documented optional research stop, not a fit to
    # this development set followed by an unsupported generalization claim.
    write(output/'small-scan.json', dict(rows=rows,construction_seconds=construction,selected=selected,
          empirical_response_fit_used=False,stopped_B7_if_no_candidate=selected is None))
    np.savez_compressed(output/'small-raw.npz', **raw)
    return selected


def feedback_scene(output, selected):
    full_rule = source_rule(45., True)
    rule = full_rule if selected is None else compress(full_rule,selected['groups_per_partition'],selected['method'])
    operators = {'full':operator(full_rule),'compressed':operator(rule)}
    free = [(1,1),(2,2),(1,0),(2,0)]
    schedule = np.r_[np.linspace(0,1,6),np.ones(3),np.linspace(.8,0,5),np.zeros(3)]
    trajectories = {}
    for name,op in operators.items():
        records = []; guess = np.zeros(4)
        for step,scale in enumerate(schedule):
            q = np.zeros((9,3));q[0,0]=.04*scale;q[3,1]=.08*scale;q[6,0]=-.08*scale
            def objective(values):
                for idx,value in zip(free,values):q[idx]=value
                try:r=op.evaluate(q)
                except ValueError:return 1e50,np.zeros(4)
                return r['U'],np.array([r['force'][idx] for idx in free])
            solved=minimize(objective,guess,jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':100})
            objective(solved.x);result=op.evaluate(q);gradient=np.array([result['force'][idx] for idx in free])
            full_state=operators['full'].evaluate(q)
            if np.linalg.norm(gradient)>2e-7 or full_state['min_detF']<=0:
                raise RuntimeError(('small equilibrium failed',name,step,solved.message))
            guess=solved.x.copy()
            records.append(dict(step=step,scale=float(scale),q=q.copy(),energy=result['U'],
                reaction=float(result['force'][0,0]),min_detF=full_state['min_detF'],
                true_free_residual=float(np.linalg.norm(gradient)),iterations=int(solved.nit)))
        trajectories[name]=records
    a,b=trajectories['full'],trajectories['compressed']
    reaction_error=float(np.linalg.norm([x['reaction']-y['reaction'] for x,y in zip(a,b)])/max(np.linalg.norm([x['reaction'] for x in a]),1e-5))
    # Offline B8 proposals are evaluated at identical caller-owned states.
    compact=operators['compressed'];session=MaterialSession(compact,1e-5,5e-5)
    proposed=session.propose(operators['full'], np.array(b[5]['q']), directions()['random'])
    accepted=session.commit_rule(proposed,np.array(b[5]['q']),accepted_step=True)
    write(output/'feedback-scene.json', dict(kind='small quasi-static constrained equilibrium; no inertia or time integration',
        schedule='load-hold-unload-end_hold',trajectories=trajectories,raw_reaction_relative_error=reaction_error,
        scene_passed=bool(reaction_error<=.01),dynamic_acceptance=False,rule_proposal=asdict(proposed),
        proposal_accepted=accepted,rule_ledger=session.ledger,cumulative_absolute_rule_energy=session.cumulative_absolute))
    return reaction_error <= .01


def timed_evaluate(op, q, d=None):
    start=time.perf_counter();result=op.evaluate(q,d)
    return result,time.perf_counter()-start


def tensor_scan(output, protocol):
    start=time.perf_counter();space=V22Space(ROOT);build_seconds=time.perf_counter()-start
    write(output/'v22-space.json',dict(signature=space.signature,scalar_dofs=space.ndof,vector_dofs=3*space.ndof,
        carrier_scalar_dofs=space.n,local_scalar_dofs=space.ndof-space.n,shape=space.shape,
        reconstruction_relative_error=space.reconstruction_error,build_seconds=build_seconds,
        basis_array_bytes=space.oldA.nbytes+space.raw.data.nbytes+space.raw.indices.nbytes+space.raw.indptr.nbytes+space.transform.nbytes))
    ops={o:TensorMaterialOperator(space,TensorRule.uniform(space.edges,o)) for o in [3,4,5,6,7]}
    rng=np.random.default_rng(220930);random=rng.normal(size=space.q.shape);random/=np.linalg.norm(random)
    bending=np.zeros_like(space.q);X=space.reference[:space.n]-.5
    bending[:space.n,0]=-X[:,0]*X[:,1];bending[:space.n,1]=.5*X[:,0]**2;bending/=np.linalg.norm(bending)
    local=np.zeros_like(space.q);local[-1,0]=1.
    ds={'random':random,'bending':bending,'local':local}
    base,t6=timed_evaluate(ops[6],space.q,random)
    np.savez_compressed(output/'v22-base-order6.npz',**base)
    print('V22 order6 baseline',t6,'seconds',base['material_calls'],'points',flush=True)
    fine,t7=timed_evaluate(ops[7],space.q,random)
    certification=errors(base,fine,scale=.1)
    np.savez_compressed(output/'v22-base-order7.npz',**fine)
    write(output/'reference-certification.json',dict(reference_order=6,check_order=7,errors=certification,
        passed=all(x['passed'] for x in certification.values()),seconds=[t6,t7],
        scope='v22 archived finite-deformation state; each further case certified separately'))
    # Run the unchanged original potential at its original five-point rule.
    start=time.perf_counter();old=space.original_potential().evaluate(space.reference+space.q,random,order=5)
    old_seconds=time.perf_counter()-start
    same,t5=timed_evaluate(ops[5],space.q,random)
    wiring=errors(same,old)
    write(output/'baseline-wiring.json',dict(errors=wiring,original_seconds=old_seconds,adapter_seconds=t5,
          passed=all(x['relative']<1e-7 or x['absolute']<1e-9 for x in wiring.values()),
          note='unchanged HighOrderPotential body via sparse coordinate-map facade; independent material formula also unit-tested'))
    R=expm(np.array([[0.,-.45,.2],[.45,0.,-.1],[-.2,.1,0.]]))
    rotated=(space.reference+space.q)@R.T-space.reference
    local_state=space.q.copy();local_state[-1,0]+=2e-5
    cases=[('archive_load',.5*space.q),('archive_hold',space.q),('archive_unload',.35*space.q),
           ('local_mode',local_state),('finite_rotation',rotated),('end_hold',np.zeros_like(space.q))]
    rows=[];raw={};passes={o:True for o in (3,4,5)};certified=True
    # Random at each stage; extra bending/local actions at peak load ensure
    # high-order and low-stiffness directions cannot hide behind total energy.
    jobs=[(name,q,'random',random) for name,q in cases]+[('archive_hold',space.q,k,d) for k,d in ds.items() if k!='random']
    for name,q,label,d in jobs:
        if name=='archive_hold' and label=='random':reference,check=base,fine
        else:
            reference,_=timed_evaluate(ops[6],q,d)
            check,_=timed_evaluate(ops[7],q,d)
        ce=errors(reference,check,scale=.1);certified &= all(x['passed'] for x in ce.values())
        for order in (3,4,5):
            candidate,seconds=timed_evaluate(ops[order],q,d)
            error=errors(candidate,reference)
            passed=all(x['passed'] for x in error.values())
            passes[order]&=passed
            rows.append(dict(case=name,direction=label,order=order,errors=error,reference_self_check=ce,
                samples=candidate['material_calls'],full_samples=reference['material_calls'],
                min_detF=candidate['min_detF'],seconds=seconds,passed=passed))
            for key in ('U','force','tangent_action','weak_moments','slab_energy','slab_tangent_work'):
                raw[f'{name}-{label}-{order}-{key}']=candidate[key]
        for order,result in ((6,reference),(7,check)):
            for key in ('U','force','tangent_action','weak_moments','slab_energy','slab_tangent_work'):
                raw[f'{name}-{label}-{order}-{key}']=result[key]
        print('V22',name,label,'candidate passes',passes,'reference',certified,flush=True)
    selected=next((o for o in (3,4,5) if passes[o] and certified),None)
    write(output/'v22-scan.json',dict(rows=rows,passes=passes,reference_certified=bool(certified),selected_order=selected,
        reconstruction=False,constant_fiber='F45',scope='same fixed v22 space, no continuum certification'))
    np.savez_compressed(output/'v22-raw.npz',**raw)
    if selected:
        write(output/'v22-selected-rule.json',dict(kind='nonoverlapping-cell tensor Gauss conditional constant direction',
            orders=ops[selected].rule.orders,signature=ops[selected].rule.signature,
            full_reference_order=6,original_v22_order=5,
            note='Savings are relative to certified six-point reference, not a speedup over the original five-point implementation.'))
        timings=[]
        # Warm up both before interleaved diagnostic measurements.
        for o in (selected,6):ops[o].evaluate(space.q)
        for repeat in range(3):
            for o in ((selected,6) if repeat%2==0 else (6,selected)):
                r,seconds=timed_evaluate(ops[o],space.q)
                timings.append(dict(repeat=repeat,order=o,seconds=seconds,material_calls=r['material_calls']))
        write(output/'cpu-cost.json',dict(basis_preprocessing_seconds=build_seconds,interleaved=timings,
             peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
             load_average=os.getloadavg(),exclusive_host=False,formal_speedup_claim=False,
             Newton_Krylov='not measured: C/D integrated path unavailable',CUDA='not run'))
    return selected,certified


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['all','small','tensor'],default='all')
    args=parser.parse_args()
    start=time.perf_counter();protocol=prepare(args.output)
    selected=small_scan(args.output,protocol) if args.stage in ('all','small') else None
    feedback=feedback_scene(args.output,selected) if args.stage in ('all','small') else None
    tensor,certified=tensor_scan(args.output,protocol) if args.stage in ('all','tensor') else (None,None)
    frozen=dict(schema_version=1,source_sha256=source_hashes(),small_candidate=selected,
        tensor_candidate_order=tensor,reference_certified=certified,small_feedback_passed=feedback,
        fallback='Reject invalid F. Between steps, request local sufficient integration; C owns acceptance. Never change rule inside Newton.',
        hidden_status='not opened; independent review required',no_response_training=True,
        B9='not accepted: no matching C dynamic interface',B10_CUDA='not run: D backend unavailable',
        continuum_spatial_accuracy='v22 remains unpassed',elapsed_seconds=time.perf_counter()-start)
    write(args.output/'candidate-freeze.json',frozen)
    print('CANDIDATE_FROZEN',str(args.output),frozen,flush=True)


if __name__=='__main__':main()
