"""One-process practical implementation of S01--S12; every result is scoped."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_d.common_state import CommonState
from engine.aniso_phase1.research_c.stage2.dynamics import AVF, StepRejected
from engine.aniso_phase1.research_sequential.condensation import Condensation, CondensedModel
from benchmarks.research_c.stage2.run import load, ROOT, PARENT

C = ROOT/'docs/results/parallel-v22-stage2/C/20260930T152000Z-common-dynamics-rank-audit'
E = ROOT/'docs/results/parallel-v22-stage2/E/20260930T071848Z_E_stage2'
TIMES = [0., .25, .5, .6, .85, 1.1, 1.35, 1.6]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sources():
    folders = ['engine/aniso_phase1/research_sequential', 'benchmarks/research_sequential']
    return {str(p.relative_to(ROOT)): sha(p) for folder in folders for p in sorted((ROOT/folder).glob('*.py'))}


def log(*values):
    print(time.strftime('%H:%M:%S'), *values, flush=True)


def relative(a, b, floor=1e-12):
    return float(la.norm(np.asarray(a)-b)/max(la.norm(b), floor))


def record(out, phase, value):
    write(out/(phase+'.json'), dict(value, source_sha256=sources(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    log(phase, value.get('status', value.get('passed')))


def protocol():
    return dict(schema='sequential-practical-v2', parent_sha256=PARENT, seed=20260930,
        sequence='one process, one numerical job, no process pool', dtype='float64',
        full_material_order=7, comparison_order=8, compressed_material_order=5,
        mass_order=5, density_kg_m3=1., AVF_path_order=2, cycle_times_s=TIMES,
        tolerances=dict(residual_atol=1e-7, residual_rtol=1e-5, ledger_J=1e-7,
            min_detF=.1, backend_relative=2e-5, backend_cycle_relative=.005, material_relative=.02,
            tangent_relative=.03, finite_difference_relative=.002,
            cycle_field_relative=.05, reaction_absolute_N=1e-4),
        limits=dict(spatial_certified=False, time_convergence_certified=False,
            long_cycle=False, independent_generalization=False, exclusive_performance=False,
            coupled_3D_poroelasticity=False),
        physics=dict(no_artificial_mass=True, original_Ks=True, all_mass_cross_terms=True,
            unchanged_144_local_functions=True, no_grid_transfer=True, no_damping=True))


def open_reduction(out):
    s, _, _, parent_audit = load()
    binding = json.loads((out/'S01.json').read_text())
    for name, expected in binding['old_sources'].items():
        if sha(ROOT/name) != expected:raise ValueError('changed sealed source: '+name)
    if sha(C/'operators.npz') != binding['operators_sha256']:
        raise ValueError('mass/stiffness input changed')
    with np.load(C/'operators.npz', allow_pickle=False) as z:
        reduction = Condensation(s, z['M5'], z['K'])
    if (out/'common-model-package.json').exists():
        package = json.loads((out/'common-model-package.json').read_text())
        if package['reduction_sha256'] != reduction.signature:
            raise ValueError('independent coordinate model changed')
    return reduction


def prepare(out):
    if (out/'S01.json').exists():raise ValueError('prepare needs a new output directory')
    old = json.loads(Path(__file__).with_name('baseline.json').read_text())
    for name, expected in {**old['source_snapshot'], **old['identities']}.items():
        if sha(ROOT/name) != expected:raise ValueError('baseline changed: '+name)
    write(out/'protocol.json', protocol())
    record(out, 'S01', dict(passed=True, old_sources=old['source_snapshot'],
        input_identities=old['identities'], operators_sha256=sha(C/'operators.npz'),
        verified_old_sources=len(old['source_snapshot']), version='v22 + two A-E rounds + common freeze',
        git_commit=None, free_system_GiB=os.statvfs('/').f_bavail*os.statvfs('/').f_frsize/2**30))
    r = open_reduction(out); s = r.parent
    np.savez_compressed(out/'coordinates.npz', P=r.P, offset=r.offset, N=r.N, R=r.R,
                        M=r.M, K=r.K, free=r.free, fixed=r.fixed)
    rng = np.random.default_rng(20260930)
    w = rng.normal(size=(r.P.shape[1], 3))*.001
    full = r.expand(w); unrelaxed = full+r.N @ rng.normal(size=(r.N.shape[1], 3))
    field_error = relative(s.oldA @ full[:s.n], s.oldA @ unrelaxed[:s.n])
    null_constraint = float(la.norm(r.algebraic_residual(full)))
    v = rng.normal(size=w.shape)
    mass_work_error = abs(float(np.sum(v*(r.M @ v))-np.sum(r.velocity(v)*(r.original_mass @ r.velocity(v)))))
    eig, modes = la.eigh(r.K[np.ix_(r.ids,r.ids)], np.kron(r.M[np.ix_(r.free,r.free)],np.eye(3)), subset_by_index=[0,5])
    residual = r.K[np.ix_(r.ids,r.ids)] @ modes-np.kron(r.M[np.ix_(r.free,r.free)],np.eye(3)) @ (modes*eig)
    check = dict(r.audit, physical_field_null_difference=field_error,
        stationary_residual=null_constraint, kinetic_work_absolute=mass_work_error,
        lowest_six_eigenvalues=eig.tolist(), modal_residual=relative(residual, np.zeros_like(residual), floor=la.norm(r.K[np.ix_(r.ids,r.ids)] @ modes)),
        passed=bool(field_error<1e-8 and null_constraint<1e-8 and mass_work_error<1e-8 and eig[0]>0))
    record(out,'S02-S03-algebra',check)
    if not check['passed']:raise ValueError('independent-coordinate algebra failed')
    for stage, why in [('S04','independent h/p reference remains uncertified; retain known finite model'),
                       ('S05','reuse 72/144/288 development evidence; retain 144 functions, no new precision claim')]:
        record(out,stage,dict(status='limited_existing_evidence',reason=why,spatial_certified=False))
    write(out/'common-model-package.json',dict(schema='independent-motion-v1', parent_sha256=PARENT,
        parent_space_sha256=s.signature, reduction_sha256=r.signature, coordinate_file_sha256=sha(out/'coordinates.npz'),
        q_shape=[len(r.free)+len(r.fixed),3], free_motion_shape=[len(r.free),3], original_shape=[s.ndof,3],
        coordinates='independent free motion, then original fixed rows; xyz last; meters',
        displacement='q_full=P@w+offset', velocity='v_full=P@wdot',
        algebraic_equation='N.T@Ks@(reference_carriers+q_carriers)=0',
        mass='P.T@M5@P, all carrier-local and boundary cross terms retained',
        force='positive original potential gradient, reduced by P.T',
        tangent='P.T@H@P, no additional quadrature weights',
        spatial_certified=False, protocol_sha256=sha(out/'protocol.json'),
        source_sha256=sources()))
    record(out,'S06',dict(passed=True, reduction_sha256=r.signature,original_space_unchanged=True))


def audits(out):
    r = open_reduction(out); s = r.parent; m = CondensedModel(r)
    zero = m.rest().q; loaded = r.project(s.expand(s.q0))
    rng = np.random.default_rng(20260930); d = rng.normal(size=loaded.shape)
    d[m.fixed] = 0.; d /= la.norm(d)
    log('rest tangent and finite deformation checks')
    rest = m.evaluate(zero,d)
    tangent = relative(rest['tangent_action'],(r.K @ d.ravel()).reshape(d.shape))
    base = m.evaluate(loaded,d); h = 1e-5
    plus = m.evaluate(loaded+h*d); minus = m.evaluate(loaded-h*d)
    fd = relative((plus['force']-minus['force'])/(2*h),base['tangent_action'])
    grad = abs((plus['U']-minus['U'])/(2*h)-float(np.sum(base['force']*d)))
    result=dict(passed=bool(tangent<.002 and fd<.002 and grad<1e-7 and la.norm(rest['force'][m.free])<1e-8),
        rest_force_N=float(la.norm(rest['force'][m.free])),rest_tangent_relative=tangent,
        finite_difference_tangent_relative=fd,finite_difference_energy_absolute=grad,
        min_detF=base['min_detF'])
    record(out,'S03-derivatives',result)
    if not result['passed']:raise ValueError('original potential derivative check failed')
    log('q7 versus q8 material check; one loaded state and mixed direction')
    ref = CondensedModel(r,order=8).evaluate(loaded,d)
    keys=['material_U','material_force','weak_moments','tangent_action','slab_tangent_work']
    errors={k:relative(base[k],ref[k]) for k in keys}
    result=dict(passed=max(errors.values())<.02,errors=errors,states=['archived-loaded, algebraically equilibrated'],
        coverage='one F45 state plus rest; other material variants retain old B evidence',full_order=7,check_order=8,
        limits='not a uniform error bound over arbitrary deformation')
    record(out,'S07',result)
    if not result['passed']:raise ValueError('practical full material check failed')
    compressed=CondensedModel(r,order=5).evaluate(loaded,d)
    errors5={k:relative(compressed[k],base[k]) for k in keys}
    record(out,'S10-material',dict(passed=max(errors5.values())<.03,errors=errors5,
        mass_unchanged=True,material_point_reduction=1-(5/7)**3,order=5))
    np.savez_compressed(out/'audit-vectors.npz',loaded=loaded,direction=d,
        force=base['force'],tangent=base['tangent_action'],energy=base['U'])


def stepper(model,state=None):
    return AVF(model,state,path_order=2,residual_atol=1e-7,residual_rtol=1e-5,ledger_atol=1e-7)


def save_state(path,model,state):
    payload=state.to_dict()
    write(path,dict(payload=payload,digest=state.digest(),model_sha256=model.signature,source_sha256=sources()))


def load_state(path,model,*,allow_driver_revision=False):
    record=json.loads(Path(path).read_text())
    expected=record['source_sha256'];actual=sources()
    if allow_driver_revision:
        expected={k:v for k,v in expected.items() if k.startswith('engine/')}
        actual={k:v for k,v in actual.items() if k.startswith('engine/')}
    if record['model_sha256']!=model.signature or expected!=actual:
        raise ValueError('checkpoint model/source mismatch')
    p=record['payload'];p['q']=np.array(p['q']);p['velocity']=np.array(p['velocity'])
    if p['predictor'] is not None:p['predictor']=np.array(p['predictor'])
    state=CommonState(**p)
    if state.digest()!=record['digest']:raise ValueError('checkpoint state digest mismatch')
    model.validate(state)
    return state


def cycle(out,name,model,times=TIMES):
    folder=out/name;folder.mkdir(exist_ok=True)
    if (folder/'summary.json').exists():raise ValueError('do not overwrite completed cycle')
    rows=[]; qs=[]; vs=[]; xs=[]; stresses=[]; ts=[]
    initial=None
    if (folder/'checkpoint.json').exists():
        initial=load_state(folder/'checkpoint.json',model,allow_driver_revision=True)
        write(folder/'resume-lineage.json',dict(checkpoint_sha256=sha(folder/'checkpoint.json'),
            model_sha256=model.signature,engine_sources_unchanged=True,
            reason='practical solver tolerance revision; committed physical state preserved'))
        rows=json.loads((folder/'ledger.json').read_text()) if (folder/'ledger.json').exists() else []
        with np.load(folder/'trajectory.npz') as saved:
            qs=list(saved['q']);vs=list(saved['velocity']);xs=list(saved['x']);stresses=list(saved['PK1']);ts=list(saved['times'])
        if initial.step!=len(rows) or len(ts)!=len(rows)+1 or ts[-1]!=initial.time:
            raise ValueError('checkpoint, trajectory and ledger disagree')
    integrator=stepper(model,initial)
    before=integrator.state.digest()
    def fail(where,trial):
        if where=='before_material':raise ValueError('intentional rollback audit')
    try:integrator.step(.0001,inject=fail)
    except StepRejected:pass
    rollback=integrator.state.digest()==before
    axes=[np.linspace(e[0],e[-1],n) for e,n in zip(model.parent.edges,[33,7,7])]
    def snapshot():
        state=integrator.state; f=model.fields(state,axes)
        qs.append(state.q);vs.append(state.velocity);xs.append(f['x']);stresses.append(f['PK1']);ts.append(state.time)
        np.savez_compressed(folder/'trajectory.npz',times=ts,q=qs,velocity=vs,X=f['X'],x=xs,PK1=stresses)
        save_state(folder/'checkpoint.json',model,state)
    if initial is None:snapshot()
    start=time.perf_counter()
    for target in times[1:]:
        if target<=integrator.state.time:continue
        log(name,'advance',integrator.state.time,'->',target)
        try:row=integrator.step(target-integrator.state.time,max_iters=8)
        except StepRejected:
            write(folder/'failure.json',dict(failures=integrator.failures,last_committed_digest=integrator.state.digest()))
            raise
        rows.append(row);write(folder/'ledger.json',rows);snapshot()
        log(name,'accepted',row['step'],'detF',round(row['min_detF'],6),'residual',f'{row["true_residual"]:.2g}','seconds',round(row['wall_seconds'],2))
    state=integrator.state
    reloaded=load_state(folder/'checkpoint.json',model)
    before=state.digest()
    finite=all(np.isfinite(a).all() for a in (qs,vs,xs,stresses))
    passed=bool(finite and rollback and reloaded.digest()==before and min(a['min_detF'] for a in rows)>.1)
    summary=dict(passed=passed,status='practical_cycle_pass' if passed else 'failed',steps=len(rows),end_s=state.time,
        min_detF=min(a['min_detF'] for a in rows),max_ledger_closure_J=max(abs(a['budget_defect_J']) for a in rows),
        max_path_quadrature_error_J=max(abs(a['path_quadrature_error_J']) for a in rows),
        max_true_residual=max(a['true_residual'] for a in rows),rollback=rollback,checkpoint_reload=True,
        wall_seconds=sum(row['wall_seconds'] for row in rows),resume_segment_seconds=time.perf_counter()-start,material_order=model.rule.orders[0],device=model.device,
        temporal_accuracy_certified=False,spatial_accuracy_certified=False)
    write(folder/'summary.json',summary)
    if not passed:raise ValueError('cycle scene stability failed')
    return summary


def cpu(out):
    if not json.loads((out/'S07.json').read_text())['passed']:raise ValueError('full integration prerequisite')
    r=open_reduction(out)
    result=cycle(out,'cpu-q7',CondensedModel(r,order=7))
    record(out,'S08-S09',result)


def compare_cycles(out,a,b):
    with np.load(out/a/'trajectory.npz') as za,np.load(out/b/'trajectory.npz') as zb:
        if not np.array_equal(za['times'],zb['times']):raise ValueError('different time protocols')
        displacement=relative(za['x']-za['X'],zb['x']-zb['X'])
        stress=relative(za['PK1'],zb['PK1'])
        velocity=relative(za['velocity'],zb['velocity'])
    ra=np.array([r['reaction_N'] for r in json.loads((out/a/'ledger.json').read_text())])
    rb=np.array([r['reaction_N'] for r in json.loads((out/b/'ledger.json').read_text())])
    return dict(displacement_relative=displacement,PK1_relative=stress,coefficient_velocity_relative=velocity,
        reaction_relative=relative(ra,rb),reaction_max_absolute_N=float(np.max(abs(ra-rb))))


def gpu(out):
    if not json.loads((out/'S08-S09.json').read_text())['passed']:raise ValueError('CPU reference prerequisite')
    import warp as wp
    wp.config.kernel_cache_dir=str(out.resolve()/'warp-cache')
    r=open_reduction(out);model=CondensedModel(r,order=7,device='cuda:0')
    with np.load(out/'audit-vectors.npz') as z:
        response=model.evaluate(z['loaded'],z['direction'])
        errors=dict(force=relative(response['force'],z['force']),tangent=relative(response['tangent_action'],z['tangent']),
                    energy=relative(response['U'],z['energy']))
    record(out,'S11-static',dict(passed=max(errors.values())<2e-5,errors=errors,device='cuda:0'))
    if max(errors.values())>=2e-5:raise ValueError('actual GPU operator discrepancy')
    result=cycle(out,'gpu-q7',model)
    difference=compare_cycles(out,'gpu-q7','cpu-q7')
    result['CPU_GPU_comparison']=difference
    result['passed']=result['passed'] and max(difference.values())<.005
    record(out,'S11',result)
    if not result['passed']:raise ValueError('GPU cycle differs from CPU')
    del model
    import gc;gc.collect()
    if not json.loads((out/'S10-material.json').read_text())['passed']:
        record(out,'S10',dict(status='skipped_unqualified_compression'));return
    model=CondensedModel(r,order=5,device='cuda:0')
    result=cycle(out,'gpu-q5',model)
    difference=compare_cycles(out,'gpu-q5','gpu-q7');result['full_compressed_comparison']=difference
    result['passed']=bool(result['passed'] and difference['displacement_relative']<.05 and difference['PK1_relative']<.05
        and (difference['reaction_relative']<.05 or difference['reaction_max_absolute_N']<1e-4))
    result['mass_unchanged']=True
    record(out,'S10',result)


def mixed(out):
    from engine.aniso_phase1.research_e.stage2.handoff import load as load_e
    from engine.aniso_phase1.research_e.stage2.d_adapter import consume
    from engine.aniso_phase1.research_sequential.solver import solve_mixed
    folder=E/'handoff';metadata=json.loads((folder/'operator-package.json').read_text())
    metadata,M,v,_=load_e(folder,expected_manifest_sha256=sha(folder/'operator-package.json'),expected=metadata)
    contract=consume(out/'mixed-contract',metadata,M,v,ROOT/'engine/aniso_phase1/research_d/stage2/contracts.py')
    solution,info=solve_mixed(M,v['rhs'])
    error=relative(solution,v['solution'])
    bad_solver=False
    try:solve_mixed(M,v['rhs'],method='cg')
    except ValueError:bad_solver=True
    record(out,'S12-mixed',dict(passed=bool(contract['passed'] and error<1e-8 and bad_solver),
        final_D_contract=contract,new_general_backend=info,solution_relative=error,CG_rejected=bad_solver,
        coupled_3D_dynamics=False,scope='independent 2D three-block system, exact original residual'))
    from .transactions import audit_transaction
    record(out,'S12',audit_transaction(out,metadata,v))


def report(out):
    stages={p.stem:json.loads(p.read_text()) for p in sorted(out.glob('S*.json'))}
    write(out/'summary.json',dict(protocol=protocol(),stages=stages,
        limitations=['independent spatial reference and 2% spatial goal remain unmet',
            'seven coarse steps assess stable scene execution, not transient convergence',
            'F45 mainline only in this extension; no new multi-material generalization',
            'E independent 2D mixed backend; no physical 3D solid-fluid coupling',
            'timings include shared hardware and no exclusive performance certification']))
    write(out/'artifact-sha256.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
        if p.is_file() and 'warp-cache' not in p.parts and p.name not in ('artifact-sha256.json','.lock')})
    log('report written',out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','audits','cpu','gpu','mixed','report','all'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    global_path=ROOT/'docs/results/sequential-v22/.execution.lock'
    global_path.parent.mkdir(parents=True,exist_ok=True)
    with global_path.open('w') as global_lock,(out/'.lock').open('w') as lock:
        fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        phases=['prepare','audits','cpu','gpu','mixed','report'] if args.phase=='all' else [args.phase]
        for phase in phases:
            log('start',phase)
            try:globals()[phase](out)
            except Exception as exc:
                record(out,'failure-'+phase,dict(status='failed',type=type(exc).__name__,reason=str(exc)))
                raise


if __name__=='__main__':main()
