"""Trusted E package, same-point 2D physics, and variable-step AVF transactions."""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import copy
import shutil
import numpy as np
import scipy.sparse as sp
from .provenance import ROOT,COMMON,read,write,sha,check,verify,register_study,serial_lock,utc
from .config import protocol
from .run import load_model
from .checkpoint import GenerationStore
from engine.aniso_phase1.research_e.stage2.handoff import load as load_e
from engine.aniso_phase1.research_e.stage2.d_adapter import consume
from engine.aniso_phase1.research_e.stage2.transaction import ChildAdapter
from engine.aniso_phase1.research_e.stage2.evaluation import StressEvaluator,stress_errors,stress_passed
from engine.aniso_phase1.research_e.stage2.residual import step_residual
from engine.aniso_phase1.research_sequential.solver import GeneralBiot,solve_mixed
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF,StepRejected
from engine.aniso_phase1.types import AnisotropicMaterialParams
from engine.aniso_phase1.research_e.flow import Grid,Darcy
from engine.aniso_phase1.research_e.poro import Skeleton,Biot
from benchmarks.research_e.stage2.experiments import make_model,observed_step,summary
from benchmarks.research_e.experiments import bc_all,load_factor,relative
from benchmarks.research_e.reference import biot_manufactured,sine_fields

E=ROOT/'docs/results/parallel-v22-stage2/E/20260930T071848Z_E_stage2'
PARENT='55682a7b8e90b62c1306818cdf9c1f060b4286a174da069ce2217aa5b53b3c7c'


def trusted_package(run,folder=None):
    baseline=verify(run);manifest=E/'artifact-sha256.json';relative_path=str(manifest.relative_to(ROOT))
    if baseline['input_sha256'].get(relative_path)!=sha(manifest):raise ValueError('E manifest is not the externally frozen input')
    artifacts=read(manifest);check(E,artifacts)
    extension=E/'extension-source-sha256.json';check(ROOT,read(extension))
    expected_hash=artifacts['handoff/operator-package.json'];trusted=E/'handoff/operator-package.json'
    if sha(trusted)!=expected_hash:raise ValueError('trusted E package content changed')
    expected=read(trusted)
    if expected['parent_bundle_sha256']!=PARENT:raise ValueError('wrong E parent')
    return (*load_e(E/'handoff' if folder is None else folder,expected_manifest_sha256=expected_hash,expected=expected),
            dict(artifact_manifest_sha256=sha(manifest),package_sha256=expected_hash,verified_artifacts=len(artifacts),
                 extension_source_sha256=sha(extension),parent_bundle_sha256=PARENT))


def manufactured(out):
    old_backend=make_model(16,clamped=True);solver=GeneralBiot(old_backend.solid,old_backend.flow,old_backend.storage)
    s,g=solver.solid,solver.flow.grid;e=StressEvaluator(s)
    ue,pe,force,source=biot_manufactured(s.material,solver.flow.K[0],solver.storage)
    load=s.body_force(force);source0=g.average(lambda x:source(x,0.));source1=g.average(lambda x:source(x,1.))-source0
    old=solver.initial();reference_old=old_backend.initial();bc=bc_all(2);records=[];local=[];differences=[]
    for j in range(1,11):
        t=j*.01;src=source0+t*source1;r=solver.step(old,.01,t*load,bc,src)
        original=old_backend.step(reference_old,.01,t*load,bc,src)
        values,row,mass=observed_step(solver,old,r,.01,t*load,bc,e,src)
        differences.append(dict(time=t,u=relative(r.state.u,original.state.u),p=relative(r.state.p,original.state.p),flux=relative(r.flux,original.flux)))
        records.append(row);local.append(mass);old=r.state;reference_old=original.state
    T=.1;pex=T*g.average(pe,order=6);uex=T*np.array([ue(x) for x in s.nodes]).ravel()
    analytic=e.reference(lambda x:T*np.outer(.0001*np.array([1.,.5]),sine_fields(x)[1]),lambda x:T*pe(x))
    stress=stress_errors(e,values,analytic);ep=relative(old.p,pex);eu=relative(old.u,uex);ledgers=summary(records)
    passed=ep<.04 and eu<.04 and stress_passed(stress,.04) and ledgers['passed'] and all(max(d[k] for k in ['u','p','flux'])<1e-8 for d in differences)
    np.savez_compressed(out/'manufactured-general.npz',u=old.u,p=old.p,flux=r.flux,u_reference=uex,p_reference=pex,
        points=e.points,weights=e.dV,local_mass_defects=local,**values,**{'reference_'+k:v for k,v in analytic.items()})
    return dict(passed=bool(passed),grid=[16,16],steps=10,dt=.01,pressure_error=ep,displacement_error=eu,
        stress=stress,ledger=ledgers,original_backend_differences=differences,records=records,
        scope='same-point full 3x3 embedded total/effective stresses and cell means; fixed independent 2D model')


def study(run):
    run=Path(run);out=run/'N11'
    register_study(run,'N11/protocol.json',dict(utc=utc(),E_trusted_artifact_manifest='7220f2c7e7b549ad88b5331a7a95fb30eb1ac0299c23fea210e8cc45bf560301',
        manufactured_grid=16,manufactured_dt=.01,manufactured_steps=10,
        joint_dts=[.0005,.00075,.00025,.001],faults=['before_material','after_prepare','before_commit','before_pointer'],
        independent_2D=True,physical_3D_coupling=False,mixed_solver='general direct',block_residual_limit=1e-8))
    metadata,M,v,residual,trust=trusted_package(run)
    contract=consume(out/'D-mixed',metadata,M,v,ROOT/'engine/aniso_phase1/research_d/stage2/contracts.py')
    solution,info=solve_mixed(M,v['rhs']);solution_error=relative(solution,v['solution'])
    corrupt_rejected=False
    with TemporaryDirectory(prefix='wrong-E-',dir=out.resolve()) as directory:
        folder=Path(directory)
        for path in (E/'handoff').glob('*'):
            if path.is_file():shutil.copyfile(path,folder/path.name)
        content=read(folder/'operator-package.json');content['dt']=.02;write(folder/'operator-package.json',content)
        try:trusted_package(run,folder)
        except ValueError:corrupt_rejected=True
    cfg=metadata['config'];mat=cfg['material'];grid=Grid(tuple(cfg['shape']),cfg['lengths'])
    solid=Skeleton(grid,AnisotropicMaterialParams(mat['mu'],mat['lam'],mat['k_f'],mat['direction']),cfg['alpha'])
    flow=Darcy(grid,np.array(cfg['K']),viscosity=cfg['viscosity']);solver=GeneralBiot(solid,flow,cfg['storage'])
    if not np.array_equal(solid.fixed,cfg['fixed_u']) or not np.array_equal(flow.gravity_rhs,cfg['gravity_rhs']):raise ValueError('E boundary/gravity reconstruction differs')
    rebuilt,frhs,ffree,*_=solver._system(solver.initial(),metadata['dt'],v['load'],{(a,b):(k,x) for a,b,k,x in metadata['boundary']},0.)
    blocks={name:float(np.max(abs((actual-sp.load_npz(E/'handoff'/f'{name}.npz')).data),initial=0.))
        for name,actual in [('A',solid.A),('G',solid.G),('C',solver.C),('B',flow.B),('H',flow.H)]}
    if max(blocks.values())>1e-12:raise ValueError('E rebuilt physical blocks differ')
    physics=manufactured(out)
    if not physics['passed']:raise ValueError('new general solver failed existing E physics budget')
    print('E same-point manufactured pressure/u errors',physics['pressure_error'],physics['displacement_error'],flush=True)
    base_cfg=protocol(read(run/'baseline-lock.json')['energy_scale_J'],dt=.025)
    model,_=load_model(run,base_cfg);adapter=ChildAdapter(solver,PARENT);independent=Biot(solid,flow,cfg['storage'])
    initial=model.rest();initial.child_states['E']=adapter.initial();initial.child_states['B']=dict(material=model.rule.signature,mass='original-q5',fixed_during_trial=True)
    # Keep E's own ramp/hold/unload law and boundary; only the clock is shared.
    boundary={(a,b):(kind,value) for a,b,kind,value in metadata['boundary']};base_load=solid.traction(0,1,[-.1,0])
    def prepare(candidate,values):
        dt=candidate.time-values['E']['time']
        if not np.isfinite(dt) or dt<=0:raise ValueError('nonpositive E child increment')
        values['E']=adapter.prepare(values['E'],dt=dt,load=base_load*load_factor(candidate.time),boundary=boundary)
        return values
    rejected={};parent=ValidatedAVF(model,base_cfg,initial);before=parent.state.digest()
    for where in ['before_material','after_prepare','before_commit']:
        def fail(place,state):
            if place==where:raise ValueError('intentional joint failure '+where)
        try:parent.step(.0005,prepare_children=prepare,validate_trial=adapter.validate_parent,inject=fail);rejected[where]=False
        except StepRejected:rejected[where]=parent.state.digest()==before
    if not all(rejected.values()):raise ValueError('joint rollback failed')
    # A stale child must be rejected, then the same physical step may be retried.
    try:parent.step(.0005,prepare_children=lambda candidate,values:values,validate_trial=adapter.validate_parent);rejected['stale_child']=False
    except StepRejected:rejected['stale_child']=parent.state.digest()==before
    identity=dict(model=model.identity,child=adapter.signature,config=base_cfg,protocol_sha256=sha(out/'protocol.json'))
    store=GenerationStore(out/'joint-checkpoints',identity);store.save(initial,[])
    state_e=independent.initial();rows=[];child_comparisons=[]
    for h in [.0005,.00075,.00025,.001]:
        row=parent.step(h,prepare_children=prepare,validate_trial=adapter.validate_parent);rows.append(row)
        t=parent.state.time;r=independent.step(state_e,h,base_load*load_factor(t),boundary);state_e=r.state
        child=parent.state.child_states['E'];diff={k:relative(np.asarray(child[k]),value) for k,value in [('u',r.state.u),('p',r.state.p),('flux',r.flux)]}
        child_comparisons.append(dict(time=t,dt=h,differences=diff,residual=step_residual(independent,
            type(state_e)(np.array(child['last_step']['old_u']),np.array(child['last_step']['old_p']),child['last_step']['old_time']),r,h,base_load*load_factor(t),boundary)))
        if max(diff.values())>1e-8:raise ValueError('joint E differs from independent same-history advance')
        if len(rows)==1:
            def fail_publish(where):
                if where=='before_pointer':raise OSError('intentional publication failure')
            try:store.save(parent.state,rows,inject=fail_publish)
            except OSError:pass
            rejected['unpublished_state']=store.load()['state'].digest()==initial.digest()
            # Reconstruct from the durable parent and retry; no orphan adoption.
            restart=ValidatedAVF(model,base_cfg,store.load()['state']);restart.step(h,prepare_children=prepare,validate_trial=adapter.validate_parent)
            for attr in ['q','velocity','predictor']:
                if not np.array_equal(getattr(restart.state,attr),getattr(parent.state,attr)):raise ValueError('joint crash retry differs')
        store.save(parent.state,rows)
        loaded=store.load(validator=adapter.validate_parent)
        if loaded['state'].digest()!=parent.state.digest():raise ValueError('joint checkpoint content differs')
        # Every next step really starts from the serialized complete generation.
        parent=ValidatedAVF(model,base_cfg,loaded['state'])
    try:store.save(parent.state,rows);rejected['duplicate_publication']=False
    except ValueError:rejected['duplicate_publication']=True
    # Deliberately mismatched stored identity is rejected without file mutation.
    try:GenerationStore(out/'joint-checkpoints',dict(identity,child='foreign')).load();rejected['foreign_checkpoint']=False
    except ValueError:rejected['foreign_checkpoint']=True
    if not all(rejected.values()):raise ValueError('joint stale/persistence gate failed')
    inherited={name:dict(sha256=sha(E/(name+'.json')),passed=read(E/(name+'.json'))['passed'])
               for name in ['E5_consolidation','E5_manufactured','E6_pressure','E7_cycle','E8_robustness']}
    result=dict(utc=utc(),status='passed_scoped',trusted_package=trust,corrupt_package_rejected=corrupt_rejected,
        reconstructed_block_max_abs=blocks,full_system_residual=residual,general_solution=info,general_solution_relative=solution_error,
        D_contract=contract,physical_recheck=physics,inherited_unchanged_E_evidence=inherited,
        joint=dict(accepted_steps=parent.state.step,end_s=parent.state.time,dt=[r['dt'] for r in rows],
            rejection_controls=rejected,independent_child_comparisons=child_comparisons,C_ledger=rows,E_ledger=parent.state.child_states['E']['ledger'],
            durable_generations=len(store.history()),reload_after_every_step=True),
        capabilities=dict(independent_2D_physics=True,variable_step_transaction=True,physical_3D_coupling=False),
        scope='independent CPU 2D E model and GPU C model share transaction clock, not physical interface forces or fluxes')
    if not corrupt_rejected or not contract['passed'] or solution_error>1e-8:raise ValueError('trusted mixed contract gate failed')
    write(out/'result.json',result);caps=read(run/'capabilities.json');caps['N11']=dict(status='passed_scoped',evidence='N11/result.json',physical_3D_coupling=False);write(run/'capabilities.json',caps)
    print('E independent physics and four variable-step parent/child commits passed',flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
