"""Content-bound post-release CLI, fixed explicit times and atomic generations."""
from pathlib import Path
import argparse
import copy
import resource
import time
import numpy as np
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.run import load_initial as inherited_initial, probe_frame
from engine.aniso_phase1.research_basis_allocation_next.rules import permission,rule_branch
from benchmarks.research_sequential_next.model_package import load_reduction
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF, StepRejected
from engine.aniso_phase1.research_post_release.runtime_rules import RuntimeRetry,advance_publish
from . import config
from .provenance import PARENT, ROOT, read, write, sha, digest, utc, verify, source_files, snapshot, serial_lock, resources


def load_model(run, cfg, *, shared=None):
    cfg=config.validate(cfg);lock=verify(run)
    import warp as wp
    wp.config.kernel_cache_dir=str(Path(run).resolve()/'warp-cache')
    from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
    if cfg.get('implementation',{}).get('reuse_transpose_buffers',False):
        from engine.aniso_phase1.research_local_span_next.reuse import ReusedSegmentedModel as SegmentedModel
    from .spaces import load_selected
    if shared is None:
        reduction,package=load_selected(cfg.get('physical_space'))
    else:
        reduction,package=shared.acquire(cfg['physical_space'],cfg['device'])
    if cfg['mass_order']!=package['mass_order']:raise ValueError('declared mass order differs from loaded space')
    if cfg['post_release'].get('full_order',7)!=package['full_order']:raise ValueError('declared sufficient rule differs from loaded package')
    model=SegmentedModel(reduction,order=cfg['material_order'],device=cfg['device'])
    if cfg.get('cost_phase',{}).get('shared_reduction',False):
        from engine.aniso_phase1.research_cost_phase_next.shared import SharedReduction
        model._shared_space=shared or SharedReduction(cfg['physical_space'],reduction,package,cfg['device'])
    from engine.aniso_phase1.research_sequential_next.scenarios import install_peak
    install_peak(model,cfg.get('scenario',{}).get('peak_m',.005))
    if cfg.get('implementation',{}).get('force_only_responses',False):
        from engine.aniso_phase1.research_reference_next.force_only import install_force_only
        install_force_only(model)
    if cfg.get('implementation',{}).get('coarse_instrumentation',False):
        from engine.aniso_phase1.research_phase_stress_next.warm import install_warm
        install_warm(model)
    if cfg['post_release']['rule_policy']!='full_only':
        entry=cfg['post_release']['qualification'];path=Path(entry['path'])
        if sha(path)!=entry['sha256']:raise ValueError('q5 qualification changed')
        q=read(path)
        if not q['qualified'] or q['reduction_sha256']!=model.reduction.signature:raise ValueError('unqualified physical space')
        if q['compressed_model']!=model.identity:raise ValueError('q5 material/boundary/device differs from qualification')
        if q['mass_sha256']!=digest(model.M.tolist()):raise ValueError('q5 qualification mass mismatch')
        allowed,reason=permission(q,cfg,numerical_sources=source_files())
        if not allowed:raise ValueError('q5 permission rejected: '+reason)
    return model, lock


def make_stepper(run,cfg,model,state):
    if cfg['post_release']['rule_policy']=='full_only':return ValidatedAVF(model,cfg,state)
    fullcfg=copy.deepcopy(cfg);fullcfg['material_order']=cfg['post_release'].get('full_order',7);fullcfg['post_release']['rule_policy']='full_only'
    full,_=load_model(run,fullcfg,shared=getattr(model,'_shared_space',None))
    entry=cfg['post_release']['qualification'];certificate=Path(entry['path'])
    if sha(certificate)!=entry['sha256'] or read(certificate)['full_model']!=full.identity:raise ValueError('full retry model differs from qualification')
    if cfg['post_release']['rule_policy']=='q5_fixed':
        stepper=ValidatedAVF(model,cfg,state);stepper.full=full
        return stepper
    return RuntimeRetry(model,full,cfg,state)


def case_identity(run,cfg,model,state):
    return dict(schema='cross-direction-case-v1',input_lock_sha256=sha(Path(run)/'input-lock.json'),
        parent_release_sha256=read(Path(run)/'input-lock.json')['parent_release_sha256'],
        model=model.identity,model_sha256=model.signature,protocol_sha256=config.identity(cfg),
        initial_state_digest=state.digest(),numerical_source_sha256=source_files())


def create_config(run,name,**kwargs):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('sealed release; fork it before configuring a new case')
    lock=verify(run);folder=run/'cases'/name
    choice=read(run/'selected-space.json')
    kwargs.setdefault('space',choice.get('package'))
    kwargs.setdefault('mass_order',choice['mass_order'])
    kwargs.setdefault('full_order',choice['full_order'])
    decision=run/'S4/performance-decision.json'
    kwargs.setdefault('shared_reduction',True)
    kwargs.setdefault('coarse_instrumentation',True)
    kwargs.setdefault('reuse_transpose_buffers',read(decision).get('reuse_transpose_buffers',False) if decision.exists() else True)
    if Path(name).name!=name or name in ('.','..'):raise ValueError('single case name required')
    if folder.exists():raise ValueError('case exists; create a new identity')
    initial=kwargs.pop('initial',None)
    if initial is not None:
        initial=Path(initial).resolve();initial=dict(path=str(initial),sha256=sha(initial))
    requested=kwargs.get('rule_policy','full_only')
    explicit=kwargs.pop('qualification_path',None)
    path=Path(explicit).resolve() if explicit is not None else (run/'S6/qualification-final.json').resolve()
    # Construct a valid provisional full config, then explicitly authorize compression.
    kwargs['rule_policy']='full_only'
    cfg=config.make(lock['energy_scale_J'],initial=initial,**kwargs)
    reason='full material requested'
    if requested!='full_only':
        if path.exists():
            qualify=read(path)
            allowed,reason=permission(qualify,cfg,numerical_sources=source_files())
            if allowed:
                cfg['material_order']=5;cfg['post_release']['rule_policy']=requested
                cfg['post_release']['qualification']=dict(path=str(path),sha256=sha(path))
        elif explicit is not None:raise ValueError('explicit certificate does not exist')
        else:reason='current main trajectory certificate not available'
    cfg['post_release']['requested_rule_policy']=requested
    cfg['post_release']['permission_reason']=reason
    if requested!=cfg['post_release']['rule_policy']:cfg['post_release']['scope_fallback']=reason
    config.validate(cfg)
    write(folder/'execution-protocol.json',cfg)
    return cfg


def load_initial(cfg,model,full=None):
    if not cfg.get('initial_state') or cfg['post_release']['rule_policy']=='full_only':
        return inherited_initial(cfg,model)
    if full is None:raise ValueError('full model required for authenticated material branch')
    state=inherited_initial(cfg,full)
    return rule_branch(state,model,full,cfg,cfg['initial_state'])


def run_case(run,name,*,stop_after=None,quiet=False):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('sealed release; fork it before advancing a case')
    folder=run/'cases'/name;cfg=config.validate(read(folder/'execution-protocol.json'))
    start=time.perf_counter();model,lock=load_model(run,cfg)
    bootstrap=make_stepper(run,cfg,model,model.rest())
    initial=load_initial(cfg,model,getattr(bootstrap,'full',None))
    stepper=(RuntimeRetry(model,bootstrap.full,cfg,initial) if isinstance(bootstrap,RuntimeRetry) else ValidatedAVF(model,cfg,initial)) if cfg.get('initial_state') else bootstrap
    identity=case_identity(run,cfg,model,initial)
    if isinstance(stepper,RuntimeRetry):identity['controller']=stepper.identity
    validator=stepper.validate if isinstance(stepper,RuntimeRetry) else model.validate
    ip=folder/'identity.json'
    if ip.exists():
        if read(ip)!=identity:raise ValueError('source/protocol/model changed; refusing continuation')
    else:
        snapshot(folder/'source',identity['numerical_source_sha256']);write(ip,identity)
    store=GenerationStore(folder,identity);loaded=store.load(validator=validator)
    if loaded is not None:store.history();initial=loaded['state'];rows=loaded['rows']
    else:rows=[]
    if loaded is not None:
        stepper=RuntimeRetry(model,stepper.full,cfg,initial) if isinstance(stepper,RuntimeRetry) else ValidatedAVF(model,cfg,initial)
    times=cfg['times'];selected=set(np.linspace(0,len(times)-1,min(cfg['display_frames'],len(times)),dtype=int))
    probe_cache=None
    if cfg['post_release'].get('field_cache',False):
        from engine.aniso_phase1.research_post_release.fields import CachedProbes
        probe_cache=CachedProbes(model,tuple(cfg['probe_shape']))
    frame_at=probe_cache.frame if probe_cache is not None else lambda state:probe_frame(model,state,cfg['probe_shape'])
    build=time.perf_counter()-start
    if loaded is None:store.save(stepper.state,rows,frame=frame_at(stepper.state))
    step_seconds=field_seconds=io_seconds=0.;attempted=0
    for index,target in enumerate(times[1:],1):
        if target<=stepper.state.time+1e-12:continue
        if stop_after is not None and attempted>=stop_after:break
        tick=time.perf_counter()
        try:row=advance_publish(stepper,store,rows,target-stepper.state.time,
                frame_builder=frame_at if index in selected else None)
        except StepRejected as error:
            write(folder/'failure.json',dict(utc=utc(),error=str(error),failures=getattr(stepper,'failures',[]),last_committed_digest=stepper.state.digest()))
            raise
        step_seconds+=time.perf_counter()-tick
        rows.append(row);attempted+=1;state=stepper.state
        # advance_publish includes recovery and publication; disjoint detailed
        # partitions are measured separately by the S5 profiling study.
        if not quiet and (index%8==0 or index==len(times)-1):
            print(utc(),name,state.step,round(state.time,5),'detF',round(row['min_detF'],6),flush=True)
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20>cfg['resources']['max_rss_GiB']:
            raise RuntimeError('RSS limit reached after safe commit')
        if resources()['system_free_GiB']<5:
            write(folder/'resource-warning.json',dict(utc=utc(),**resources()))
            raise RuntimeError('system disk below 5 GiB; migrate before resuming')
    restored=store.load(validator=validator)
    if restored['state'].digest()!=stepper.state.digest():raise ValueError('checkpoint differs from accepted state')
    report=dict(status='passed_scoped' if abs(stepper.state.time-times[-1])<1e-12 else 'in_progress',
        utc=utc(),case=name,steps=len(rows),end_s=stepper.state.time,
        source_sha256=identity['numerical_source_sha256'],protocol_sha256=identity['protocol_sha256'],
        min_detF=min((r['min_detF'] for r in rows),default=1.),
        max_ledger_closure_J=max((abs(r['budget_defect_J']) for r in rows),default=0.),
        max_true_residual=max((r['true_residual'] for r in rows),default=0.),
        rule_policy=cfg['post_release']['rule_policy'],active_rule=stepper.state.child_states['identity']['material'],
        material_attempts=sum(r.get('material_attempts',1) for r in rows),
        cumulative_rule_switch_error_J=stepper.state.child_states.get('cumulative_abs_rule_switch_error_J',0.),
        display_frames=len(selected),checkpoint_reload=True,temporal_certified=False,spatial_certified=False,
        this_segment=dict(build_seconds=build,advance_and_publish_seconds=step_seconds,
            includes_fields_and_checkpoint=True,wall_seconds=time.perf_counter()-start,accepted_steps=attempted,
            peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
    if probe_cache is not None:report['field_cache']=dict(bytes=probe_cache.bytes,build_seconds=probe_cache.build_seconds,signature=probe_cache.signature)
    if hasattr(model.operator,'memory_budget'):report['gpu_memory']=model.operator.memory_budget.report()
    segments=read(folder/'segments.json') if (folder/'segments.json').exists() else []
    segments.append(report['this_segment']);write(folder/'segments.json',segments)
    write(folder/'summary.json',report);write(folder/'ledger.json',rows)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    make=sub.add_parser('configure');make.add_argument('--dt',type=float,default=.0125)
    make.add_argument('--start',type=float,default=0.);make.add_argument('--end',type=float,default=1.6)
    make.add_argument('--times-json',type=Path,help='JSON array of explicit strictly increasing times')
    make.add_argument('--initial',type=Path)
    make.add_argument('--qualification',type=Path,help='explicit authenticated window certificate; outside scope uses full rule')
    make.add_argument('--peak-displacement',type=float,default=.005)
    make.add_argument('--force-only',action='store_true')
    make.add_argument('--field-cache',action='store_true',help='bounded fixed-probe geometry cache')
    make.add_argument('--rule-policy',choices=['full_only','q5_fixed','q5_with_full_retry'],default='full_only')
    cycle=sub.add_parser('cycle');cycle.add_argument('--stop-after',type=int)
    for p in (make,cycle):p.add_argument('--run',type=Path,required=True);p.add_argument('--case',required=True)
    args=parser.parse_args()
    if Path(args.case).name!=args.case or args.case in ('.','..'):parser.error('single case name required')
    if args.command=='cycle' and args.stop_after is not None and args.stop_after<1:parser.error('positive stop-after required')
    with serial_lock(args.run):
        if args.command=='configure':
            create_config(args.run,args.case,dt=args.dt,start=args.start,end=args.end,
                          times=read(args.times_json) if args.times_json else None,initial=args.initial,qualification_path=args.qualification,rule_policy=args.rule_policy,field_cache=args.field_cache,peak_m=args.peak_displacement,force_only=args.force_only)
        else:
            result=run_case(args.run,args.case,stop_after=args.stop_after)
            print({k:result[k] for k in ('status','case','steps','end_s','min_detF','this_segment')},flush=True)


if __name__=='__main__':main()
