"""Sequential practical research CLI with immutable numerical case protocols."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
from . import config as configuration
from .checkpoint import GenerationStore
from .provenance import PREVIOUS, ROOT, digest, read, resources, serial_lock, sha, source_files, utc, verify, write


def log(*values):
    print(utc(),*values,flush=True)


def load_model(run,cfg):
    cfg=configuration.validate(cfg)
    baseline=verify(run)
    if cfg['device']!='cpu':
        import warp as wp
        wp.config.kernel_cache_dir=str(Path(run).resolve()/'warp-cache')
    from benchmarks.research_sequential.run import open_reduction
    from engine.aniso_phase1.research_sequential_next.model import PracticalModel
    if (Path(run)/'model/model-package.json').exists():
        from .model_package import load_reduction
        r=load_reduction(run)
    else:r=open_reduction(PREVIOUS)
    scenario=cfg.get('scenario',{})
    scenario_audit=None
    if scenario.get('fiber_angle_degrees') is not None:
        from engine.aniso_phase1.research_sequential_next.scenarios import material_reduction
        r,scenario_audit=material_reduction(r,scenario['fiber_angle_degrees'])
    implementation=cfg.get('implementation',{})
    cls=PracticalModel;kwargs={}
    if implementation.get('operator','original')=='segmented':
        from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
        cls=SegmentedModel
    if implementation.get('linearization_cache',False):
        from engine.aniso_phase1.research_sequential_next.linearization import CachedModel
        cls=CachedModel;kwargs=dict(cache_bytes=implementation['cache_bytes'],cache_entries=implementation['cache_entries'])
    model=cls(r,order=cfg['material_order'],device=cfg['device'],**kwargs)
    if scenario.get('peak_m',.005)!=.005:
        from engine.aniso_phase1.research_sequential_next.scenarios import install_peak
        install_peak(model,scenario['peak_m'])
    model.scenario_audit=scenario_audit
    return model,baseline


def case_identity(run,cfg,model,initial_digest):
    return dict(schema='numerical-case-v1',baseline_sha256=sha(Path(run)/'baseline-lock.json'),
        model=model.identity,model_sha256=model.signature,protocol_sha256=configuration.identity(cfg),
        initial_state_digest=initial_digest,numerical_source_sha256=source_files())


def create_config(run,name,dt=.05,end=1.6,start=0.,material_order=7,path_order=2,device='cuda:0',initial=None,
                  operator_mode=None,linearization_cache=False,cache_bytes=3<<30,cache_entries=4,
                  preconditioner='original',fiber_angle=None,peak_m=.005):
    run=Path(run);baseline=verify(run);folder=run/'cases'/name
    if folder.exists():raise ValueError('case already exists; do not overwrite its frozen protocol')
    cfg=configuration.protocol(baseline['energy_scale_J'],dt=dt,start=start,end=end,
        material_order=material_order,device=device,path_order=path_order,operator_mode=operator_mode,
        linearization_cache=linearization_cache,cache_bytes=cache_bytes,cache_entries=cache_entries,
        preconditioner=preconditioner)
    if fiber_angle is not None or peak_m!=.005:
        cfg['scenario']=dict(fiber_angle_degrees=fiber_angle,peak_m=peak_m)
        cfg=configuration.validate(cfg)
    if initial is not None:
        initial=Path(initial).resolve()
        cfg['initial_state']=dict(path=str(initial),sha256=sha(initial))
    elif start!=0:raise ValueError('nonzero start requires an actual committed state')
    write(folder/'execution-protocol.json',cfg)
    log('created case',name,'steps',len(cfg['times'])-1)
    return cfg


def load_initial(cfg,model):
    if 'initial_state' not in cfg:return model.rest()
    entry=cfg['initial_state'];p=Path(entry['path'])
    if sha(p)!=entry['sha256']:raise ValueError('initial-state file changed')
    # Initial files must be from a published, verified generation. Never accept
    # an arbitrary edited JSON as an authenticated shared window initial state.
    generation=p.parent
    manifest=read(generation/'manifest.json')
    if manifest['files'].get(p.name)!=entry['sha256']:raise ValueError('unbound initial state')
    if manifest['identity']['model_sha256']!=model.signature:raise ValueError('initial physical model mismatch')
    folder=generation.parent.parent
    chain=GenerationStore(folder,manifest['identity']).history()
    matches=[item for item in chain if item['folder'].resolve()==generation.resolve()]
    if len(matches)!=1:raise ValueError('initial state is not in a committed generation chain')
    state=matches[0]['state'];model.validate(state,material=True)
    if abs(state.time-cfg['times'][0])>1e-12:raise ValueError('initial time mismatch')
    return state


def probe_frame(model,state,shape):
    axes=[np.linspace(e[0],e[-1],n) for e,n in zip(model.parent.edges,shape)]
    f=model.fields(state,axes)
    return dict(time=np.array(state.time),step=np.array(state.step),**f)


def run_case(run,name,*,stop_after=None):
    from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF,StepRejected
    run=Path(run);folder=run/'cases'/name
    cfg=configuration.validate(read(folder/'execution-protocol.json'))
    start=time.perf_counter();model,baseline=load_model(run,cfg)
    initial=load_initial(cfg,model);identity=case_identity(run,cfg,model,initial.digest())
    identity_path=folder/'identity.json'
    if identity_path.exists():
        if read(identity_path)!=identity:raise ValueError('numerical identity changed; refusing implicit continuation')
    else:
        write(identity_path,identity)
        for source_name,expected in identity['numerical_source_sha256'].items():
            source=ROOT/source_name
            if sha(source)!=expected:raise ValueError('source changed during case publication')
            destination=folder/'source'/source_name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(source.read_bytes())
    store=GenerationStore(folder,identity);loaded=store.load(validator=model.validate)
    if loaded is not None:
        store.history()  # verify every published predecessor, including probes
        initial=loaded['state'];rows=loaded['rows']
    else:rows=[]
    cls=ValidatedAVF
    if cfg.get('implementation',{}).get('preconditioner','original')=='reuse_static':
        from engine.aniso_phase1.research_sequential_next.preconditioner import ReusedAVF
        cls=ReusedAVF
    stepper=cls(model,cfg,initial)
    build_seconds=time.perf_counter()-start
    times=cfg['times'];selected=set(np.linspace(0,len(times)-1,min(cfg['display_frames'],len(times)),dtype=int).tolist())
    initial_step=stepper.state.step-len(rows)
    if loaded is None:store.save(stepper.state,rows,frame=probe_frame(model,stepper.state,cfg['probe_shape']))
    attempted=0;step_seconds=0.;io_seconds=0.;field_seconds=0.
    segment_start=time.perf_counter()
    for index,target in enumerate(times[1:],1):
        state=stepper.state
        if target<=state.time+1e-12:continue
        if stop_after is not None and attempted>=stop_after:break
        log(name,'advance',state.time,'->',target)
        begun=time.perf_counter()
        try:row=stepper.step(target-state.time)
        except StepRejected:
            write(folder/'failure.json',dict(utc=utc(),failures=stepper.failures,last_committed_digest=stepper.state.digest(),
                current_generation=read(store.pointer),source_sha256=source_files()))
            raise
        step_seconds+=time.perf_counter()-begun;attempted+=1
        rows.append(row);state=stepper.state
        begun=time.perf_counter();frame=probe_frame(model,state,cfg['probe_shape']) if index in selected else None
        field_seconds+=time.perf_counter()-begun
        begun=time.perf_counter();store.save(state,rows,frame=frame)
        io_seconds+=time.perf_counter()-begun
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
        log(name,'accepted',state.step,'detF',round(row['min_detF'],6),'residual',f'{row["true_residual"]:.3g}',
            'seconds',round(row['wall_seconds'],3),'rss_GiB',round(rss,3))
        if rss>cfg['resources']['max_rss_GiB']:raise RuntimeError('resource_limited: RSS budget exceeded after safe commit')
        disk=resources()
        if disk['system_free_GiB']<cfg['resources']['min_system_free_GiB']:
            write(folder/'resource-warning.json',dict(utc=utc(),**disk))
            raise RuntimeError('resource_limited: system disk below migration threshold; committed checkpoint retained')
    restored=store.load(validator=model.validate)
    if restored['state'].digest()!=stepper.state.digest():raise ValueError('checkpoint restore differs from accepted state')
    complete=abs(stepper.state.time-times[-1])<1e-12
    record=dict(status='passed_scoped' if complete else 'in_progress',case=name,utc=utc(),
        steps=len(rows),initial_step=initial_step,end_s=stepper.state.time,protocol_sha256=identity['protocol_sha256'],
        material_order=cfg['material_order'],device=cfg['device'],source_sha256=identity['numerical_source_sha256'],
        min_detF=min((r['min_detF'] for r in rows),default=1.),
        max_true_residual=max((r['true_residual'] for r in rows),default=0.),
        max_ledger_closure_J=max((abs(r['budget_defect_J']) for r in rows),default=0.),
        cumulative_path_error_J=stepper.state.child_states.get('cumulative_abs_path_quadrature_error_J',0.),
        cumulative_solve_error_J=stepper.state.child_states.get('cumulative_abs_solve_work_error_J',0.),
        checkpoint_reload=True,internal_steps=len(times)-1,display_frames=len(selected),
        temporal_accuracy_certified=False,spatial_accuracy_certified=False,
        this_segment=dict(build_seconds=build_seconds,advance_seconds=step_seconds,fields_seconds=field_seconds,
            checkpoint_seconds=io_seconds,wall_seconds=time.perf_counter()-segment_start+build_seconds,
            accepted_steps=attempted,peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
    if hasattr(stepper,'factors'):record['preconditioner']=stepper.factors.report()
    if hasattr(model.operator,'memory_budget'):record['gpu_memory']=model.operator.memory_budget.report()
    if hasattr(model,'linearizations'):record['cache']=model.linearizations.report()
    history=read(folder/'segments.json') if (folder/'segments.json').exists() else []
    history.append(record['this_segment']);write(folder/'segments.json',history)
    write(folder/'summary.json',record);write(folder/'ledger.json',rows)
    log(name,record['status'],record['steps'],'steps')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    make=sub.add_parser('configure');make.add_argument('--dt',type=float,default=.05)
    make.add_argument('--start',type=float,default=0.);make.add_argument('--end',type=float,default=1.6)
    make.add_argument('--material-order',type=int,default=7);make.add_argument('--path-order',type=int,default=2)
    make.add_argument('--device',default='cuda:0');make.add_argument('--initial',type=Path)
    make.add_argument('--fiber-angle',type=float)
    make.add_argument('--peak-displacement',type=float,default=.005)
    make.add_argument('--operator',choices=['original','segmented'])
    make.add_argument('--linearization-cache',action='store_true')
    make.add_argument('--cache-mib',type=int,default=3072);make.add_argument('--cache-entries',type=int,default=4)
    make.add_argument('--preconditioner',choices=['original','reuse_static'],default='original')
    cycle=sub.add_parser('cycle');cycle.add_argument('--stop-after',type=int)
    for p in (make,cycle):
        p.add_argument('--run',type=Path,required=True);p.add_argument('--case',required=True)
    args=parser.parse_args()
    if Path(args.case).name!=args.case or args.case in ('.','..'):parser.error('case must be a single directory name')
    if args.command=='cycle' and args.stop_after is not None and args.stop_after<1:parser.error('stop-after must be positive')
    with serial_lock(args.run):
        if args.command=='configure':
            create_config(args.run,args.case,args.dt,args.end,args.start,args.material_order,args.path_order,args.device,args.initial,
                args.operator,args.linearization_cache,args.cache_mib<<20,args.cache_entries,args.preconditioner,args.fiber_angle,args.peak_displacement)
        else:run_case(args.run,args.case,stop_after=args.stop_after)


if __name__=='__main__':main()
