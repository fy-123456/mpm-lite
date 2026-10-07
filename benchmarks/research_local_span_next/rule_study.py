"""Current-space certificates from actual sufficient states, exact time paths."""
from pathlib import Path
import argparse,copy
import numpy as np
from .provenance import APP,read,write,sha,digest,register,source_files,verify,serial_lock,utc
from .spaces import load_selected
from .run import create_config,load_model,make_stepper
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.diagnostics import weak_moments
from benchmarks.research_sequential_next.material_study import material_metrics,compare_fields
from benchmarks.research_sequential_next.compare import metric
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_sequential_next.scenarios import install_peak
from engine.aniso_phase1.research_phase_stress_next.warm import install_warm


def sensitive_prefix(run):
    run=Path(run);verify(run)
    register(run,'S3/evidence-vs-permission.json',dict(parent_main_sha256=sha(APP/'S6/qualification-final.json'),parent_window_sha256=sha(APP/'S3/window-final-0060-0065.json'),
        selected_space_changed=read(run/'selected-space.json')['package']['sha256']!=read(APP/'selected-space.json')['package']['sha256'],
        current_permission='full_only until current model/time/source checks',historical_windows=[[.5,.55],[.6,.65],[.7,.75]],old_pending_not_current=True,sensitive_full_cycle_q5=False))
    create_config(run,'sensitive-prefix',end=.75,peak_m=.0075,field_cache=True,display_frames=3)


def certify(run,source,grid,peak,samples,path,*,rest=False):
    import warp as wp
    run=Path(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');choice=read(run/'selected-space.json');order=choice['full_order'];r,_=load_selected(choice['package'])
    Model=SegmentedModel
    if (run/'S4/performance-decision.json').exists() and read(run/'S4/performance-decision.json').get('reuse_transpose_buffers',False):
        from engine.aniso_phase1.research_local_span_next.reuse import ReusedSegmentedModel as Model
    models={o:install_warm(install_peak(Model(r,order=o,device='cuda:0'),peak)) for o in (5,order,order+1)}
    if not all(np.array_equal(models[order].M,m.M) for m in models.values()):raise ValueError('material compression changed mass')
    h=history(source);lookup={round(x['state'].time,10):x for x in h};m=models[order]
    if read(source/'identity.json')['model']!=m.identity:raise ValueError('sufficient trajectory model differs')
    modal=read(run/'S2/modal-observable-map.json')
    if modal['space']!=r.signature:raise ValueError('stale modal basis')
    j=max(modal['selected_modes'],key=lambda x:x['estimated_output_peak_Pa'])['mode']
    with np.load(run/'S2/modal-basis.npz') as z:vector=z['vectors'][:,j].copy()
    rng=np.random.default_rng(20261001);mixed=rng.normal(size=m.rest().q.shape);mixed[m.fixed]=0;mixed/=np.linalg.norm(mixed);sensitive=np.zeros_like(mixed);sensitive[m.free]=vector.reshape(-1,3);sensitive/=np.linalg.norm(sensitive)
    results=[]
    for t in samples:
        item=lookup[round(t,10)];q=item['state'].q;weak={o:weak_moments(z,q) for o,z in models.items()}
        for label,d in [('mixed',mixed),('sensitive',sensitive)]:
            values={o:z.evaluate(q,d) for o,z in models.items()}
            results.append(dict(time_s=t,state_sha256=sha(item['folder']/'state.json'),direction=label,
                sufficient=material_metrics(values[order],values[order+1],weak[order],weak[order+1]),compressed=material_metrics(values[5],values[order],weak[5],weak[order])))
    sufficient=all(x['sufficient']['passed'] for x in results);qualified=sufficient and all(x['compressed']['passed'] for x in results)
    q=dict(schema='basis-allocation-q5-qualification-v1',utc=utc(),qualified=qualified,sufficient_qualified=sufficient,numerical_source_sha256=source_files(),
        reduction_sha256=r.signature,mass_sha256=digest(m.M.tolist()),compressed_model=models[5].identity,full_model=m.identity,source_case=str(source),
        evidence=dict(source_identity_sha256=sha(source/'identity.json'),results=results,sensitive_mode=j),
        scope=dict(time_grid_s=grid,max_dt_s=float(max(np.diff(grid))),peak_m=peak,fiber_angle_degrees=45.,physical_space_sha256=choice['package']['sha256'],mass_order=choice['mass_order'],full_order=order,rest_start=rest,
            initial_states=[dict(time_s=t,sha256=sha(lookup[round(t,10)]['folder']/'state.json')) for t in grid[:-1]]),
        scope_limit='exact selected solid space, time nodes, source and authenticated full initial; no coupled q5 or sensitive full cycle')
    write(run/path,q)
    if not sufficient:raise ValueError('full material rule insufficient; no compressed permission')
    return q


def windows_prepare(run):
    run=Path(run);source=run/'cases/sensitive-prefix'
    register(run,'S3/window-protocol.json',dict(peak_m=.0075,windows=[[.6,.65],[.65,.7]],full_prefix_steps=60,steps_per_window=4,restart_after=2,new_space=True,source_identity_sha256=sha(source/'identity.json')))
    lookup={round(x['state'].time,10):x for x in history(source)};records=[]
    for i,(a,b) in enumerate(((.6,.65),(.65,.7))):
        grid=[round(a+k*.0125,10) for k in range(5)];path=f'S3/window-{i}-final.json';q=certify(run,source,grid,.0075,[a,(a+b)/2,b],path)
        cfg=create_config(run,f'sensitive-window-{i}',start=a,end=b,peak_m=.0075,initial=lookup[a]['folder']/'state.json',rule_policy='q5_with_full_retry',qualification_path=run/path,field_cache=True,display_frames=3)
        outside=create_config(run,f'sensitive-outside-{i}',start=a,end=b+.0125,peak_m=.0075,initial=lookup[a]['folder']/'state.json',rule_policy='q5_with_full_retry',qualification_path=run/path,field_cache=True,display_frames=2)
        if outside['post_release']['rule_policy']!='full_only':raise ValueError('window permission escaped certified nodes')
        records.append(dict(window=[a,b],qualified=q['qualified'],actual_policy=cfg['post_release']['rule_policy'],certificate=path,outside_policy='full_only'))
    write(run/'S3/local-material-check.json',dict(status='passed_scoped',records=records));write(run/'S3/scope-contract.json',dict(contiguous_times=True,source_bound=True,model_bound=True,whole_step_rollback=True,energy_bridge=True,coupled_q5=False,sensitive_full_cycle_q5=False))
    print('WINDOWS_READY',records,flush=True)


def windows_check(run):
    run=Path(run);source=run/'cases/sensitive-prefix';lookup={round(x['state'].time,10):x for x in history(source)};records=[]
    for i in range(2):
        folder=run/'cases'/f'sensitive-window-{i}';cfg=read(folder/'execution-protocol.json');m,_=load_model(run,cfg);c=make_stepper(run,cfg,m,m.rest());h=history(folder);checks=[]
        for item in h[1:]:
            ref=lookup[round(item['state'].time,10)];fields=compare_fields(getattr(c,'full',m),item['state'],ref['state'],cfg);reaction=metric(item['rows'][-1]['reaction_N'],ref['rows'][-1]['reaction_N'],1e-4,.05)
            checks.append(dict(time_s=item['state'].time,regions=fields,reaction=reaction,passed=reaction['passed'] and all(x['passed'] for region in fields.values() for x in region.values())))
        initial=h[0]['state'];ref0=lookup[round(initial.time,10)]['state'];unchanged=all(np.array_equal(getattr(initial,k),getattr(ref0,k)) for k in ('q','velocity','predictor'))
        if len(h)!=5 or not unchanged or not all(x['passed'] for x in checks):raise ValueError('window fields/initial failed')
        records.append(dict(window=cfg['times'][::4],initial_unchanged=True,rule_switch_J=initial.child_states.get('material_scope_bridge',{}).get('material_energy_difference_J',0.),checks=checks,
            actual_new_process_restart=len(read(folder/'segments.json'))>=2,certificate=cfg['post_release'].get('qualification'),rule_policy=cfg['post_release']['rule_policy']))
    if not records[0]['actual_new_process_restart']:raise ValueError('real restart not demonstrated')
    write(run/'S3/window-runtime-check.json',dict(status='passed_scoped',records=records,numeric_sources=source_files(),sensitive_full_cycle_q5=False));print('WINDOWS_PASSED',[x['rule_switch_J'] for x in records],flush=True)


def qualification(run):
    run=Path(run);source=run/'cases/final-full';cfg=read(source/'execution-protocol.json');register(run,'S6/qualification-protocol.json',dict(source=str(source),samples=[0.,.5,1.1],independent_sample=.8,actual_grid=cfg['times'],mass_order=cfg['mass_order']))
    q=certify(run,source,cfg['times'],.005,[0.,.5,1.1,.8],'S6/qualification-final.json',rest=True)
    write(run/'S6/material-decision.json',dict(status='qualified_main' if q['qualified'] else 'full_only',sufficient_qualified=q['sufficient_qualified'],compressed_qualified=q['qualified'],sensitive_full_cycle=False,coupled_q5=False))
    print('MAIN_QUALIFIED',q['qualified'],flush=True)

def refresh_windows(run):
    run=Path(run);source=run/'cases/sensitive-prefix';lookup={round(x['state'].time,10):x for x in history(source)}
    previous=read(run/'S3/window-runtime-check.json');records=[]
    for i,(a,b) in enumerate(((.6,.65),(.65,.7))):
        old=run/f'S3/window-{i}-final.json';path=f'S3/window-{i}-release.json'
        q=certify(run,source,[round(a+k*.0125,10) for k in range(5)],.0075,[a,(a+b)/2,b],path)
        records.append(dict(window=[a,b],old_certificate_sha256=sha(old),new_certificate=path,qualified=q['qualified']))
    create_config(run,'sensitive-window-release',start=.6,end=.65,peak_m=.0075,initial=lookup[.6]['folder']/'state.json',rule_policy='q5_with_full_retry',qualification_path=run/'S3/window-0-release.json',field_cache=True,display_frames=3)
    write(run/'S3/source-refresh.json',dict(status='material_rechecked_pending_runtime',records=records,numeric_sources=source_files(),
        unchanged_equations=True,operator_equivalence_sha256=sha(run/'S4/operator-equivalence.json'),original_runtime_sha256=sha(run/'S3/window-runtime-check.json'),old_states_not_rewritten=True))

def refresh_check(run):
    run=Path(run);folder=run/'cases/sensitive-window-release';cfg=read(folder/'execution-protocol.json');m,_=load_model(run,cfg);h=history(folder);old=history(run/'cases/sensitive-window-0');errors={}
    for k in ('q','velocity','predictor'):errors[k]=max(float(np.max(abs(getattr(x['state'],k)-getattr(y['state'],k)))) for x,y in zip(h,old))
    if len(h)!=5 or max(errors.values())>1e-8 or len(read(folder/'segments.json'))<2:raise ValueError('final window continuation differs')
    record=read(run/'S3/source-refresh.json');record.update(status='passed_scoped',current_window_errors=errors,actual_new_process_restart=True,current_case=str(folder),numeric_sources=source_files())
    write(run/'S3/window-runtime-final.json',record);print('FINAL_WINDOW_REFRESH',errors,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prefix','window-prepare','window-check','qualification','refresh','refresh-check']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):{'prefix':sensitive_prefix,'window-prepare':windows_prepare,'window-check':windows_check,'qualification':qualification,'refresh':refresh_windows,'refresh-check':refresh_check}[a.phase](a.run)
