"""Current-space certificates from actual sufficient states, exact time paths."""
from pathlib import Path
import argparse,copy,shutil
import numpy as np
from .provenance import APP,LOCAL,read,write,sha,digest,register,source_files,verify,serial_lock,utc
from .spaces import load_selected
from .run import create_config,load_model,make_stepper
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.diagnostics import weak_moments
from benchmarks.research_sequential_next.material_study import material_metrics,compare_fields
from benchmarks.research_sequential_next.compare import metric
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_sequential_next.scenarios import install_peak
from engine.aniso_phase1.research_phase_stress_next.warm import install_warm


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
    modal=read(run/'S1/modal-observable-map.json')
    if modal['space']!=r.signature:raise ValueError('stale modal basis')
    j=max(modal['selected_modes'],key=lambda x:x['estimated_output_peak_Pa'])['mode']
    with np.load(run/'S1/modal-basis.npz') as z:vector=z['vectors'][:,j].copy()
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


def freeze_numeric(run):
    run=Path(run);verify(run)
    register(run,'S5/numeric-lock.json',dict(numerical_source_sha256=source_files(),space=read(run/'selected-space.json'),times=read(run/'S2/time-scope-decision.json')['times'],coupled_q5=False))
    register(run,'S5/main-qualification-protocol.json',dict(samples=[0.,.5,.8,1.0875,1.1],source='future cases/final-full only',qualification_pending=True,mass_order=7,material_orders=[5,7,8]))
    same=read(run/'selected-space.json')['package']['sha256']==read(run/'baseline-space.json')['package']['sha256']
    if same:
        shutil.copyfile(APP/'S1/modal-basis.npz',run/'S1/modal-basis.npz')
        write(run/'S1/modal-observable-map.json',read(APP/'S1/modal-observable-map.json'))
    else:
        write(run/'S5/window-runtime-final.json',dict(status='not_triggered',reason='new solid space cannot inherit historical sensitive initial coordinates',numeric_sources=source_files()))
        write(run/'S5/scope-boundaries.json',dict(status='not_triggered',reason='no new-space sensitive-window license issued; main scope tests after final certificate'))
    print('NUMERICS_FROZEN','baseline' if same else 'new space',flush=True)

def windows_prepare(run):
    run=Path(run)
    if read(run/'selected-space.json')['package']['sha256']!=read(run/'baseline-space.json')['package']['sha256']:raise ValueError('historical sensitive trajectory is BASELINE only')
    source=LOCAL/'cases/sensitive-prefix';lookup={round(x['state'].time,10):x for x in history(source)}
    grid=[round(.6+k*.0125,10) for k in range(9)]
    register(run,'S5/window-protocol.json',dict(peak_m=.0075,window=[.6,.7],steps=8,restart_after=4,samples=[.6,.65,.7],reserved_material_sample=.6375,
        full_source=str(source),source_identity_sha256=sha(source/'identity.json'),old_window_permissions_not_combined=True,new_material_qualification=True,new_sensitive_full_cycle=False))
    register(run,'S5/evidence-vs-permission.json',dict(parent_main_sha256=sha(APP/'S6/qualification-final.json'),old_continuous_window_sha256=sha(APP/'S5/continuous-window-qualification.json'),
        reuse='immutable sufficient trajectory states on unchanged formal space/physics',new_scope='continuous eight-step window only after same-state q7/q8/q5 qualification',numeric_source_sha256=source_files()))
    qualification=certify(run,source,grid,.0075,[.6,.65,.7,.6375],'S5/continuous-window-qualification.json')
    if not qualification['qualified']:raise ValueError('continuous window not qualified')
    cfg=create_config(run,'sensitive-continuous',start=.6,end=.7,times=grid,peak_m=.0075,initial=lookup[.6]['folder']/'state.json',rule_policy='q5_with_full_retry',qualification_path=run/'S5/continuous-window-qualification.json',field_cache=True,display_frames=3)
    if cfg['post_release']['rule_policy']!='q5_with_full_retry':raise ValueError('continuous window unexpectedly downgraded')
    from engine.aniso_phase1.research_basis_allocation_next.rules import permission
    cases=[]
    for name in ('beyond_end','wrong_amplitude','old_source','skip_node','missing_initial'):
        bad=copy.deepcopy(cfg);proof=copy.deepcopy(qualification)
        if name=='beyond_end':bad['times'].append(.7125)
        elif name=='wrong_amplitude':bad['scenario']['peak_m']=.008
        elif name=='old_source':proof['numerical_source_sha256']={}
        elif name=='skip_node':bad['times'].pop(2)
        else:bad['initial_state']=None
        allowed,reason=permission(proof,bad,numerical_sources=source_files())
        if allowed:raise ValueError('scope escaped: '+name)
        cases.append(dict(request=name,rejected=True,reason=reason))
    write(run/'S5/scope-boundaries.json',dict(status='passed_scoped',records=cases))
    print('CONTINUOUS_WINDOW_READY',flush=True)

def windows_check(run):
    run=Path(run);folder=run/'cases/sensitive-continuous';cfg=read(folder/'execution-protocol.json');m,_=load_model(run,cfg);c=make_stepper(run,cfg,m,m.rest());h=history(folder)
    ref={round(x['state'].time,10):x for x in history(LOCAL/'cases/sensitive-prefix')};checks=[]
    for item in h[1:]:
        comparison=ref[round(item['state'].time,10)];fields=compare_fields(c.full,item['state'],comparison['state'],cfg);reaction=metric(item['rows'][-1]['reaction_N'],comparison['rows'][-1]['reaction_N'],1e-4,.05)
        checks.append(dict(time_s=item['state'].time,regions=fields,reaction=reaction,passed=reaction['passed'] and all(x['passed'] for v in fields.values() for x in v.values())))
    if len(h)!=9 or not all(x['passed'] for x in checks) or len(read(folder/'segments.json'))!=2:raise ValueError('eight-step continuous runtime check failed')
    if not all(np.array_equal(getattr(h[0]['state'],k),getattr(ref[.6]['state'],k)) for k in ('q','velocity','predictor')):raise ValueError('branch changed initial physical fields')
    actual=h[4]['state'];higher=install_warm(install_peak(type(m)(m.reduction,order=8,device='cuda:0'),.0075))
    d=np.random.default_rng(20261001).normal(size=actual.q.shape);d[m.fixed]=0;d/=np.linalg.norm(d)
    models=[m,c.full,higher];values=[x.evaluate(actual.q,d) for x in models];weak=[weak_moments(x,actual.q) for x in models]
    junction=dict(time_s=actual.time,actual_q5_checkpoint_sha256=sha(h[4]['folder']/'state.json'),compressed=material_metrics(values[0],values[1],weak[0],weak[1]),sufficient=material_metrics(values[1],values[2],weak[1],weak[2]),restarted_from_same_committed_generation=True)
    if abs(actual.time-.65)>1e-12 or not junction['compressed']['passed'] or not junction['sufficient']['passed']:raise ValueError('actual q5 junction not certified')
    write(run/'S5/junction-check.json',dict(status='passed_scoped',**junction))
    write(run/'S5/window-runtime-final.json',dict(status='passed_scoped',steps=8,window=[.6,.7],actual_new_process_restart=True,restart_time_s=.65,checks=checks,numeric_sources=source_files(),sensitive_full_cycle_q5=False,initial_unchanged=True))
    print('CONTINUOUS_WINDOW_PASSED',flush=True)

def qualification(run):
    run=Path(run);source=run/'cases/final-full';cfg=read(source/'execution-protocol.json');register(run,'S6/qualification-protocol.json',dict(source=str(source),samples=[0.,.5,1.1,1.0875],independent_sample=.8,actual_grid=cfg['times'],mass_order=cfg['mass_order']))
    q=certify(run,source,cfg['times'],.005,[0.,.5,1.1,.8,1.0875],'S6/qualification-final.json',rest=True)
    write(run/'S6/material-decision.json',dict(status='qualified_main' if q['qualified'] else 'full_only',sufficient_qualified=q['sufficient_qualified'],compressed_qualified=q['qualified'],sensitive_full_cycle=False,coupled_q5=False))
    print('MAIN_QUALIFIED',q['qualified'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','window-prepare','window-check','qualification']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):{'freeze':freeze_numeric,'window-prepare':windows_prepare,'window-check':windows_check,'qualification':qualification}[a.phase](a.run)
