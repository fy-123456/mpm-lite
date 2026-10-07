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
    if (run/'S5/performance-decision.json').exists() and read(run/'S5/performance-decision.json').get('reuse_transpose_buffers',False):
        from engine.aniso_phase1.research_local_span_next.reuse import ReusedSegmentedModel as Model
    if (run/'S5/performance-decision.json').exists() and read(run/'S5/performance-decision.json').get('vectorized_segment_metadata',False):
        from engine.aniso_phase1.research_phase_boundary_next.segments import VectorizedModel as Model
    models={o:install_warm(install_peak(Model(r,order=o,device='cuda:0'),peak)) for o in (5,order,order+1)}
    if not all(np.array_equal(models[order].M,m.M) for m in models.values()):raise ValueError('material compression changed mass')
    h=history(source);lookup={round(x['state'].time,10):x for x in h};m=models[order]
    if read(source/'identity.json')['model']!=m.identity:raise ValueError('sufficient trajectory model differs')
    modal=read(run/'S1/modal-observable-map.json')
    if modal['space']!=r.signature:raise ValueError('stale modal basis')
    j=int(modal['selected_modes'][-1])
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


def qualification(run):
    run=Path(run);source=run/'cases/final-full';cfg=read(source/'execution-protocol.json');register(run,'S6/qualification-protocol.json',dict(source=str(source),samples=[0.,.5,1.1,1.0875],independent_sample=.8,actual_grid=cfg['times'],mass_order=cfg['mass_order']))
    q=certify(run,source,cfg['times'],.005,[0.,.5,1.1,.8,1.0875],'S6/qualification-final.json',rest=True)
    write(run/'S6/material-decision.json',dict(status='qualified_main' if q['qualified'] else 'full_only',sufficient_qualified=q['sufficient_qualified'],compressed_qualified=q['qualified'],sensitive_full_cycle=False,coupled_q5=False))
    print('MAIN_QUALIFIED',q['qualified'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):qualification(a.run)
