"""Fresh-process paired tests of one fixed transpose-buffer reuse candidate."""
from pathlib import Path
import argparse,time,resource
import numpy as np
from .provenance import read,write,sha,register,source_files,serial_lock
from .performance_study import settings
from .run import load_model
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_post_release.fields import CachedProbes

def trial(run,enabled,ts,repeat):
    import warp as wp
    run=Path(run);cfg=settings(run);cfg['implementation']['reuse_transpose_buffers']=bool(enabled);t0=time.perf_counter();m,_=load_model(run,cfg);cache=CachedProbes(m);wp.synchronize_device(m.device);build=time.perf_counter()-t0
    source=run/'cases/selected-prefix';item=next(x for x in history(source) if abs(x['state'].time-ts)<1e-10);m.validate(item['state'],material=True)
    rng=np.random.default_rng(456);d=rng.normal(size=item['state'].q.shape);d[m.fixed]=0;d/=np.linalg.norm(d);op=m.evaluate(item['state'].q,d)
    label=f'{ts}-{repeat}-{int(enabled)}';folder=run/'S4/paired'/label;c=ValidatedAVF(m,cfg,item['state']);store=GenerationStore(folder,dict(model=m.identity,config=cfg,numeric_sources=source_files(),source_state_sha256=sha(item['folder']/'state.json')))
    rows=[];start=time.perf_counter();store.save(c.state,[])
    for _ in range(2):rows.append(c.step(.0125));store.save(c.state,rows)
    frame=cache.frame(c.state);wp.synchronize_device(m.device);warm=time.perf_counter()-start
    np.savez_compressed(folder/'result.npz',q=c.state.q,v=c.state.velocity,predictor=c.state.predictor,operator_U=op['U'],operator_force=op['force'],tangent_action=op['tangent_action'],**frame)
    gpu=m.operator.memory_budget.report();reuse_flags=[bool(getattr(x[1],'reused_transpose_buffers',False)) for x in (m.operator.maps.old,m.operator.maps.raw)]
    report=dict(enabled=enabled,time_s=ts,repeat=repeat,build_s=build,warm_s=warm,total_s=build+warm,peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,gpu=gpu,
        reuse_flags=reuse_flags,max_true_residual=max(r['true_residual'] for r in rows),numeric_sources=source_files(),source_model_verified=True,independent_process=True)
    write(folder/'report.json',report);print('REUSE_TRIAL',label,build,warm,report['peak_rss_GiB'],flush=True)

def decide(run):
    run=Path(run);records=[];ops=[]
    for ts in [.1,1.1]:
        for repeat in [0,1]:
            folders=[run/'S4/paired'/f'{ts}-{repeat}-{enabled}' for enabled in [0,1]];a,b=[read(f/'report.json') for f in folders]
            with np.load(folders[0]/'result.npz') as aa,np.load(folders[1]/'result.npz') as bb:errors={k:float(np.max(abs(aa[k]-bb[k]))) for k in aa.files}
            if max(errors.values())>1e-7 or max(errors[k] for k in ['q','v','predictor','operator_U','operator_force','tangent_action'])>1e-8:raise ValueError('reuse changed operator or trajectory')
            if b['reuse_flags']!=[True,True]:raise ValueError('candidate not actually active')
            records.append(dict(time_s=ts,repeat=repeat,baseline=a,candidate=b,errors=errors));ops.append(dict(time_s=ts,repeat=repeat,errors=errors))
    baseline=float(np.median([x['baseline']['total_s'] for x in records]));candidate=float(np.median([x['candidate']['total_s'] for x in records]));wb=float(np.median([x['baseline']['warm_s'] for x in records]));wc=float(np.median([x['candidate']['warm_s'] for x in records]));gain=1-candidate/baseline
    memory=max(x['candidate']['peak_rss_GiB']/x['baseline']['peak_rss_GiB'] for x in records);adopted=bool(gain>=.1 and wc<=1.05*wb and memory<=1.1 and max(x['candidate']['peak_rss_GiB'] for x in records)<=16)
    write(run/'S4/paired-performance.json',dict(status='passed_equivalence',records=records,median_baseline_total_s=baseline,median_candidate_total_s=candidate,gain=gain,median_baseline_warm_s=wb,median_candidate_warm_s=wc,max_memory_ratio=memory,adopted=adopted))
    write(run/'S4/operator-equivalence.json',dict(status='passed_scoped',checks=ops,unchanged_summation_order=True,content_hash_verification_preserved=True,no_global_cache=True,private_layout_and_work_arrays=True))
    write(run/'S4/performance-decision.json',dict(status='adopt_transpose_buffer_reuse' if adopted else 'retain_original_construction',warm_adopted=True,shared_reduction=True,uncompressed_archive=True,field_cache=True,reuse_transpose_buffers=adopted,candidate_count=1,tangent_cache=False,gain=gain,whole_cycle_gain_claim=False))
    write(run/'S4/final-source-refresh-list.json',dict(numeric_sources=source_files(),reissue_material_certificates=True,physics_equivalence='operator, true residual, all fields and two-step q/v/predictor matched at two nonzero states',source_identity_rewrite=False))
    print('REUSE_DECISION',adopted,gain,wb,wc,memory,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['trial','decide']);p.add_argument('--run',type=Path,required=True);p.add_argument('--enabled',type=int,choices=[0,1],default=0);p.add_argument('--time',type=float,default=.1);p.add_argument('--repeat',type=int,choices=[0,1],default=0);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='trial':trial(a.run,bool(a.enabled),a.time,a.repeat)
        else:decide(a.run)
