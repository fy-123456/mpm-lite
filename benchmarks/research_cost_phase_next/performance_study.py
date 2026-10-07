"""Bounded construction profiling and same-state shared-reduction comparison."""
from pathlib import Path
import argparse,time,cProfile,pstats,copy,resource
import numpy as np
import warp as wp
from .provenance import APP,read,write,sha,register,source_files,verify,serial_lock
from . import config
from .run import load_model,make_stepper
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_reference_next.performance_study import branch
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_post_release.runtime_rules import RuntimeRetry,RetryableRuleError
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF

def settings(run,shared):
    run=Path(run);choice=read(run/'selected-space.json');cert=run/'P5/qualification.json'
    return config.make(read(run/'input-lock.json')['energy_scale_J'],rule_policy='q5_with_full_retry',
        qualification=dict(path=str(cert.resolve()),sha256=sha(cert)),field_cache=True,
        space=choice['package'],mass_order=choice['mass_order'],full_order=choice['full_order'],shared_reduction=shared)

def construct(run,cfg):
    prof=cProfile.Profile();prof.enable();t=time.perf_counter()
    m,_=load_model(run,cfg);first=time.perf_counter()-t
    t=time.perf_counter();control=make_stepper(run,cfg,m,m.rest());second=time.perf_counter()-t
    t=time.perf_counter();cache=CachedProbes(m,tuple(cfg['probe_shape']));wp.synchronize_device(m.device);fields=time.perf_counter()-t
    prof.disable()
    stats=pstats.Stats(prof);rows=[]
    for (file,line,name),(cc,nc,tt,ct,_) in sorted(stats.stats.items(),key=lambda kv:kv[1][3],reverse=True)[:40]:
        rows.append(dict(file=file,line=line,function=name,calls=nc,self_s=tt,cumulative_s=ct))
    return m,control.full,cache,dict(q5_build_s=first,q7_build_s=second,field_cache_s=fields,
        total_s=first+second+fields,nested_profile=rows,profile_timings_not_additive=True,
        rss_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)

def trial(run,label,cfg,m,full,cache,item):
    state,origin=branch(item,m,full);c=RuntimeRetry(m,full,cfg,state)
    folder=Path(run)/'P1/profiles'/label
    store=GenerationStore(folder,dict(schema='cost-phase-profile-v1',source_sha256=source_files(),initial=state.digest(),protocol=config.identity(cfg)))
    t=time.perf_counter();store.save(state,[]);io=time.perf_counter()-t;advance=0.;rows=[]
    for _ in range(2):
        t=time.perf_counter();rows.append(c.step(.0125));wp.synchronize_device(m.device);advance+=time.perf_counter()-t
        t=time.perf_counter();store.save(c.state,rows);io+=time.perf_counter()-t
    t=time.perf_counter();frame=cache.frame(c.state);wp.synchronize_device(m.device);field=time.perf_counter()-t
    return dict(warm_s=advance+io+field,advance_s=advance,io_s=io,field_s=field,origin=origin,
        q=c.state.q,v=c.state.velocity,predictor=c.state.predictor,frame=frame)

def study(run):
    run=Path(run);verify(run)
    register(run,'P1/protocol.json',dict(baseline='current authenticated array cache enabled',times=[.1,1.1],steps=2,paired_repeats=2,
        candidate='one loaded immutable reduction for q5/q7; private material operators and trials',
        cold_scope='new model objects in one process, existing Warp kernel and OS caches; compiler excluded',
        minimum_launch_gain=.1,warm_regression_limit=.05))
    folder=APP/'cases/final-q7-dt0125';history=GenerationStore(folder,read(folder/'identity.json')).history()
    states={round(x['state'].time,10):x for x in history};models={}
    for enabled in (False,True):
        cfg=settings(run,enabled);models[enabled]=(cfg,*construct(run,cfg))
        print('CONSTRUCT',enabled,models[enabled][-1]['total_s'],flush=True)
    cfg,m,f,cache,baseline=models[False];fcfg,fm,ff,fcache,optimized=models[True]
    shared=fm.reduction is ff.reduction and fm.M is ff.M
    private=fm.operator is not ff.operator and fm.operator.maps.layouts is not ff.operator.maps.layouts
    assert shared and private
    immutable=all(not a.flags.writeable for a in (fm.M,fm.reduction.original_mass,fm.reduction.original_stiffness,fm.reduction.P,fm.parent.raw.data))
    assert immutable
    rejected=False
    try:fm._shared_space.acquire(fcfg['physical_space'],'cuda:99')
    except ValueError:rejected=True
    assert rejected
    checks=[]
    rng=np.random.default_rng(20261001)
    for t in (.1,1.1):
        q=states[t]['state'].q;d=rng.normal(size=q.shape);d[m.fixed]=0;d/=np.linalg.norm(d)
        for direction in (d,np.eye(1,q.size,3*int(m.free[0])).reshape(q.shape)):
            a,b=m.evaluate(q,direction),fm.evaluate(q,direction)
            errors={k:float(np.max(abs(np.asarray(a[k])-np.asarray(b[k])))) for k in ('U','force','tangent_action')}
            if max(errors.values())>1e-8:raise ValueError('shared reduction changed operator')
            checks.append(dict(time=t,errors=errors))
    records=[]
    for t in (.1,1.1):
        for rep in range(2):
            pair={}
            for enabled in ([False,True] if rep==0 else [True,False]):
                cfg,m,f,cache,_=models[enabled]
                pair[enabled]=trial(run,f'{t}-{rep}-{int(enabled)}',cfg,m,f,cache,states[t])
            a,b=pair[False],pair[True]
            err=max(float(np.max(abs(a[k]-b[k]))) for k in ('q','v','predictor'))
            ferr=max(float(np.max(abs(a['frame'][k]-b['frame'][k]))) for k in a['frame'])
            assert err<1e-8 and ferr<1e-7
            records.append(dict(time=t,repeat=rep,state_error=err,field_error=ferr,
                baseline={k:v for k,v in a.items() if k not in ('q','v','predictor','frame')},
                shared={k:v for k,v in b.items() if k not in ('q','v','predictor','frame')}))
    c=RuntimeRetry(fm,ff,fcfg)
    def fail(attempt,where,state):
        if attempt==0 and where=='after_prepare':raise RetryableRuleError('shared rollback qualification')
    row=c.step(.0125,inject=fail);ref=ValidatedAVF(ff,fcfg);ref.step(.0125)
    error=max(float(np.max(abs(getattr(c.state,k)-getattr(ref.state,k)))) for k in ('q','velocity','predictor'))
    assert row['material_attempts']==2 and error<1e-8
    oldwarm=float(np.median([x['baseline']['warm_s'] for x in records]));newwarm=float(np.median([x['shared']['warm_s'] for x in records]))
    gain=1-(optimized['total_s']+newwarm)/(baseline['total_s']+oldwarm)
    # Warm numerical path is identical; observed variation is reported, never hidden.
    warm_ok=newwarm<=1.05*oldwarm
    adopted=gain>=.1 and warm_ok
    write(run/'P1/baseline-profile.json',baseline)
    write(run/'P1/shared-profile.json',optimized)
    write(run/'P1/shared-resource-check.json',dict(status='passed_scoped',shared_reduction=shared,private_operators=private,
        readonly_arrays=immutable,foreign_device_rejected=rejected,no_global_cache=True,one_validation_per_construction_tree=True))
    write(run/'P1/operator-equivalence.json',dict(status='passed_scoped',records=checks,matrices_equal=bool(np.array_equal(fm.M,models[False][1].M))))
    write(run/'P1/transaction-check.json',dict(status='passed_scoped',full_retry_state_error=error,actual_attempts=row['material_attempts']))
    write(run/'P1/performance-decision.json',dict(status='passed_scoped',adopted=adopted,construction_baseline_s=baseline['total_s'],
        construction_shared_s=optimized['total_s'],warm_baseline_s=oldwarm,warm_shared_s=newwarm,independent_launch_gain=gain,warm_gate=warm_ok,
        records=records,only_change='immutable reduction ownership',no_whole_cycle_gain_claim=True))
    print('PERFORMANCE',adopted,gain,oldwarm,newwarm,flush=True)

def archive_study(run):
    import shutil
    import scipy.sparse as sp
    run=Path(run);verify(run)
    register(run,'P1/archive-protocol.json',dict(candidate_number=2,change='uncompressed numeric raw.npz only',
        reason='shared profile retains 3.34 s zlib decompression; preserve exact raw CSR values and all coordinates',
        times=[.1,1.1],steps=2,pairs=2,minimum_launch_gain=.1))
    choice=read(run/'selected-space.json');entry=choice['package']['cache'];original=Path(entry['path'])
    cfg=settings(run,True);m,f,cache,oldbuild=construct(run,cfg)
    target=run/'P1/uncompressed-space-cache';target.mkdir(exist_ok=False)
    begin=time.perf_counter()
    metadata=read(original)
    for name in metadata['files']:
        if name!='raw.npz':shutil.copyfile(original.parent/name,target/name)
    sp.save_npz(target/'raw.npz',m.parent.raw,compressed=False)
    metadata['files']={name:sha(target/name) for name in metadata['files']}
    metadata['construction']='same exact arrays; raw CSR ZIP members stored without compression; other numeric arrays unchanged'
    metadata['source_cache_sha256']=entry['sha256'];write(target/'cache.json',metadata)
    newentry=dict(path=str((target/'cache.json').resolve()),sha256=sha(target/'cache.json'))
    fcfg=copy.deepcopy(cfg);fcfg['physical_space']['cache']=newentry
    fm,ff,fcache,newbuild=construct(run,fcfg);creation=time.perf_counter()-begin-newbuild['total_s']
    equal={k:bool(np.array_equal(getattr(m.reduction,k),getattr(fm.reduction,k))) for k in ('M','K','P','offset','original_mass','original_stiffness')}
    assert all(equal.values()) and m.reduction.signature==fm.reduction.signature
    for k in ('data','indices','indptr'):assert np.array_equal(getattr(m.parent.raw,k),getattr(fm.parent.raw,k))
    folder=APP/'cases/final-q7-dt0125';states={round(x['state'].time,10):x for x in GenerationStore(folder,read(folder/'identity.json')).history()}
    checks=[];records=[]
    for t in (.1,1.1):
        d=np.zeros_like(states[t]['state'].q);d[m.free[0],0]=1.
        a,b=m.evaluate(states[t]['state'].q,d),fm.evaluate(states[t]['state'].q,d)
        errors={k:float(np.max(abs(np.asarray(a[k])-np.asarray(b[k])))) for k in ('U','force','tangent_action')}
        assert max(errors.values())<1e-8;checks.append(dict(time=t,errors=errors))
        for repeat in range(2):
            pair={}
            for enabled in ([False,True] if repeat==0 else [True,False]):
                pair[enabled]=trial(run,f'archive-{t}-{repeat}-{int(enabled)}',fcfg if enabled else cfg,fm if enabled else m,ff if enabled else f,fcache if enabled else cache,states[t])
            a,b=pair[False],pair[True]
            error=max(float(np.max(abs(a[k]-b[k]))) for k in ('q','v','predictor'))
            field=max(float(np.max(abs(a['frame'][k]-b['frame'][k]))) for k in a['frame'])
            assert error<1e-8 and field<1e-7
            records.append(dict(time=t,repeat=repeat,state_error=error,field_error=field,baseline_s=a['warm_s'],candidate_s=b['warm_s']))
    oldwarm=float(np.median([x['baseline_s'] for x in records]));newwarm=float(np.median([x['candidate_s'] for x in records]))
    gain=1-(newbuild['total_s']+newwarm)/(oldbuild['total_s']+oldwarm)
    adopted=gain>=.1 and newwarm<=oldwarm*1.05
    report=dict(status='passed_scoped',adopted=adopted,old_build=oldbuild,new_build=newbuild,old_warm_s=oldwarm,new_warm_s=newwarm,
        launch_gain=gain,write_seconds=creation,bytes=sum(x.stat().st_size for x in target.iterdir()),matrices_equal=equal,operator_checks=checks,records=records,
        cache_entry=newentry,amortization_launches=creation/max(oldbuild['total_s']-newbuild['total_s'],1e-12))
    write(run/'P1/compact-representation.json',report)
    previous=read(run/'P1/performance-decision.json');write(run/'P1/shared-decision.json',previous)
    previous['uncompressed_archive']=adopted;previous['archive_evidence']='P1/compact-representation.json'
    previous['memory_note']='paired process retains both variants; cumulative high-water RSS is not isolated variant memory; final independent cases verify budget'
    if adopted:
        choice['package']['cache']=newentry;choice['performance']='shared immutable reduction plus uncompressed exact raw CSR archive';write(run/'selected-space.json',choice)
    write(run/'P1/performance-decision.json',previous)
    print('ARCHIVE',adopted,oldbuild['total_s'],newbuild['total_s'],gain,oldwarm,newwarm,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--phase',choices=['shared','archive'],default='shared');a=p.parse_args()
    with serial_lock(a.run):(study if a.phase=='shared' else archive_study)(a.run)
