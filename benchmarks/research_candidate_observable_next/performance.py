"""Profile real coupled steps and compare one bounded static-metadata candidate."""
from pathlib import Path
import argparse,time,cProfile,pstats,resource,subprocess,os
import numpy as np
from .provenance import *
from .physics import baseline_model
from .coupling import setup as setup_new
from benchmarks.research_pressure_window_next.coupling_study import frame
from benchmarks.research_phase_stress_next.time_study import history
from engine.aniso_phase1.research_pressure_window_next.coupled import WindowCoupling
from engine.aniso_phase1.research_candidate_observable_next.rt0_cache import install,StaticRT0Metadata
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish
from benchmarks.research_sequential_next.checkpoint import GenerationStore


def prepare(run):
    run=Path(run);verify(run);new=run/'cases/observable-coarse-h';physical=run/'S3/coarse-physical-check.json';eligible=(new/'summary.json').exists() and read(new/'summary.json')['status']=='passed_scoped' and physical.exists() and read(physical)['status']=='passed_scoped' and read(physical)['time_refinement_passed']
    case=new if eligible else APP/'cases/coupled-coarse';indices=[1,3] if eligible else [2,18];h=history(case)
    register(run,'S5/profile-protocol.json',dict(status='registered',source_case=str(case),source_case_identity_sha256=sha(case/'identity.json'),observable_fixture=eligible,indices=indices,with_frames=[False,True],steps=2,states=[dict(path=str(h[i]['folder']/'state.json'),sha256=sha(h[i]['folder']/'state.json')) for i in indices],scope='new observable fixture' if eligible else 'inherited stable early pressure fixture only; not observable coupling qualification',candidate_limit=1,paired_steps=4,hard_extra_steps=2,cache_max_bytes=256*2**20,cache_fraction_of_initial_gpu_free=.02))


def setup(run,which):
    p=read(Path(run)/'S5/profile-protocol.json');item=history(Path(p['source_case']))[p['indices'][which]]
    if sha(item['folder']/'state.json')!=p['states'][which]['sha256']:raise ValueError('profile origin changed')
    if p['observable_fixture']:c,m,cfg=setup_new(run,'coarse',False,item['state'])
    else:
        protocol=read(APP/'S2/coupled-protocol.json');m,cfg=baseline_model(run,pressure=True);c=WindowCoupling(m,cfg,protocol['times_s'],cuts=protocol['cuts']['coarse'],state=item['state'])
    return p,item,c,m,cfg


def one(run,which,kind):
    import warp as wp
    run=Path(run);verify(run);begin=time.perf_counter();p,item,c,m,cfg=setup(run,which);wp.synchronize_device('cuda:0');build=time.perf_counter()-begin;cache=CachedProbes(m);meta=None
    observations=read((run/'S3/new-scene-protocol.json') if p['observable_fixture'] else (APP/'S2/coupled-protocol.json'))['observations_s']
    if (c.times[c.state.step+1] in observations)!=p['with_frames'][which]:raise ValueError('profile frame cadence differs from actual source scene')
    cap=min(256*2**20,int(.02*m.operator.memory_budget.initial_free))
    if kind=='B':meta=install(c.geometry,max_bytes=cap)
    name=f'{kind}-{which}';out=run/'S5'/name
    identity=dict(coupling=c.identity,initial_state_sha256=sha(item['folder']/'state.json'),driver_sha256=sha(__file__),numeric_sources=source_files(),fixture_source_sha256=sha(ROOT/'engine/aniso_phase1/research_candidate_observable_next/fixture.py') if p['observable_fixture'] else None,metadata_source_sha256=sha(ROOT/'engine/aniso_phase1/research_candidate_observable_next/rt0_cache.py') if kind=='B' else None)
    write(out/'identity.json',identity);store=GenerationStore(out,identity);store.save(c.state,[]);parts={'field_s':0.,'commit_s':0.,'geometry_inclusive_s':0.,'rt0_assembly_s':0.,'material_s':0.};trace=cProfile.Profile();save=store.save
    def publish(*a,**kw):
        t=time.perf_counter()
        try:return save(*a,**kw)
        finally:parts['commit_s']+=time.perf_counter()-t
    store.save=publish
    def emit(s):
        t=time.perf_counter()
        try:return frame(c,cache,s)
        finally:parts['field_s']+=time.perf_counter()-t
    def timed(obj,method,key):
        original=getattr(obj,method)
        def measured(*a,**kw):
            t=time.perf_counter()
            try:return original(*a,**kw)
            finally:parts[key]+=time.perf_counter()-t
        setattr(obj,method,measured)
    if kind=='profile':
        timed(c.geometry,'evaluate','geometry_inclusive_s');timed(c.geometry.topology,'assemble','rt0_assembly_s');timed(m,'evaluate','material_s');trace.enable()
    def sharing():
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        if result.returncode:return dict(verified=False,output=result.stderr)
        pids=[int(line.split(',')[0]) for line in result.stdout.splitlines() if line.strip()]
        return dict(verified=True,exclusive=all(pid==os.getpid() for pid in pids),pids=pids,output=result.stdout)
    sharing_before=sharing()
    write(out/'attempt.json',dict(attempts=1,source_step=item['state'].step,accepted=False))
    wp.synchronize_device('cuda:0');tick=time.perf_counter()
    try:row=advance_publish(c,store,[],frame_builder=emit if p['with_frames'][which] else None)
    except Exception as error:
        write(out/'failure.json',dict(status='limited',error=repr(error),last_digest=c.state.digest(),rolled_back=store.load()['state'].digest()==c.state.digest()));raise
    wp.synchronize_device('cuda:0');elapsed=time.perf_counter()-tick
    write(out/'attempt.json',dict(attempts=1,source_step=item['state'].step,accepted=True))
    calls=[];solver_profile_s=transfer_profile_self_s=0.
    if kind=='profile':
        trace.disable();stats=pstats.Stats(trace);calls=[dict(file=f,line=l,function=n,calls=nc,self_s=tt,cumulative_s=ct) for (f,l,n),(cc,nc,tt,ct,_) in sorted(stats.stats.items(),key=lambda x:x[1][2],reverse=True)[:60]]
        solver_profile_s=sum(v[3] for (f,l,n),v in stats.stats.items() if n=='balanced_solve')
        transfer_profile_self_s=sum(v[2] for (f,l,n),v in stats.stats.items() if '/warp/' in f and n in ('copy','numpy'))
    write(out/'measurement.json',dict(status='passed_scoped',kind=kind,initial_index=p['indices'][which],origin_sha256=sha(item['folder']/'state.json'),model_build_s=build,advance_s=elapsed,solver_and_geometry_s=elapsed-parts['field_s']-parts['commit_s'],**parts,linear_solver_profile_inclusive_s=solver_profile_s,warp_transfer_profile_self_s=transfer_profile_self_s,nested_measurements_not_additive=True,profile=calls,frame_written=p['with_frames'][which],iterations=row['iterations'],true_residual_fraction=row['true_scaled_residual'],peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,metadata_bytes=meta.bytes if meta else 0,metadata_setup_s=meta.build_seconds if meta else 0,cache_cap_bytes=cap,profile_is_not_speed_ratio=kind=='profile',end_digest=c.state.digest(),sharing_before=sharing_before,sharing_after=sharing()))
    print('PERFORMANCE_STEP',name,elapsed,'metadata',meta.bytes if meta else 0,flush=True)


def operator(run):
    run=Path(run);verify(run);p,item,c,m,cfg=setup(run,0);g=c.geometry;cap=min(256*2**20,int(.02*m.operator.memory_budget.initial_free));started=time.perf_counter()
    profiles=[read(run/f'S5/profile-{i}/measurement.json') for i in (0,1)]
    write(run/'S5/profile.json',dict(status='passed_scoped',records=profiles,actual_steps=2,model_build_separate=True,one_output_step=True,nested_geometry_timings_not_additive=True))
    try:meta=StaticRT0Metadata(g.topology,g.X,g.total_weights,g.cell_ids,g.mobility,max_bytes=cap,owner_identity=g.identity)
    except MemoryError as error:
        estimate=len(g.total_weights)*32+72
        write(run/'S5/operator-equivalence.json',dict(status='not_triggered',reason=str(error),required_bytes=estimate,byte_cap=cap,production_operator_unchanged=True))
        register(run,'S5/performance-candidate.json',dict(status='not_triggered',candidate='owned RT0 indices and reference basis factors',eligible_paired=False,reason='complete chosen metadata representation exceeds frozen memory cap',required_bytes=estimate,byte_cap=cap,no_second_candidate=True,physics_changed=False));return
    build=time.perf_counter()-started
    cases=[('rest',np.broadcast_to(np.eye(3),(len(g.X),3,3))),('manufactured',np.broadcast_to(np.array([[1.05,.12,0],[.02,.98,.07],[0,.03,1.02]]),(len(g.X),3,3)))]
    h=history(Path(p['source_case']));cases.extend((f'saved-{i}',g.field(h[i]['state'].q).numpy()) for i in p['indices']);records=[]
    for label,F in cases:
        t=time.perf_counter();a,j=g.topology.assemble(g.X,g.total_weights,g.cell_ids,F,g.mobility);ta=time.perf_counter()-t;t=time.perf_counter();b,k=meta.assemble(g.X,g.total_weights,g.cell_ids,F,g.mobility);tb=time.perf_counter()-t;err=float(np.max(abs(a-b)));passed=np.allclose(a,b,atol=1e-8,rtol=2e-5) and j==k
        records.append(dict(state=label,H_max_error=err,minJ=j,passed=bool(passed),uncached_s=ta,cached_s=tb,assembly_gain=1-tb/ta))
    q0=h[p['indices'][0]]['state'].q;q1=h[p['indices'][1]]['state'].q
    old_values=[g.evaluate(q) for q in (q0,q1)];old_work=g.discrete(q0,q1)
    g.topology.assemble=meta.assemble;g.cache.clear()
    new_values=[g.evaluate(q) for q in (q0,q1)];new_work=g.discrete(q0,q1)
    geometry_errors={k:max(float(np.max(abs(a[k]-b[k]))) for a,b in zip(old_values,new_values)) for k in ('H','volume','gradient')}
    geometry_errors['discrete_gradient']=float(np.max(abs(old_work-new_work)))
    geometry_errors['pressure_work']=float(np.max(abs(np.einsum('cij,ij->c',old_work-new_work,q1-q0))))
    geometry_ok=all(v<1e-8 for k,v in geometry_errors.items() if k!='H') and all(np.allclose(a['H'],b['H'],atol=1e-8,rtol=2e-5) for a,b in zip(old_values,new_values))
    write(run/'S5/operator-equivalence.json',dict(status='passed_scoped' if geometry_ok and all(x['passed'] for x in records) else 'limited',records=records,geometry_and_work_errors=geometry_errors,owned_static_bytes=meta.bytes,byte_cap=cap,setup_s=build,current_H_always_rebuilt=True))
    profiles=[read(run/f'S5/profile-{i}/measurement.json') for i in (0,1)];ratio=np.median([x['rt0_assembly_s']/x['advance_s'] for x in profiles]);gain=np.median([x['assembly_gain'] for x in records]);eligible=geometry_ok and all(x['passed'] for x in records) and ratio*gain>=.05
    register(run,'S5/compact-performance-candidate.json',dict(status='registered' if eligible else 'not_triggered',candidate='same static metadata candidate, three axis minus-face basis representation',expected_step_gain=float(ratio*gain),eligible_paired=bool(eligible),static_only=True,estimated_fraction_is_not_end_to_end_measurement=True,byte_cap=cap,bytes=meta.bytes,physics_changed=False))
    write(run/'S5/profile.json',dict(status='passed_scoped',records=profiles,actual_steps=2,model_build_separate=True,one_output_step=True,nested_geometry_timings_not_additive=True))
    print('METADATA_GATE',eligible,ratio,gain,meta.bytes,flush=True)


def finish(run):
    run=Path(run);candidate=read(run/'S5/compact-performance-candidate.json') if (run/'S5/compact-performance-candidate.json').exists() else read(run/'S5/performance-candidate.json');numeric=source_files();records=[]
    if candidate['eligible_paired']:
        for i in (0,1):
            a=read(run/f'S5/A-{i}/measurement.json');b=read(run/f'S5/B-{i}/measurement.json');sa=history(run/f'S5/A-{i}')[-1]['state'];sb=history(run/f'S5/B-{i}')[-1]['state'];errors={k:float(np.max(abs(getattr(sa,k)-getattr(sb,k)))) for k in ('q','velocity','predictor')};fa=sa.child_states['fluid'];fb=sb.child_states['fluid']
            for k in ('pressure_Pa','flux_interval_m3_s','content_m3','cumulative_source_m3','cumulative_boundary_m3'):errors[k]=float(np.max(abs(np.asarray(fa[k])-fb[k])))
            passed=max(errors.values())<1e-8;records.append(dict(state=i,gain=1-b['advance_s']/a['advance_s'],A=a,B=b,physical_errors=errors,passed=passed))
        gains=[x['gain'] for x in records];median=float(np.median(gains));variation=float(np.ptp(gains));exclusive=all(x[k][phase].get('exclusive',False) for x in records for k in ('A','B') for phase in ('sharing_before','sharing_after'));selected=exclusive and all(x['passed'] for x in records) and min(gains)>=0 and median>=.05 and median>variation
        write(run/'S5/paired-performance.json',dict(status='passed_scoped' if selected else 'limited',records=records,median_gain=median,between_state_gain_range=variation,repeated_noise_estimate=False,order=['A0','B0','B1','A1'],new_steps=4,setup_amortization=[r['B']['metadata_setup_s']/max(r['A']['advance_s']-r['B']['advance_s'],1e-30) for r in records]))
    else:
        selected=False;write(run/'S5/paired-performance.json',dict(status='not_triggered',reason='predicted end-to-end gain under five percent or operator gate failed',new_steps=0))
    write(run/'S5/performance-decision.json',dict(status='research_cache_qualified' if selected else 'retain_original',selected=selected,formal_solid_changed=False,preoptimization_S3_not_relabelled=True,production_default_changed=False,scope=read(run/'S5/profile-protocol.json')['scope'],one_candidate=True,conditional_opt_in_only=selected))
    write(run/'S5/final-numeric-lock.json',dict(status='passed_scoped',solid_numerical_source_sha256=numeric,fixture_source_sha256=sha(ROOT/'engine/aniso_phase1/research_candidate_observable_next/fixture.py'),cache_source_sha256=sha(ROOT/'engine/aniso_phase1/research_candidate_observable_next/rt0_cache.py'),default='inherited parent solid only'))
    print('PERFORMANCE_DECISION',selected,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','profile','A','B','operator','finish']);p.add_argument('--run',type=Path,required=True);p.add_argument('--which',type=int,choices=[0,1],default=0);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase in ('profile','A','B'):one(a.run,a.which,a.phase)
        else:globals()[a.phase](a.run)
