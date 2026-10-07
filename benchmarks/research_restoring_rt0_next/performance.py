"""Same-state device RT0 qualification and four real serial paired steps."""
from pathlib import Path
import argparse,time,os,subprocess,resource
import numpy as np
from .provenance import *
from .physics import baseline_model
from engine.aniso_phase1.research_candidate_observable_next.fixture import ObservableCoupling
from engine.aniso_phase1.research_restoring_rt0_next.fixture import construct,physical_payload
from engine.aniso_phase1.research_restoring_rt0_next.device_rt0 import DeviceGeometry
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish
from benchmarks.research_pressure_window_next.coupling_study import frame
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric

def numerical_sources():
    return dict(source_files(),**{str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'engine/aniso_phase1/research_restoring_rt0_next').glob('*.py')})

def sharing():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
    pids=[int(s.split(',')[0]) for s in result.stdout.splitlines() if s.strip()] if result.returncode==0 else []
    return dict(exclusive=result.returncode==0 and all(p==os.getpid() for p in pids),pids=pids,raw=result.stdout)

def origin(index):
    case=APP/'cases/observable-coarse-h';h=history(case);return h[(1,3)[index]]

def operators(run):
    import warp as wp
    run=Path(run);verify(run)
    if not read(run/'S2/optimization-protocol.json')['eligible']:raise ValueError('candidate not triggered')
    m,cfg=baseline_model(run,pressure=True);protocol=read(APP/'S3/new-scene-protocol.json');c=ObservableCoupling(m,cfg,protocol,'coarse');g=c.geometry;cap=min(256*2**20,int(.02*m.operator.memory_budget.initial_free));fast=DeviceGeometry.from_owned(g,cap_bytes=cap);records=[]
    for label in ('rest','manufactured','saved-1','saved-3'):
        if label.startswith('saved'):F=g.field(origin(0 if label=='saved-1' else 1)['state'].q)
        else:
            f=np.eye(3) if label=='rest' else np.array([[1.05,.12,0],[.02,.98,.07],[0,.03,1.02]])
            F=wp.full(g.count,wp.mat33d(f),dtype=wp.mat33d,device=m.device)
        host=F.numpy();tick=time.perf_counter();a,j=g.topology.assemble(g.X,g.total_weights,g.cell_ids,host,g.mobility);ta=time.perf_counter()-tick
        fast.assembler.assemble(F)  # Prewarm, not a trajectory step.
        wp.synchronize_device(m.device);tick=time.perf_counter();b,k=fast.assembler.assemble(F);tb=time.perf_counter()-tick
        passed=np.allclose(a,b,atol=1e-8,rtol=2e-5) and abs(j-k)<1e-12
        records.append(dict(state=label,H_max_error=float(np.max(abs(a-b))),original_s=ta,device_s=tb,passed=bool(passed),minJ=k,assembly_gain=1-tb/ta))
        del host,F
    q0=origin(0)['state'].q;q1=origin(1)['state'].q
    errors={k:0. for k in ('H','volume','gradient')};passed=True
    for q in (q0,q1):
        a=g.evaluate(q);b=fast.evaluate(q)
        for key in errors:
            errors[key]=max(errors[key],float(np.max(abs(a[key]-b[key]))));passed &=bool(np.allclose(a[key],b[key],atol=1e-8,rtol=2e-5))
        b['H'][:]=0
        if np.linalg.norm(fast.evaluate(q)['H'])==0:raise ValueError('cached result leaked ownership')
    da=g.discrete(q0,q1);db=fast.discrete(q0,q1);errors['discrete_gradient']=float(np.max(abs(da-db)));errors['pressure_work']=float(np.max(abs(np.einsum('cij,ij->c',da-db,q1-q0))))
    passed &=errors['pressure_work']<1e-10 and all(r['passed'] for r in records)
    gain=float(np.median([r['assembly_gain'] for r in records]));eligible=bool(passed and gain*.68>=.05)
    write(run/'S2/operator-equivalence.json',dict(status='passed_scoped' if passed else 'limited',records=records,geometry_errors=errors,additional_persistent_bytes=fast.assembler.bytes,additional_transient_bytes=fast.assembler.transient_bytes,setup_seconds=fast.assembler.build_seconds,numerical_source_sha256=numerical_sources(),current_F_always_used=True,owned_results=True))
    register(run,'S2/paired-protocol.json',dict(status='registered' if eligible else 'not_triggered',eligible=eligible,predicted_gain_not_end_to_end=.68*gain,source_states=[dict(path=str(origin(i)['folder']/'state.json'),sha256=sha(origin(i)['folder']/'state.json')) for i in (0,1)],order=['A0','B0','B1','A1'],with_frames=[False,True],max_attempts=8))
    print('DEVICE_OPERATOR',eligible,records,errors,flush=True)

def one(run,which,kind,fault=False):
    import warp as wp
    run=Path(run);verify(run);proto=read(run/'S2/paired-protocol.json')
    if not proto['eligible']:raise ValueError('paired test not eligible')
    item=origin(which)
    if sha(item['folder']/'state.json')!=proto['source_states'][which]['sha256']:raise ValueError('origin changed')
    started=time.perf_counter();m,cfg=baseline_model(run,pressure=True);p=read(APP/'S3/new-scene-protocol.json');c,bridge=construct(m,cfg,p,'coarse',state=item['state'],backend='original' if kind=='A' else 'device',bridge=kind!='A');wp.synchronize_device(m.device);build=time.perf_counter()-started
    name=('fault-' if fault else '')+kind+str(which);folder=run/'S2'/name
    identity=dict(schema='paired-current-RT0-v1',coupling=c.identity,initial_state_sha256=sha(item['folder']/'state.json'),sources=numerical_sources(),driver_sha256=sha(__file__),kind=kind)
    if (folder/'identity.json').exists():raise ValueError('paired trial already exists')
    write(folder/'identity.json',identity);snapshot(folder/'source',dict(numerical_sources(),**{str(Path(__file__).relative_to(ROOT)):sha(__file__)}));store=GenerationStore(folder,identity);store.save(c.state,[]);write(folder/'bridge.json',bridge or dict(physical_state_exact=True,original=True));cache=CachedProbes(m);parts=dict(field_s=0.,commit_s=0.)
    if fault:
        before=c.state.digest()
        def inject(stage,state):
            if stage=='before_commit':raise ValueError('registered fault before commit')
        write(folder/'attempt.json',dict(attempts=1,accepted=False))
        try:c.step(inject=inject)
        except Exception as e:
            if c.state.digest()!=before or store.load()['state'].digest()!=before:raise ValueError('fault rollback differs')
            write(folder/'failure.json',dict(status='passed_scoped',injected=True,error=repr(e),rollback_exact=True));print('FAULT_ROLLBACK',before,flush=True);return
        raise ValueError('fault did not trigger')
    save=store.save
    def publish(*args,**kwargs):
        tick=time.perf_counter()
        try:return save(*args,**kwargs)
        finally:parts['commit_s']+=time.perf_counter()-tick
    store.save=publish
    def emit(state):
        tick=time.perf_counter()
        try:return frame(c,cache,state)
        finally:parts['field_s']+=time.perf_counter()-tick
    before=sharing();write(folder/'attempt.json',dict(attempts=1,accepted=False));wp.synchronize_device(m.device);tick=time.perf_counter()
    row=advance_publish(c,store,[],frame_builder=emit if proto['with_frames'][which] else None);wp.synchronize_device(m.device);elapsed=time.perf_counter()-tick
    write(folder/'attempt.json',dict(attempts=1,accepted=True));write(folder/'measurement.json',dict(status='passed_scoped',advance_s=elapsed,build_s=build,**parts,sharing_before=before,sharing_after=sharing(),iterations=row['iterations'],residual_fraction=row['true_scaled_residual'],rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,additional_setup_s=c.geometry.assembler.build_seconds if kind=='B' else 0.,frame_written=proto['with_frames'][which],source_sha256=numerical_sources()))
    print('PAIR',name,elapsed,row['iterations'],flush=True)

def finish(run):
    run=Path(run);p=read(run/'S2/paired-protocol.json');records=[]
    if p['eligible']:
        for i in (0,1):
            a=history(run/f'S2/A{i}')[-1]['state'];b=history(run/f'S2/B{i}')[-1]['state'];checks={}
            for k,tol in [('q',1e-8),('velocity',1e-8),('predictor',1e-8)]:checks[k]=metric(getattr(a,k),getattr(b,k),tol,2e-5)
            for k,tol in [('pressure_Pa',1e-6),('flux_interval_m3_s',1e-10),('content_m3',1e-10),('cumulative_source_m3',1e-10),('cumulative_boundary_m3',1e-10)]:checks[k]=metric(a.child_states['fluid'][k],b.child_states['fluid'][k],tol,2e-5)
            A=read(run/f'S2/A{i}/measurement.json');B=read(run/f'S2/B{i}/measurement.json');gain=1-B['advance_s']/A['advance_s'];records.append(dict(index=i,A=A,B=B,gain=gain,checks=checks,passed=all(x['passed'] for x in checks.values()),setup_break_even_steps=max(0.,B['build_s']-A['build_s'])/max(A['advance_s']-B['advance_s'],1e-30)))
        gains=[r['gain'] for r in records];spread=float(np.ptp(gains));median=float(np.median(gains));exclusive=all(r[k][s]['exclusive'] for r in records for k in ('A','B') for s in ('sharing_before','sharing_after'));selected=all(r['passed'] for r in records) and exclusive and min(gains)>=0 and median>=max(.05,spread) and max(r['setup_break_even_steps'] for r in records)<=16
    else:selected=False;spread=median=0.;exclusive=False
    write(run/'S2/paired-performance.json',dict(status='passed_scoped' if selected else 'limited',records=records,median_gain=median,between_state_gain_range=spread,repeated_noise_estimate=False,exclusive=exclusive))
    write(run/'S2/performance-decision.json',dict(status='research_device_qualified' if selected else 'retain_original',selected=bool(selected),backend='device' if selected else 'original',formal_solid_changed=False,production_default_changed=False,scope='same observable fixture; second-grid operator must be checked',source_sha256=numerical_sources()))
    print('PERFORMANCE_DECISION',selected,median,spread,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['operators','A','B','fault','finish']);p.add_argument('--run',type=Path,required=True);p.add_argument('--which',type=int,choices=[0,1],default=0);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase in ('A','B','fault'):one(a.run,a.which,'B' if a.phase=='fault' else a.phase,a.phase=='fault')
        else:globals()[a.phase](a.run)
