"""Extend an authenticated independent fine prefix by sixteen steps."""
from pathlib import Path
import argparse,time
import numpy as np
from .provenance import *
from . import config
from .run import load_model
from benchmarks.research_candidate_observable_next.candidate import parents
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_restoring_rt0_next.reference import compare_pair
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_post_release.runtime_rules import advance_publish
from engine.aniso_phase1.research_post_release.fields import CachedProbes

def study(run):
    run=Path(run);verify(run)
    if read(APP/'S1/diagnosis-decision.json')['unexplained_numeric_anomaly']:raise ValueError('unexplained anomaly')
    origin=parents()[0];coarse=history(PREV/'cases/candidate-quarter')[:17]
    if coarse[0]['state'].digest()!=origin['state'].digest():raise ValueError('coarse origin differs')
    prior=read(PREV/'cases/candidate-quarter/execution-protocol.json');times=np.linspace(1.075,1.0751953125,33).tolist()
    if not np.allclose([x['state'].time for x in coarse],times[::2],rtol=0,atol=1e-14):raise ValueError('coarse times differ')
    register(run,'S2/reference-protocol.json',dict(status='registered',coarse_source=str(PREV/'cases/candidate-quarter'),origin_path=str(origin['folder']/'state.json'),origin_sha256=sha(origin['folder']/'state.json'),origin_digest=origin['state'].digest(),coarse_steps_inherited=16,fine_steps_inherited=16,fine_steps_new=16,times_s=times,coarse_times_s=times[::2],space=prior['physical_space'],source_compatibility=str(run/'S0/impact-and-compatibility.json'),coarse_reuse='same five numerical core modules and complete physical model; no S1 formula change',max_attempts=60,hard_seconds=600))
    cfg=config.make(read(run/'input-lock.json')['energy_scale_J'],space=prior['physical_space'],start=times[0],end=times[-1],times=times,initial=prior['initial_state'],field_cache=True,display_frames=1)
    m,_=load_model(run,cfg);m.validate(origin['state'],material=True);cache=CachedProbes(m);folder=run/'cases/candidate-extended-half'
    identity=dict(schema='pressure-startup-extended-reference-v1',model=m.identity,model_sha256=m.signature,initial_digest=origin['state'].digest(),numerical_source_sha256=source_files(),driver_sha256=sha(__file__),protocol_sha256=config.identity(cfg))
    if (folder/'identity.json').exists():raise ValueError('prefix already started')
    write(folder/'identity.json',identity);write(folder/'execution-protocol.json',cfg);snapshot(folder/'source',dict(source_files(),**{str(Path(__file__).relative_to(ROOT)):sha(__file__)}));store=GenerationStore(folder,identity);store.save(origin['state'],[])
    inherited=history(APP/'cases/candidate-prefix-half')
    if len(inherited)!=17 or inherited[0]['state'].digest()!=origin['state'].digest():raise ValueError('fine prefix origin/history mismatch')
    for i,item in enumerate(inherited):
        m.validate(item['state'],material=True)
        if abs(item['state'].time-times[i])>1e-14:raise ValueError('prefix time mismatch')
        if i:store.save(item['state'],item['rows'])
    write(run/'S2/prefix-reuse.json',dict(status='passed_scoped',source=str(APP/'cases/candidate-prefix-half'),origin_digest=origin['state'].digest(),prefix_final_digest=inherited[-1]['state'].digest(),inherited_steps=16,physical_states_unchanged=True,source_manifests=[dict(path=str(x['folder']/'manifest.json'),sha256=sha(x['folder']/'manifest.json')) for x in inherited],no_coarse_state_reset=True))
    st=ValidatedAVF(m,cfg,inherited[-1]['state']);rows=list(inherited[-1]['rows']);begun=time.perf_counter()
    for index,target in enumerate(times[1:],1):
        if index<=16:continue
        if time.perf_counter()-begun>600:raise TimeoutError('prefix budget')
        write(folder/'attempts.json',dict(attempts=index-16,inherited=16,committed=st.state.step))
        try:row=advance_publish(st,store,rows,target-st.state.time,frame_builder=cache.frame if index==32 else None)
        except Exception as e:
            write(folder/'failure.json',dict(status='limited',error=repr(e),rollback=store.load()['state'].digest()==st.state.digest()));raise
        rows.append(row);print('PREFIX',index,row['min_detF'],flush=True)
    write(folder/'ledger.json',rows);write(folder/'summary.json',dict(status='passed_scoped',new_attempts=16,inherited_steps=16,steps=32,min_detF=min(r['min_detF'] for r in rows),max_energy_closure_J=max(abs(r['budget_defect_J']) for r in rows),seconds=time.perf_counter()-begun,checkpoint_reload=store.load()['state'].digest()==st.state.digest()))
    for label,trajectory in [('coarse',coarse),('fine',history(folder))]:
        values={k:[] for k in ('x','velocity','PK1')}
        for item in trajectory:
            f=cache.frame(item['state'])
            for k in values:values[k].append(f[k])
        np.savez_compressed(run/'S2'/f'{label}-probes.npz',X=f['X'],fiber=m.parent.params.fiber_direction,times=[x['state'].time for x in trajectory],**{k:np.asarray(v) for k,v in values.items()})
    with np.load(run/'S2/coarse-probes.npz') as a,np.load(run/'S2/fine-probes.npz') as b:report=compare_pair(a,b,coarse[-1]['rows'],rows,2)
    write(run/'S2/short-reference-comparison.json',dict(status='passed_scoped' if report['passed'] else 'limited',**report,raw_event_status='unobserved',engineering_output_status='passed_scoped' if report['passed'] else 'limited',common_initial_digest=origin['state'].digest(),independent_fine_origin=True,diagnostic_probes_not_visual_frames=True))
    write(run/'S2/reference-scope.json',dict(status='passed_scoped' if report['passed'] else 'reference_limited',local_prefix_time_passed=report['passed'],old_full_window_time_passed=False,global_temporal_accuracy=False,spatial_retraining_eligible=report['passed'],new_steps=16,inherited_steps=16,window_s=[times[0],times[-1]],max_budget_fractions=report['max_budget_fractions']))
    print('PREFIX_DECISION',report['passed'],report['max_budget_fractions'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
