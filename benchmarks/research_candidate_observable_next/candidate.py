"""Authenticated candidate continuation and same-space time reference, one model per process."""
from pathlib import Path
import argparse,time
import numpy as np
from .provenance import *
from .run import load_model
from . import config
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_pressure_window_next.candidate_study import fields,good
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric,impulse_average
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_post_release.runtime_rules import advance_publish
from engine.aniso_phase1.research_post_release.fields import CachedProbes


def parents():
    a=history(APP/'cases/candidate-h');b=history(APP/'cases/candidate-h-continuation')
    if len(a)!=43 or len(b)!=21 or a[-1]['state'].digest()!=b[0]['state'].digest():raise ValueError('candidate prefix chain changed')
    if b[-1]['state'].step!=62 or b[-1]['state'].digest()!='3f668f27321ef2448e461924124830b20d0112fdcbefece9b8e1932b1b8f7237':raise ValueError('candidate step62 changed')
    if a[0]['state'].digest()!='70c8c51157b8e3708c7045d676ce4fea71e2f0a7b0a62758791abbed3af29816':raise ValueError('projected origin changed')
    return a+b[1:]


def prepare(run):
    run=Path(run);verify(run);h=parents();cfg=read(APP/'cases/candidate-h/execution-protocol.json')
    if sha(APP/'cases/candidate-h/execution-protocol.json')!=sha(APP/'cases/candidate-h-continuation/execution-protocol.json'):raise ValueError('candidate prefix protocols differ')
    record=dict(status='passed_scoped',source_initial=str(h[0]['folder']/'state.json'),source_initial_sha256=sha(h[0]['folder']/'state.json'),source_initial_digest=h[0]['state'].digest(),source_last=str(h[-1]['folder']/'state.json'),source_last_sha256=sha(h[-1]['folder']/'state.json'),source_last_digest=h[-1]['state'].digest(),prefix_steps=62,prefix_read_only=True,velocity_predictor_history_preserved=True)
    register(run,'S1/checkpoint-bridge.json',record)
    register(run,'S1/continuation-protocol.json',dict(source=record,remaining_steps=2,times_s=cfg['times'][63:],witness='one replay of old step61 to step62, separate process',max_attempts=3,single_live_model=True))
    register(run,'S1/candidate-half-protocol.json',dict(source=record,origin='original projected step0, not continuation62',times_s=np.linspace(1.075,1.078125,129).tolist(),steps=128,max_attempts=128,mass_order=7,material_order=7,display_frames=3))
    print('CANDIDATE_BRIDGE',record['source_last_digest'],flush=True)


def cycle(run,kind):
    run=Path(run);verify(run);h=parents();cfg=read(APP/'cases/candidate-h/execution-protocol.json');m,_=load_model(run,cfg)
    if kind=='witness':
        st=ValidatedAVF(m,cfg,h[-2]['state']);st.step(h[-1]['state'].time-st.state.time)
        errors={key:float(np.max(abs(getattr(st.state,key)-getattr(h[-1]['state'],key)))) for key in ('q','velocity','predictor')}
        differences=[]
        def compare_tree(a,b,path=''):
            if isinstance(a,dict) and isinstance(b,dict):
                for k in set(a)|set(b):compare_tree(a.get(k),b.get(k),path+'/'+k)
            elif isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
                if a!=b:
                    if a and all(isinstance(v,(float,int)) for v in a+b):differences.append(dict(path=path,max_error=float(np.max(abs(np.asarray(a)-np.asarray(b))))))
                    else:
                        for i,(x,y) in enumerate(zip(a,b)):compare_tree(x,y,path+'/'+str(i))
            elif a!=b:differences.append(dict(path=path,actual=a,expected=b))
        compare_tree(st.state.to_dict(),h[-1]['state'].to_dict())
        physical_differences=[x for x in differences if x['path']!='/child_states/last_ledger/wall_seconds']
        write(run/'S1/prefix-witness.json',dict(status='passed_scoped' if not physical_differences else 'needs_review',attempts=2,errors=errors,full_state_digest_equal=not differences,differences=differences,physical_state_exact=not physical_differences,only_nonnumerical_wall_time_excluded=True,original_checkpoint_digest_still_strict=True))
        if physical_differences:raise ValueError('prefix replay differs; see exact structural differences')
        return
    name='candidate-tail' if kind=='tail' else 'candidate-half';folder=run/'cases'/name
    origin=h[-1] if kind=='tail' else h[0]
    times=cfg['times'] if kind=='tail' else read(run/'S1/candidate-half-protocol.json')['times_s']
    if kind=='half':cfg=config.make(read(run/'input-lock.json')['energy_scale_J'],space=cfg['physical_space'],start=times[0],end=times[-1],times=times,initial=cfg['initial_state'],display_frames=3)
    identity=dict(schema='candidate-observable-solid-branch-v1',model=m.identity,model_sha256=m.signature,initial_digest=origin['state'].digest(),source_checkpoint=str(origin['folder']/'state.json'),source_checkpoint_sha256=sha(origin['folder']/'state.json'),numerical_source_sha256=source_files(),driver_sha256=sha(__file__),protocol_sha256=config.identity(cfg))
    if (folder/'identity.json').exists():
        if read(folder/'identity.json')!=identity:raise ValueError('branch sources changed')
    else:
        write(folder/'identity.json',identity);write(folder/'execution-protocol.json',cfg);snapshot(folder/'source',source_files())
    store=GenerationStore(folder,identity);loaded=store.load(validator=m.validate);cache=CachedProbes(m)
    if loaded is None:store.save(origin['state'],[],frame=cache.frame(origin['state']));initial=origin['state'];rows=[]
    else:store.history();initial=loaded['state'];rows=loaded['rows']
    stepper=ValidatedAVF(m,cfg,initial);started=time.perf_counter();attempts=0
    for target in times[stepper.state.step+1:]:
        attempts+=1
        write(folder/'attempts.json',dict(attempts=attempts,start_global_step=initial.step,current_committed_step=stepper.state.step))
        try:row=advance_publish(stepper,store,rows,float(target-stepper.state.time),frame_builder=cache.frame if stepper.state.step+1 in (len(times)//2,len(times)-1) else None)
        except Exception as error:
            write(folder/'failure.json',dict(status='limited',error=repr(error),last_committed_step=stepper.state.step,digest=stepper.state.digest()));raise
        rows.append(row)
        if stepper.state.step%16==0:print(name,stepper.state.step,'detF',row['min_detF'],flush=True)
        if time.perf_counter()-started>1200:raise TimeoutError('candidate case hard time budget')
    if store.load()['state'].digest()!=stepper.state.digest():raise ValueError('published state differs')
    write(folder/'ledger.json',rows);write(folder/'summary.json',dict(status='passed_scoped',new_attempts=attempts,accepted_steps=len(rows),global_step=stepper.state.step,end_s=stepper.state.time,min_detF=min(r['min_detF'] for r in rows),max_energy_closure_J=max(abs(r['budget_defect_J']) for r in rows),seconds=time.perf_counter()-started,checkpoint_reload=True))
    if kind=='tail':write(run/'S1/candidate-coarse-check.json',dict(status='passed_scoped',inherited_steps=62,new_steps=2,total_steps=64,end_s=stepper.state.time,formal_space_changed=False))


def trajectory(run,label):
    if label=='coarse':return parents()+history(Path(run)/'cases/candidate-tail')[1:]
    if label=='half':return history(Path(run)/'cases/candidate-half')
    return history(OLD/'cases/window0-half')


def export(run,which):
    run=Path(run);verify(run);labels=('coarse','half') if which=='candidate' else ('formal',)
    cfg=read((APP/'cases/candidate-h' if which=='candidate' else APP/'cases/final-full')/'execution-protocol.json');m,_=load_model(run,cfg);cache=CachedProbes(m)
    for label in labels:
        h=trajectory(run,label);values={k:[] for k in ('x','velocity','PK1')};rows=[];times=[]
        for i,item in enumerate(h):
            f=cache.frame(item['state']);times.append(item['state'].time)
            for k in values:values[k].append(f[k])
            if i:rows.append(item['rows'][-1])
        np.savez_compressed(run/'S1'/f'{label}-physical-probes.npz',X=f['X'],times=np.array(times),fiber=m.parent.params.fiber_direction,**{k:np.asarray(v) for k,v in values.items()})
        write(run/'S1'/f'{label}-raw-rows.json',rows)
    write(run/'S1'/f'{which}-probe-source.json',dict(model=m.identity,driver_sha256=sha(__file__),one_live_model=True,diagnostic_probes_are_not_extra_visual_frames=True))


def compare(run):
    run=Path(run);reports={}
    for label,other,stride in (('coarse','half',2),('half','formal',1)):
        with np.load(run/'S1'/f'{label}-physical-probes.npz') as a,np.load(run/'S1'/f'{other}-physical-probes.npz') as b:
            rows=read(run/'S1'/f'{label}-raw-rows.json');rhs=read(run/'S1'/f'{other}-raw-rows.json');records=[]
            for i,row in enumerate(rows,1):
                j=stride*i
                if abs(a['times'][i]-b['times'][j])>1e-12:raise ValueError('reference nodes differ')
                fa={k:a[k][i] for k in ('x','velocity','PK1')};fb={k:b[k][j] for k in fa};fa['X']=a['X'];fb['X']=b['X']
                f=fields(fa,fb,a['fiber']);r=metric(row['reaction_N'],impulse_average(rhs,row['time']-row['dt'],row['time']),1e-4,.05)
                records.append(dict(time_s=float(a['times'][i]),fields=f,reaction=r,passed=good(f) and r['passed']))
        reports[label+'_vs_'+other]=dict(passed=all(r['passed'] for r in records),records=records,max_velocity_budget_fraction=max(v['velocity']['absolute']/v['velocity']['budget'] for r in records for v in r['fields'].values()),max_reaction_budget_fraction=max(r['reaction']['absolute']/r['reaction']['budget'] for r in records))
    write(run/'S1/time-vs-space-comparison.json',dict(status='passed_scoped' if reports['coarse_vs_half']['passed'] else 'limited',records=reports,comparison_frames_match_physical_nodes=True,formal_fine_is_not_continuum_truth=True))
    decision=dict(status='limited',candidate_time_refinement_passed=reports['coarse_vs_half']['passed'],candidate_vs_formal_fine_passed=reports['half_vs_formal']['passed'],coarse_complete=True,fine_complete=True,short_dynamic_stable=True,new_accepted_steps=130,witness_attempts=2,formal_space_changed=False,spatial_accuracy=False,q5=False,coupled=False)
    write(run/'S1/candidate-reference-decision.json',decision);print('CANDIDATE_REFERENCE',decision,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','witness','tail','half','candidate','formal','compare']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase in ('witness','tail','half'):cycle(a.run,a.phase)
        elif a.phase in ('candidate','formal'):export(a.run,a.phase)
        else:globals()[a.phase](a.run)
