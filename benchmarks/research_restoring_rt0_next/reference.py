"""One candidate quarter-step reference and common physical output attribution."""
from pathlib import Path
import argparse,time
import numpy as np
import scipy.linalg as la
from .provenance import *
from . import config
from .run import load_model
from benchmarks.research_candidate_observable_next.candidate import parents
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_pressure_window_next.candidate_study import fields,good
from benchmarks.research_sequential_next.compare import metric,impulse_average
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_post_release.runtime_rules import advance_publish
from engine.aniso_phase1.research_post_release.fields import CachedProbes


def prepare(run):
    run=Path(run);verify(run);item=parents()[0]
    if item['state'].digest()!='70c8c51157b8e3708c7045d676ce4fea71e2f0a7b0a62758791abbed3af29816':raise ValueError('candidate origin differs')
    prior=read(SOLID/'cases/candidate-h/execution-protocol.json');times=np.linspace(1.075,1.078125,257).tolist()
    register(run,'S1/reference-protocol.json',dict(status='registered',steps=256,times_s=times,origin_path=str(item['folder']/'state.json'),origin_sha256=sha(item['folder']/'state.json'),origin_digest=item['state'].digest(),mass_order=7,material_order=7,display_frames=2,diagnostic_indices=[0,64,128,192,256],candidate_package=prior['physical_space'],formal_reference=str(APP/'cases/phase-quarter'),max_attempts=260,hard_seconds=1200))
    cfg=config.make(read(run/'input-lock.json')['energy_scale_J'],space=prior['physical_space'],start=times[0],end=times[-1],times=times,initial=prior['initial_state'],field_cache=True,display_frames=2)
    write(run/'cases/candidate-quarter/execution-protocol.json',cfg)


def cycle(run):
    run=Path(run);verify(run);p=read(run/'S1/reference-protocol.json');folder=run/'cases/candidate-quarter';cfg=read(folder/'execution-protocol.json');m,_=load_model(run,cfg);origin=parents()[0]['state'];m.validate(origin,material=True)
    identity=dict(schema='restoring-candidate-reference-v1',model=m.identity,model_sha256=m.signature,initial_digest=origin.digest(),source_sha256=p['origin_sha256'],numerical_source_sha256=source_files(),driver_sha256=sha(__file__),protocol_sha256=config.identity(cfg))
    ip=folder/'identity.json'
    if ip.exists():
        if read(ip)!=identity:raise ValueError('candidate identity changed')
    else:write(ip,identity);snapshot(folder/'source',dict(source_files(),**{str(Path(__file__).relative_to(ROOT)):sha(__file__)}))
    store=GenerationStore(folder,identity);loaded=store.load(validator=m.validate);cache=CachedProbes(m)
    if loaded is None:store.save(origin,[],frame=cache.frame(origin));state=origin;rows=[]
    else:store.history();state=loaded['state'];rows=loaded['rows']
    stepper=ValidatedAVF(m,cfg,state);begun=time.perf_counter();old=read(folder/'attempts.json') if (folder/'attempts.json').exists() else {};attempts=old.get('attempts',0);elapsed=old.get('seconds',0.)
    for index,target in enumerate(cfg['times'][1:],1):
        if index<=stepper.state.step:continue
        if attempts>=p['max_attempts'] or elapsed+time.perf_counter()-begun>p['hard_seconds']:raise TimeoutError('candidate budget exhausted')
        attempts+=1;write(folder/'attempts.json',dict(attempts=attempts,seconds=elapsed+time.perf_counter()-begun))
        try:row=advance_publish(stepper,store,rows,target-stepper.state.time,frame_builder=cache.frame if index==256 else None)
        except Exception as e:
            write(folder/'failure.json',dict(status='limited',error=repr(e),committed=stepper.state.step,rollback=store.load()['state'].digest()==stepper.state.digest()));raise
        rows.append(row)
        if index%32==0:print('CANDIDATE_QUARTER',index,row['min_detF'],flush=True)
    write(folder/'attempts.json',dict(attempts=attempts,seconds=elapsed+time.perf_counter()-begun));write(folder/'ledger.json',rows)
    write(folder/'summary.json',dict(status='passed_scoped',steps=len(rows),end_s=stepper.state.time,min_detF=min(r['min_detF'] for r in rows),seconds=elapsed+time.perf_counter()-begun,checkpoint_reload=store.load()['state'].digest()==stepper.state.digest()))


def export(run,label):
    run=Path(run);verify(run);folder=run/'cases/candidate-quarter' if label=='candidate' else APP/'cases/phase-quarter';cfg=read(folder/'execution-protocol.json');m,_=load_model(run,cfg);cache=CachedProbes(m);h=history(folder);out={k:[] for k in ('x','velocity','PK1')};diagnostics=[];components={k:[] for k in ('material','stabilization','boundary_cross_mass','total')}
    if len(h)!=257:raise ValueError('incomplete reference')
    for index,item in enumerate(h):
        s=item['state'];f=cache.frame(s)
        for k in out:out[k].append(f[k])
        if index in (0,64,128,192,256):
            response=m.evaluate(s.q);fm=response['material_force'];fs=response['force']-fm;parts={}
            for k,force in [('material',fm),('stabilization',fs)]:
                a=np.zeros_like(s.q);a[m.free]=la.cho_solve(m.mass_factor,-force[m.free]);parts[k]=cache.maps[0]@a
            if not 1.075-1e-12<=s.time<=1.078125+1e-12:raise ValueError('boundary acceleration formula outside registered unload phase')
            ab=m.boundary.unit*(-.005*2*np.pi**2*np.cos(2*np.pi*(s.time-.6)));acc=ab.copy();acc[m.free]=la.cho_solve(m.mass_factor,-m.M[np.ix_(m.free,m.fixed)]@ab[m.fixed]);parts['boundary_cross_mass']=cache.maps[0]@acc;parts['total']=sum(parts.values())
            direct=ab.copy();direct[m.free]=la.cho_solve(m.mass_factor,-response['force'][m.free]-m.M[np.ix_(m.free,m.fixed)]@ab[m.fixed]);err=float(np.max(abs(cache.maps[0]@direct-parts['total'])))
            if err>1e-8:raise ValueError('acceleration parts do not reconstruct')
            for k in components:components[k].append(parts[k])
            diagnostics.append(dict(local_index=index,global_step=s.step,time_s=s.time,reconstruction_error=err,material_J=response['material_U'],stabilization_J=response['stabilization_U'],kinetic_J=m.kinetic(s.velocity),state_sha256=sha(item['folder']/'state.json')))
    np.savez_compressed(run/'S1'/f'{label}-quarter-probes.npz',X=f['X'],fiber=m.parent.params.fiber_direction,times=[x['state'].time for x in h],**{k:np.array(v) for k,v in out.items()})
    np.savez_compressed(run/'S1'/f'{label}-acceleration.npz',**{k:np.array(v) for k,v in components.items()})
    write(run/'S1'/f'{label}-diagnostic.json',dict(status='passed_scoped',source=str(folder),model=m.identity,records=diagnostics,driver_sha256=sha(__file__)));write(run/'S1'/f'{label}-quarter-rows.json',h[-1]['rows'])
    print('EXPORTED',label,len(h),flush=True)


def compare_pair(a,b,rows,rhs,stride):
    # NPZ indexing decompresses an entire array each time; materialize once.
    a={k:a[k] for k in a.files};b={k:b[k] for k in b.files}
    records=[]
    for i,row in enumerate(rows,1):
        j=i*stride
        if abs(a['times'][i]-b['times'][j])>1e-12:raise ValueError('physical time mismatch')
        fa=dict(X=a['X'],**{k:a[k][i] for k in ('x','velocity','PK1')});fb=dict(X=b['X'],**{k:b[k][j] for k in ('x','velocity','PK1')});v=fields(fa,fb,a['fiber']);R=metric(row['reaction_N'],impulse_average(rhs,row['time']-row['dt'],row['time']),1e-4,.05);records.append(dict(time_s=row['time'],fields=v,reaction=R,passed=good(v) and R['passed']))
    maxima={k:max(z[k]['absolute']/z[k]['budget'] for row in records for z in row['fields'].values()) for k in ('x','velocity','PK1','fiber')};maxima['reaction']=max(r['reaction']['absolute']/r['reaction']['budget'] for r in records)
    return dict(passed=all(x['passed'] for x in records),max_budget_fractions=maxima,records=records)


def analyze(run):
    run=Path(run);reports={}
    with np.load(APP/'S1/half-physical-probes.npz') as a,np.load(run/'S1/candidate-quarter-probes.npz') as b,np.load(run/'S1/formal-quarter-probes.npz') as c:
        rows=read(run/'S1/candidate-quarter-rows.json');reports['candidate_half_vs_quarter']=compare_pair(a,b,read(APP/'S1/half-raw-rows.json'),rows,2);reports['candidate_vs_formal_quarter']=compare_pair(b,c,rows,read(run/'S1/formal-quarter-rows.json'),1)
    old=read(APP/'S1/time-vs-space-comparison.json')['records']['coarse_vs_half'];shrinks={k:reports['candidate_half_vs_quarter']['max_budget_fractions'][k]/old['max_'+k+'_budget_fraction'] for k in ('velocity','reaction')};passed=reports['candidate_half_vs_quarter']['passed'] and all(v<=.8 for v in shrinks.values())
    write(run/'S1/time-comparison.json',dict(status='passed_scoped' if passed else 'limited',records=reports,old_error_ratios=shrinks,common_physical_nodes=True,formal_is_not_continuum_truth=True))
    with np.load(run/'S1/candidate-acceleration.npz') as a,np.load(run/'S1/formal-acceleration.npz') as b:
        parts={k:[float(la.norm(x-y)/np.sqrt(len(x))) for x,y in zip(a[k],b[k])] for k in a.files}
    write(run/'S1/force-attribution.json',dict(status='diagnostic',indices=[0,64,128,192,256],physical_rms_difference_m_s2=parts,norms_not_additive=True,original_stabilization_preserved=True))
    write(run/'S1/candidate-reference-decision.json',dict(status='passed_scoped' if passed else 'reference_limited',same_space_time_passed=passed,cross_space_passed=reports['candidate_vs_formal_quarter']['passed'],ratios=reports['candidate_half_vs_quarter']['max_budget_fractions'],shrinkage=shrinks,new_steps=256,formal_space_changed=False,global_spatial_accuracy=False,global_temporal_accuracy=False,spatial_retraining_eligible=passed))
    print('REFERENCE_DECISION',passed,reports['candidate_half_vs_quarter']['max_budget_fractions'],shrinks,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','cycle','candidate','formal','analyze']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase in ('candidate','formal'):export(a.run,a.phase)
        else:globals()[a.phase](a.run)
