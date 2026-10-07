"""Recheck raw events and sampled nonlinear output sensitivity without stepping."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import *
from .run import load_model
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_observable_boundary_next.time_study import reaction
from benchmarks.research_observable_boundary_next.events import events,difference
from engine.aniso_phase1.endpoint_boundary import prescribed_speed
from engine.aniso_phase1.research_post_release.fields import CachedProbes


def review(run):
    run=Path(run);verify(run);cfg=read(APP/'cases/final-full/execution-protocol.json');m,_=load_model(run,cfg)
    settings=read(APP/'S1/time-reference-protocol.json');modal=read(APP/'S1/modal-observable-map.json')
    with np.load(APP/'S1/modal-basis.npz') as z:V=z['vectors'].copy()
    K=m.rest_K[np.ix_(m.ids,m.ids)];orth=float(np.linalg.norm(V.T@m.M3ff@V-np.eye(V.shape[1])))
    if orth>1e-6:raise ValueError('inherited modes differ from full mass')
    rec=[]
    for window in range(2):
        hh=[history(APP/f'cases/window{window}-{s}') for s in ('h','half')];raw=[]
        for j in settings['important_modes']:
            signals=[]
            for hist in hh:
                signals.append(events([x['state'].time for x in hist],[float(V[:,j]@m.M3ff@(x['state'].velocity-m.boundary.unit*prescribed_speed(x['state'].time))[m.free].ravel()) for x in hist]))
            cmp=difference(*signals,settings['event_budgets_s'][str(j)]);observed=bool(signals[0]['events'] or signals[1]['events'])
            raw.append(dict(mode=j,comparison=cmp,status=cmp['status'] if observed else 'unobserved'))
        rec.append(dict(window=window,events=raw,source_identities=[sha(APP/f'cases/window{window}-{s}/identity.json') for s in ('h','half')]))
    hist=history(APP/'cases/final-full');target=max([x for x in modal['attribution'] if x['mode']==615],key=lambda x:x['eta']['complete_reaction']);i=min(range(1,len(hist)),key=lambda i:abs(hist[i]['state'].time-target['time_s']));s0,s1=hist[i-1]['state'],hist[i]['state'];cache=CachedProbes(m);samples=[]
    for factor in (0.,.5,1.):
        altered=[]
        for s in (s0,s1):
            out=s.clone();qf=(s.q-m.boundary.lift(s.time))[m.free].ravel();vf=(s.velocity-m.boundary.unit*prescribed_speed(s.time))[m.free].ravel();phi=V[:,615]
            out.q[m.free]-=factor*(phi*(phi@m.M3ff@qf)).reshape(-1,3);out.velocity[m.free]-=factor*(phi*(phi@m.M3ff@vf)).reshape(-1,3);altered.append(out)
        R=reaction(m,cfg,*altered);P=cache.frame(altered[1])['PK1'];samples.append(dict(factor=factor,reaction=R,stress=P))
    if abs(samples[0]['reaction']['total']-hist[i]['rows'][-1]['reaction_N'])>1e-7:raise ValueError('complete saved reaction not reproduced')
    remainderR=abs(samples[2]['reaction']['total']-2*samples[1]['reaction']['total']+samples[0]['reaction']['total']);remainderP=float(np.max(abs(samples[2]['stress']-2*samples[1]['stress']+samples[0]['stress'])))
    result=dict(status='limited',accepted_steps=0,reused_short_steps=384,events=rec,mass_orthogonality=orth,sampled_sensitivity=dict(mode=615,time_s=s1.time,reactions=[dict(factor=x['factor'],**x['reaction']) for x in samples],second_difference_reaction_N=remainderR,second_difference_PK1_Pa=remainderP),all_important_modes_retained=True,uniform_in_time_bound=False,unobserved_events_passed=False,inertia_stabilization_endpoint_included=True,old_scope_preserved=True)
    write(run/'S4/observable-phase-review.json',result)
    times=read(APP/'S6/final-protocol.json')['times'];write(run/'S4/time-decision.json',dict(status='retain_252_scoped',times=times,steps=len(times)-1,trial=False,new_accepted_steps=0,global_temporal_accuracy=False,reason='raw events and sampled sensitivity do not establish uniform output insignificance; no new full-cycle step increase'))
    print('PHASE_REVIEW',len(times)-1,remainderR,remainderP,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):review(a.run)
