"""Bounded modal output diagnostics: physical regions and complete interval reaction."""
from pathlib import Path
import argparse
import numpy as np
import scipy.linalg as la
from .provenance import *
from .run import load_model
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_observable_boundary_next.time_study import reaction
from benchmarks.research_pressure_window_next.candidate_study import fields
from benchmarks.research_sequential_next.compare import regions
from engine.aniso_phase1.research_post_release.fields import CachedProbes


def outputs(run,label):
    run=Path(run);folder=run/'cases/candidate-half' if label=='candidate' else OLD/'cases/window0-half';h=history(folder);cfg=read(folder/'execution-protocol.json');m,_=load_model(run,cfg);cache=CachedProbes(m);s0,s1=h[0]['state'],h[1]['state'];lam,V=la.eigh(m.rest_K[np.ix_(m.ids,m.ids)],m.M3ff);force=m.evaluate(s0.q)['force'][m.free].ravel();q=(s0.q-m.boundary.lift(s0.time))[m.free].ravel();v=(s0.velocity-m.boundary.speed(s0.time))[m.free].ravel();energy=.5*((V.T@m.M3ff@v)**2+lam*(V.T@m.M3ff@q)**2);ids=sorted(set((int(np.argmax(energy)),int(np.argmax(abs(V.T@force))))));baseR=reaction(m,cfg,s0,s1);base=cache.frame(s1);records=[]
    if abs(baseR['total']-h[1]['rows'][-1]['reaction_N'])>1e-7:raise ValueError('complete interval reaction reconstruction failed')
    for j in ids:
        samples=[]
        for factor in (.5,1.):
            changed=[]
            for s in (s0,s1):
                out=s.clone();qf=(s.q-m.boundary.lift(s.time))[m.free].ravel();vf=(s.velocity-m.boundary.speed(s.time))[m.free].ravel();phi=V[:,j];out.q[m.free]-=factor*(phi*(phi@m.M3ff@qf)).reshape(-1,3);out.velocity[m.free]-=factor*(phi*(phi@m.M3ff@vf)).reshape(-1,3);changed.append(out)
            R=reaction(m,cfg,*changed);R['stabilization']=R['total']-R['material']-R['inertia']-R['endpoint'];samples.append(dict(factor=factor,regional_fields=fields(cache.frame(changed[1]),base,m.parent.params.fiber_direction),complete_interval_reaction=R,reaction_delta_N=R['total']-baseR['total']))
        records.append(dict(mode=j,period_s=float(2*np.pi/np.sqrt(lam[j])),samples=samples))
    write(run/'S2'/f'{label}-modal-output.json',dict(status='diagnostic',sampled_interval=[s0.time,s1.time],source_initial_sha256=sha(h[0]['folder']/'state.json'),source_final_sha256=sha(h[1]['folder']/'state.json'),complete_reaction=baseR,records=records,trajectories_not_modified=True,uniform_error_bound=False))
    paths=[run/'S2'/f'{v}-modal-output.json' for v in ('formal','candidate')]
    if all(p.exists() for p in paths):
        prior=read(run/'S2/modal-output-diagnostic.json');prior['actual_output_directions']={p.stem:read(p) for p in paths};write(run/'S2/modal-output-diagnostic.json',prior)
        with np.load(run/'S2/formal-physical-diagnostic.npz') as a,np.load(run/'S2/candidate-physical-diagnostic.npz') as b:
            regional={region:{k:float(np.sqrt(np.sum(w.reshape(-1,1)*(b[k]-a[k])**2)/np.sum(w))) for k in ('material','stabilization','boundary_cross_mass','total')} for region,w in regions(cache.X).items()}
        old=read(run/'S2/physical-force-acceleration-decomposition.json');old['regional_rms_difference_m_s2']=regional;old['inherited_mass_equilibration']=dict(path=str(APP/'S3/mass-conditioning-review.json'),sha256=sha(APP/'S3/mass-conditioning-review.json'));write(run/'S2/physical-force-acceleration-decomposition.json',old)
        old=read(run/'S2/research-space-decision.json');old['single_next_experiment']='same frozen candidate origin and original stabilization, use a short finer time reference focused on measured high-restoring-force directions; stop if time refinement still exceeds regional velocity/reaction budgets; no retraining or stabilization change simultaneously';write(run/'S2/research-space-decision.json',old)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['formal','candidate']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):outputs(a.run,a.phase)
