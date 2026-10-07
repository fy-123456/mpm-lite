"""Read-only physical-field review of the short new-space trajectory."""
from pathlib import Path
import argparse,copy
import numpy as np
from .provenance import *
from .run import load_model
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.compare import metric,regions
from engine.aniso_phase1.research_post_release.fields import CachedProbes

def review(run):
    run=Path(run);cfg=read(run/'cases/space-smoke/execution-protocol.json');new,_=load_model(run,cfg);oldcfg=copy.deepcopy(cfg);oldcfg['physical_space']=read(run/'baseline-space.json')['package'];old,_=load_model(run,oldcfg)
    nc,oc=CachedProbes(new),CachedProbes(old);h=history(run/'cases/space-smoke');baseline={round(x['state'].time,10):x for x in history(APP/'cases/final-full')};records=[]
    for row in h:
        t=row['state'].time;a,b=nc.frame(row['state']),oc.frame(baseline[round(t,10)]['state']);fields={}
        for reg,w in regions(a['X']).items():fields[reg]={k:metric(a[k]-a['X'] if k=='x' else a[k],b[k]-b['X'] if k=='x' else b[k],at,.05,w) for k,at in [('x',5e-5),('velocity',1e-4),('PK1',.02)]}
        records.append(dict(time_s=t,regions=fields,max_velocity_m_s=float(np.max(np.linalg.norm(a['velocity'],axis=-1))),old_max_velocity_m_s=float(np.max(np.linalg.norm(b['velocity'],axis=-1)))))
    maxima={k:max(e['regions'][reg][k]['absolute'] for e in records for reg in e['regions']) for k in ('x','velocity','PK1')};good=all(v['passed'] for e in records for row in e['regions'].values() for v in row.values())
    write(run/'S1/smoke-field-comparison.json',dict(status='passed_scoped' if good else 'requires_review',records=records,maxima=maxima,coordinate_comparison=False,common_physical_probes=True,new_integration_steps=0))
    if not good:raise ValueError('short dynamic fields need review before full-cycle adoption')
    peaks={}
    for label,path in [('F45',run/'S1/candidates/cross-direction-snapshot6/F45.npz'),('F60',run/'S1/candidates/cross-direction-snapshot6/F60.npz'),('reserved52.5',run/'S1/reserved/candidate/state.npz')]:
        with np.load(path) as z:
            P=z['PK1'];X=z['X'];angle=52.5 if label.startswith('reserved') else float(label[1:]);a=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0.]);fiber=np.einsum('i,...ij,j->...',a,P,a);values={}
            for key,v in [('PK1_Frobenius',np.linalg.norm(P,axis=(-2,-1))),('abs_fiber_PK1',abs(fiber))]:
                i=np.unravel_index(np.argmax(v),v.shape);values[key]=dict(max_Pa=float(v[i]),X_m=X[i].tolist())
            peaks[label]=values
    write(run/'S1/sample-field-extrema.json',dict(scope='33x7x7 public probes only; regional RMS separately integrated on common cells',records=peaks))
    print('SHORT_SPACE_FIELDS',maxima,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):review(a.run)
