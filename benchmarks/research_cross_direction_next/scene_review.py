"""Same physical outputs across old/new spaces; coefficients are never compared."""
from pathlib import Path
import argparse,copy
from .provenance import *
from .run import load_model
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.compare import metric,regions
from engine.aniso_phase1.research_post_release.fields import CachedProbes

def review(run):
    run=Path(run);cfg=read(run/'cases/final-full/execution-protocol.json');new,_=load_model(run,cfg);oldcfg=copy.deepcopy(cfg);oldcfg['physical_space']=read(run/'baseline-space.json')['package'];old,_=load_model(run,oldcfg);nc,oc=CachedProbes(new),CachedProbes(old)
    hs={label:{round(x['state'].time,10):x for x in history(folder)} for label,folder in [('new',run/'cases/final-full'),('old',APP/'cases/final-full')]};records=[]
    for t in (0.,.5,.6,1.075,1.1,1.225,1.25,1.6):
        a,b=nc.frame(hs['new'][t]['state']),oc.frame(hs['old'][t]['state']);fields={}
        for reg,w in regions(a['X']).items():fields[reg]={k:metric(a[k]-a['X'] if k=='x' else a[k],b[k]-b['X'] if k=='x' else b[k],at,.05,w) for k,at in [('x',5e-5),('velocity',1e-4),('PK1',.02)]}
        records.append(dict(time_s=t,regions=fields))
    raw_new=hs['new'][1.6]['rows'];raw_old=hs['old'][1.6]['rows'];reaction=[dict(time_s=a['time'],**metric(a['reaction_N'],b['reaction_N'],1e-4,.05)) for a,b in zip(raw_new,raw_old)]
    failed=[dict(time_s=e['time_s'],region=r,field=k,absolute=v['absolute'],budget=v['budget']) for e in records for r,row in e['regions'].items() for k,v in row.items() if not v['passed']]
    write(run/'S6/parent-scene-review.json',dict(status='diagnostic_new_space',records=records,reaction=reaction,outputs_outside_old_space_comparison_budget=failed,scope='old approximate space is a regression witness, not an accurate dynamic reference; no coefficient comparison or old temporal qualification transfer',new_steps=0))
    print('PARENT_SCENE_DIFFERENCES',len(failed),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):review(a.run)
