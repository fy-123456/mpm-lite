"""N08a: verify exact segmented maps before considering their performance."""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
from .provenance import read,write,serial_lock,source_files,utc
from .run import load_model,probe_frame
from .checkpoint import GenerationStore
from .compare import metric,regions
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF


def segmented_study(run):
    run=Path(run);cfg=read(run/'cases/gpu-q7-dt0025/execution-protocol.json');old,_=load_model(run,cfg)
    new=SegmentedModel(old.reduction,order=7,device='cuda:0')
    folder=run/'cases/gpu-q7-dt0025';history=GenerationStore(folder,read(folder/'identity.json')).history()
    by_time={round(x['state'].time,10):x['state'] for x in history}
    rng=np.random.default_rng(183);d=rng.normal(size=by_time[0.].q.shape);d[old.fixed]=0.;d/=np.linalg.norm(d)
    protocol=dict(utc=utc(),source_sha256=source_files(),operator='segmented transpose CSR, chunk 128, float64',
        states=[0.,.5,1.1],random_seed=183,relative_gate=2e-5,
        warm_repeats=3,window=[.45,.5],same_mass=True,same_tolerances=cfg['solver'])
    if (run/'N08/segmented-protocol.json').exists():raise ValueError('segmented study already registered')
    write(run/'N08/segmented-protocol.json',protocol)
    checks=[]
    for t in protocol['states']:
        q=by_time[t].q;a=old.evaluate(q,d);b=new.evaluate(q,d)
        errors={k:float(np.linalg.norm(np.asarray(a[k])-b[k])/max(np.linalg.norm(a[k]),1e-12)) for k in ('U','force','tangent_action')}
        if max(errors.values())>2e-5:raise ValueError(dict(time=t,errors=errors))
        checks.append(dict(time=t,errors=errors))
    warm=[]
    for repeat in range(3):
        states={}
        for name,model in ([('original',old),('segmented',new)] if repeat%2==0 else [('segmented',new),('original',old)]):
            model._cached_q=None;model._cached_response=None
            initial=by_time[.45].clone();initial.child_states['identity']=copy.deepcopy(model.identity)
            start=time.perf_counter();integrator=ValidatedAVF(model,cfg,initial)
            rows=[integrator.step(.025),integrator.step(.025)];seconds=time.perf_counter()-start
            states[name]=integrator.state
            warm.append(dict(repeat=repeat,variant=name,seconds=seconds,rows=rows))
            print(name,'two-step seconds',round(seconds,5),flush=True)
        error=max(float(np.max(abs(states['original'].q-states['segmented'].q))),
                  float(np.max(abs(states['original'].velocity-states['segmented'].velocity))))
        if error>1e-8:raise ValueError('segmented dynamics differs unexpectedly: '+str(error))
    medians={name:float(np.median([x['seconds'] for x in warm if x['variant']==name])) for name in ('original','segmented')}
    result=dict(utc=utc(),status='passed_scoped',same_state_checks=checks,warm_samples=warm,median_seconds=medians,
        relative_saving=1-medians['segmented']/medians['original'],
        adoption='eligible if repeated saving exceeds measured spread; cache measured separately',
        limitation='same FP64 mathematical map with reordered additions; no physics, quadrature or mass change')
    write(run/'N08/segmented-result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):segmented_study(a.run)
