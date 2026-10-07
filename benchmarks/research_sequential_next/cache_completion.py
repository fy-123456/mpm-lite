"""Close the real ownership, sliced CPU and resource gates without rerunning N08."""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
from .provenance import read,write,serial_lock,source_files,utc,register_study
from .checkpoint import GenerationStore
from .run import load_model
from engine.aniso_phase1.research_sequential_next.linearization import CachedModel,LinearizationCache
from engine.aniso_phase1.research_sequential_next.cpu_linearization import CPUStreamingLinearizations
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF,StepRejected
from engine.aniso_phase1.research_b.tensor import TensorMaterialOperator,TensorRule


def complete(run):
    run=Path(run);cfg=read(run/'cases/gpu-q7-dt0025/execution-protocol.json')
    cfg['implementation']=dict(operator='segmented',linearization_cache=False,preconditioner='original',cache_bytes=3<<30,cache_entries=4)
    register_study(run,'N08/completion-protocol.json',dict(utc=utc(),cpu_directions=2,cpu_budget_bytes=512<<20,
        gpu_cache_bytes=3<<30,gpu_cache_entries=4,states=[0.,.5,1.1],seed=881,relative_gate=2e-5,
        scope='fill missing ownership/CPU/resource gates; inherited N08 timings stay immutable'))
    base,_=load_model(run,cfg);cached=CachedModel(base.reduction,order=7,device='cuda:0')
    folder=run/'cases/gpu-q7-dt0025';hist=GenerationStore(folder,read(folder/'identity.json')).history()
    states={round(x['state'].time,10):x['state'] for x in hist}
    q=states[.5].q;full=base.reduction.expand(q);token,_=cached.linearizations.prepare(full)
    foreign=LinearizationCache(base.operator,max_bytes=3<<30,max_entries=1)
    rejected={}
    for name,operation in [
        ('foreign_cache',lambda:foreign.action(token,full,np.zeros_like(full))),
        ('wrong_state',lambda:cached.linearizations.action(token,full+1e-12,np.zeros_like(full)))]:
        try:operation();rejected[name]=False
        except ValueError:rejected[name]=True
    cached.linearizations.clear()
    try:cached.linearizations.response(token,full);rejected['expired']=False
    except ValueError:rejected['expired']=True
    if not all(rejected.values()):raise ValueError('cache ownership gate failed')
    rng=np.random.default_rng(881);directions=rng.normal(size=(2,*q.shape));directions[:,base.fixed]=0
    directions/=np.linalg.norm(directions.reshape(2,-1),axis=1)[:,None,None]
    checks=[]
    for t in [0.,.5,1.1]:
        for sign in [1.,-1.]:
            d=sign*directions[0];a=base.evaluate(states[t].q,d);b=cached.evaluate(states[t].q,d)
            errors={k:float(np.linalg.norm(np.asarray(a[k])-b[k])/max(np.linalg.norm(a[k]),1e-12)) for k in ['U','force','tangent_action']}
            if max(errors.values())>2e-5:raise ValueError('cached state/direction mismatch')
            checks.append(dict(time=t,direction_sign=sign,errors=errors))
    print('GPU ownership and loading/unloading checks passed',flush=True)
    cpu=CPUStreamingLinearizations(TensorMaterialOperator(base.parent,TensorRule.uniform(base.parent.edges,7)),max_bytes=512<<20,max_directions=2)
    cpu_token=cpu.prepare(full);d_full=np.asarray([base.reduction.velocity(d) for d in directions])
    start=time.perf_counter();actions=cpu.apply_many(cpu_token,full,d_full);cpu_seconds=time.perf_counter()-start
    cpu_errors=[]
    for i,d in enumerate(directions):
        expected=base.evaluate(q,d)['tangent_action'];actual=base.reduction.P.T@actions[i]
        error=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-12));cpu_errors.append(error)
    if max(cpu_errors)>2e-5:raise ValueError('CPU sliced spectral reuse changed full-space tangent')
    try:cpu.apply_many(cpu_token,full+1e-12,d_full);cpu_wrong_state=False
    except ValueError:cpu_wrong_state=True
    if not cpu_wrong_state:raise ValueError('CPU cache accepted wrong state')
    print('CPU two-direction slice seconds',round(cpu_seconds,3),'errors',cpu_errors,flush=True)
    before_state=states[.45].clone();before_state.child_states['identity']=copy.deepcopy(cached.identity)
    stepper=ValidatedAVF(cached,cfg,before_state);before=stepper.state.digest()
    def fail(where,state):
        if where=='after_prepare':raise ValueError('post-prepare cache rollback audit')
    try:stepper.step(.025,inject=fail)
    except StepRejected:pass
    rollback=before==stepper.state.digest();stepper.step(.025)
    control=ValidatedAVF(base,cfg,before_state);control.step(.025)
    difference=max(float(np.max(abs(stepper.state.q-control.state.q))),float(np.max(abs(stepper.state.velocity-control.state.velocity))))
    if not rollback or difference>1e-8:raise ValueError('cache retry changed trajectory')
    evidence=dict(utc=utc(),status='passed_scoped',ownership=rejected,gpu_checks=checks,
        cpu=dict(seconds=cpu_seconds,tangent_relative=cpu_errors,wrong_state_rejected=cpu_wrong_state,**cpu.report()),
        rollback=rollback,retry_difference=difference,gpu_cache=cached.linearizations.report(),
        gpu_memory=cached.operator.memory_budget.report(),
        decision=dict(default_operator='segmented',default_linearization_cache=False,
            GPU_optional='bounded exact cache for repeated tangent workloads',
            CPU_optional='bounded per-slab apply_many utility; ordinary CPU solver remains streaming'),
        timing_scope='CPU batch observation only; no new fair CPU/GPU speed ratio')
    write(run/'N08/completion-result.json',evidence)
    segmented=read(run/'N08/segmented-result.json');cache=read(run/'N08/cache-result.json')
    write(run/'N08/result.json',dict(utc=utc(),status='passed_scoped',evidence=['segmented-result.json','cache-result.json','completion-result.json'],
        decision=evidence['decision'],segmented_two_step_saving=segmented['relative_saving'],
        cache_medians=cache['medians'],full_cycle_integration='pending final N12 matched-source scene',
        limitations=['time accuracy and spatial reference remain limited','memory is sampled with conservative reservation, not continuous process peak']))
    caps=read(run/'capabilities.json');caps['N08']=dict(status='passed_scoped',evidence='N08/result.json',full_cycle_integration=False);write(run/'capabilities.json',caps)
    return evidence


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):complete(a.run)
