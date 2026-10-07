"""N08b: exact cached directions, rejected trials and separate warm costs."""
from pathlib import Path
import argparse
import copy
import time
import numpy as np
from .provenance import read,write,serial_lock,source_files,utc
from .run import load_model
from .checkpoint import GenerationStore
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_sequential_next.linearization import CachedModel,LinearizationCache
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF,StepRejected


def cache_study(run):
    run=Path(run);cfg=read(run/'cases/gpu-q7-dt0025/execution-protocol.json');base,_=load_model(run,cfg)
    plain=SegmentedModel(base.reduction,order=7,device='cuda:0')
    cached=CachedModel(base.reduction,order=7,device='cuda:0',cache_bytes=3<<30,cache_entries=4)
    folder=run/'cases/gpu-q7-dt0025';history=GenerationStore(folder,read(folder/'identity.json')).history()
    states={round(x['state'].time,10):x['state'] for x in history}
    protocol=dict(utc=utc(),source_sha256=source_files(),states=[0.,.5,1.1],directions=4,
        cache_bytes=3<<30,cache_entries=4,warm_repeats=3,random_seed=817,
        comparisons='uncached segmented versus cached segmented; CSR speed is not counted as cache benefit')
    if (run/'N08/cache-protocol.json').exists():raise ValueError('cache study already registered')
    write(run/'N08/cache-protocol.json',protocol)
    rng=np.random.default_rng(817);directions=[rng.normal(size=states[0.].q.shape) for _ in range(4)]
    for d in directions:d[plain.fixed]=0.;d/=np.linalg.norm(d)
    checks=[]
    for t in protocol['states']:
        for d in directions[:2]:
            a=plain.evaluate(states[t].q,d);b=cached.evaluate(states[t].q,d)
            errors={k:float(np.linalg.norm(np.asarray(a[k])-b[k])/max(np.linalg.norm(a[k]),1e-12)) for k in ['U','force','tangent_action']}
            if max(errors.values())>2e-5:raise ValueError('cache changed exact state response')
            checks.append(dict(time=t,errors=errors))
    q=states[.5].q;full=cached.reduction.expand(q);token,_=cached.linearizations.prepare(full)
    foreign=LinearizationCache(plain.operator,max_bytes=3<<30,max_entries=1)
    rejected={}
    for name,action in [('wrong_state',lambda:cached.linearizations.action(token,full+1e-8,np.zeros_like(full))),
        ('foreign_cache',lambda:foreign.action(token,full,np.zeros_like(full)))]:
        try:action();rejected[name]=False
        except ValueError:rejected[name]=True
    if not all(rejected.values()):raise ValueError('invalid cache handle was accepted')
    # Public response ownership: even deliberately removing write protection
    # from the returned copy must leave the private entry unchanged.
    response=cached.evaluate(q);expected=response['force'].copy()
    response['force'].setflags(write=True);response['force'][:]=123.
    if not np.array_equal(cached.evaluate(q)['force'],expected):raise ValueError('caller poisoned material cache')
    initial=states[.45].clone();initial.child_states['identity']=copy.deepcopy(cached.identity)
    integrator=ValidatedAVF(cached,cfg,initial);before=integrator.state.digest()
    def fail(where,state):
        if where=='after_prepare':raise ValueError('intentional cache-after-trial rollback')
    try:integrator.step(.025,inject=fail)
    except StepRejected:pass
    rollback=integrator.state.digest()==before
    integrator.step(.025);retry=integrator.state
    control=ValidatedAVF(plain,cfg,initial);control.step(.025)
    retry_difference=float(np.max(abs(control.state.q-retry.q)))
    if not rollback or retry_difference>1e-8:raise ValueError('cached rejected state contaminated retry')
    samples=[]
    for repeat in range(3):
        for name,model in [('uncached',plain),('cached',cached)]:
            if name=='cached':model.linearizations.clear()
            begun=time.perf_counter()
            for d in directions:model.evaluate(q,d)
            elapsed=time.perf_counter()-begun
            samples.append(dict(kind='four_same_state_directions',repeat=repeat,variant=name,seconds=elapsed))
            print('directions',name,round(elapsed,5),flush=True)
        for name,model in [('uncached',plain),('cached',cached)]:
            model._cached_q=None;model._cached_response=None
            if name=='cached':model.linearizations.clear()
            initial=states[.45].clone();initial.child_states['identity']=copy.deepcopy(model.identity)
            begun=time.perf_counter();stepper=ValidatedAVF(model,cfg,initial)
            rows=[stepper.step(.025),stepper.step(.025)];elapsed=time.perf_counter()-begun
            samples.append(dict(kind='two_steps',repeat=repeat,variant=name,seconds=elapsed,
                krylov_iterations=sum(r['krylov_iterations'] for r in rows)))
            print('two steps',name,round(elapsed,5),flush=True)
    medians={kind:{name:float(np.median([r['seconds'] for r in samples if r['kind']==kind and r['variant']==name]))
        for name in ('uncached','cached')} for kind in ('four_same_state_directions','two_steps')}
    result=dict(utc=utc(),status='passed_scoped',checks=checks,wrong_state_rejected=rejected,
        caller_copy_isolated=True,rollback=rollback,retry_q_max_difference=retry_difference,samples=samples,
        medians=medians,cache=cached.linearizations.report(),
        policy='Use bounded cache for repeated tangent workloads; ordinary cycle adoption depends on measured total cost and memory.',
        point_counter_semantics='prepared_points and tangent_points are separate; a cached direction still evaluates an exact tangent')
    write(run/'N08/cache-result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):cache_study(a.run)
