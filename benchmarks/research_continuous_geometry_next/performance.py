"""Single G1 candidate: static equivalence then four fair B/G1 steps."""
import argparse,time,resource
import numpy as np
import warp as wp
from .provenance import *
from .fixture import new_setup
from .continuous import sources,state_checks
from .runtime import attempt,update
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_restoring_rt0_next.performance import sharing
from benchmarks.research_sequential_next.compare import metric
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from benchmarks.research_pressure_window_next.coupling_study import frame
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish
from engine.aniso_phase1.research_continuous_geometry_next.geometry import BatchedGeometry,install

def setup(run,which=0,kind='B'):
    item=history(APP/'cases/boundary32-h')[(8,16)[which]];c,m,cfg,bridge=new_setup(run,state=item['state'],shared=True);extra=0.
    if kind=='G1':
        t=time.perf_counter();bridge=install(c,bridge);wp.synchronize_device(m.device);extra=time.perf_counter()-t
    src=sources()
    for f in ('benchmarks/research_continuous_geometry_next/performance.py','engine/aniso_phase1/research_continuous_geometry_next/geometry.py'):src[f]=sha(ROOT/f)
    ident=dict(schema='continuous-G1-performance-v1',coupling=c.identity,source_sha256=sha(item['folder']/'state.json'),numerical_sources=src,kind=kind,input=which)
    return c,m,ident,bridge,item,extra

def operators(run):
    run=Path(run);mutable(run)
    if read(run/'S2/optimization-protocol.json')['status']!='registered':raise ValueError('candidate not triggered')
    c,m,ident,bridge,item,_=setup(run);base=c.geometry;g=BatchedGeometry.from_shared(base);rest=m.rest().q;direction=np.zeros_like(rest);direction[m.free]=np.random.default_rng(5).normal(size=direction[m.free].shape);field=base.field(direction,True).numpy();direction*=.02/max(float(np.max(abs(field))),1e-30);del field;manufactured=rest+direction;states=[('rest',rest),('manufactured',manufactured),('25us',item['state'].q),('50us',history(APP/'cases/boundary32-h')[16]['state'].q)];records=[]
    for label,q in states:
        base.cache.clear();g.cache.clear();a=base.evaluate(q);b=g.evaluate(q);checks={k:metric(a[k],b[k],1e-8,2e-5) for k in ('volume','gradient','H')};b['gradient'][:]=0;owned=metric(g.evaluate(q)['gradient'],a['gradient'],1e-8,2e-5);records.append(dict(state=label,checks=checks,owned=owned,passed=all(x['passed'] for x in [*checks.values(),owned])))
    paths=[]
    for label,q0,q1 in [('actual',states[2][1],states[3][1]),('manufactured',rest,manufactured),('reverse',manufactured,rest)]:
        a=base.discrete(q0,q1);b=g.discrete(q0,q1);dv=g.evaluate(q1)['volume']-g.evaluate(q0)['volume'];work=float(np.max(abs(np.einsum('kij,ij->k',b,q1-q0)-dv)));check=metric(a,b,1e-8,2e-5);paths.append(dict(path=label,gradient=check,volume_work_defect=work,passed=check['passed'] and work<1e-10))
    passed=all(x['passed'] for x in records+paths);write(run/'S2/operator-check.json',dict(status='passed_scoped' if passed else 'limited',records=records,paths=paths,layout=g.identity,guard=m.operator.memory_budget.report(),source_sha256=all_sources()))
    if not passed:raise ValueError('G1 operator equivalence failed')
    register(run,'S2/integration-protocol.json',dict(order=['B0','G1_0','G1_1','B1'],sources=[read(run/'S0/state-contract.json')['states'][str(i)] for i in (8,16)],same_instrumentation=True,minimum_gain=.05,maximum_setup_recovery_steps=16,cold_case_caches=True));print('G1_OPERATOR passed',flush=True)

def one(run,which,kind):
    run=Path(run);mutable(run);start=time.perf_counter()
    if not sharing()['exclusive']:raise RuntimeError('shared GPU: no performance qualification')
    c,m,ident,bridge,item,extra=setup(run,which,kind);wp.synchronize_device(m.device);build=time.perf_counter()-start;folder=run/'S2'/f'{kind}{which}'
    if (folder/'identity.json').exists():raise ValueError('performance branch already exists')
    write(folder/'identity.json',ident);snapshot(folder/'source',ident['numerical_sources']);write(folder/'bridge.json',bridge);store=GenerationStore(folder,ident);store.save(c.state,[]);cache=CachedProbes(m);c.geometry.cache.clear();parts={k:0. for k in ('geometry_s','rt0_s','material_s','commit_s','field_s')};calls={k:0 for k in parts}
    def timed(obj,method,key):
        original=getattr(obj,method)
        def wrap(*a,**kw):
            wp.synchronize_device(m.device);t=time.perf_counter()
            try:return original(*a,**kw)
            finally:wp.synchronize_device(m.device);parts[key]+=time.perf_counter()-t;calls[key]+=1
        setattr(obj,method,wrap)
    timed(c.geometry,'evaluate','geometry_s');timed(c.geometry.assembler,'assemble','rt0_s');timed(m,'evaluate','material_s');timed(store,'save','commit_s')
    def emit(s):
        t=time.perf_counter()
        try:return frame(c,cache,s)
        finally:parts['field_s']+=time.perf_counter()-t
    before=sharing()
    if not before['exclusive']:raise RuntimeError('GPU became shared')
    wp.synchronize_device(m.device);t=time.perf_counter();row=attempt(run,'S2',f'{kind}{which}',lambda:advance_publish(c,store,[],frame_builder=emit if which==1 else None));wp.synchronize_device(m.device);elapsed=time.perf_counter()-t
    write(folder/'measurement.json',dict(advance_s=elapsed,build_s=build,additional_setup_s=extra,total_process_s=time.perf_counter()-start,parts=parts,calls=calls,nested_times_not_additive=True,sharing_before=before,sharing_after=sharing(),peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,resources=m.operator.memory_budget.report()));print('G1_PAIR',kind,which,elapsed,parts,flush=True)

def finish(run):
    run=Path(run);mutable(run);records=[]
    for i in (0,1):
        aa=history(run/'S2'/f'B{i}');bb=history(run/'S2'/f'G1{i}');checks=state_checks(aa[-1]['state'],bb[-1]['state']);ra=aa[-1]['rows'][-1];rb=bb[-1]['rows'][-1]
        for k in ('total_energy_J','darcy_dissipation_J','numerical_dissipation_J','pressure_solid_work_J','pressure_fluid_work_J','energy_balance_J','reservoir_work_J'):checks[k]=metric(ra[k],rb[k],1e-12,2e-5)
        if i==1:
            with np.load(next((run/'S2/B1').rglob('frame.npz'))) as a,np.load(next((run/'S2/G11').rglob('frame.npz'))) as b:checks.update({f'frame_{k}':metric(a[k],b[k],1e-10 if k=='face_flux_m3_s' else 1e-8,2e-5) for k in a.files})
        A=read(run/'S2'/f'B{i}/measurement.json');B=read(run/'S2'/f'G1{i}/measurement.json');gain=1-B['advance_s']/A['advance_s'];records.append(dict(input=i,B=A,G1=B,gain=gain,checks=checks,passed=all(v['passed'] for v in checks.values()),setup_recovery_steps=max(0.,B['build_s']-A['build_s'],B['additional_setup_s'])/max(A['advance_s']-B['advance_s'],1e-30)))
    gains=[r['gain'] for r in records];median=float(np.median(gains));spread=float(np.ptp(gains));exclusive=all(r[k][s]['exclusive'] for r in records for k in ('B','G1') for s in ('sharing_before','sharing_after'));selected=all(r['passed'] for r in records) and exclusive and min(gains)>=0 and median>=max(.05,spread) and max(r['setup_recovery_steps'] for r in records)<=16
    write(run/'S2/paired-performance.json',dict(status='passed_scoped' if selected else 'limited',records=records,median_gain=median,between_state_gain_range=spread,exclusive=exclusive,selected=selected,statistical_confidence_claim=False));write(run/'S2/backend-decision.json',dict(status='awaiting_continuous' if selected else 'limited',selected=selected,backend='G1' if selected else 'B',reason='performance passed; requires G1 continuous gate' if selected else 'single G1 candidate did not meet the registered whole-step performance gate',new_dynamic_attempts=4,new_exclusive_performance=selected,continuous_G1_qualified=False));update(run,f'S2 G1实测两输入降幅{gains}，中位{median:.2%}，极差{spread:.2%}；入选性能阶段={selected}。');print('G1_DECISION',selected,gains,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['operators','one','finish']);p.add_argument('--run',type=Path,required=True);p.add_argument('--which',type=int,choices=[0,1],default=0);p.add_argument('--kind',choices=['B','G1'],default='B');a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='operators':operators(a.run)
        elif a.phase=='finish':finish(a.run)
        else:one(a.run,a.which,a.kind)
