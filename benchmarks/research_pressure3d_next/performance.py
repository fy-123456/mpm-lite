"""One bounded multi-range candidate and four serial same-state paired steps."""
import argparse,time,json
import numpy as np
import warp as wp
from .provenance import *
from .fixture import setup,numerical_sources
from .runtime import attempt,update
from .trajectory import case_name,FIELDS
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_restoring_rt0_next.performance import sharing
from benchmarks.research_sequential_next.compare import metric,regions
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from benchmarks.research_pressure_window_next.coupling_study import frame
from benchmarks.research_continuous_geometry_next.continuous import state_checks
from engine.aniso_phase1.research_pressure3d_next.multirange import MultirangeGeometry,install

def initialize(run,kind,which):
    c,m,cfg,ident=setup(run,'yz');item=history(Path(run)/'cases'/case_name('yz'))[(8,16)[which]];c.restore(item['state']);extra=0.
    if kind=='G2':bridge=install(c);extra=bridge['additional_setup_s']
    src=numerical_sources()
    for f in ['benchmarks/research_pressure3d_next/performance.py','engine/aniso_phase1/research_pressure3d_next/multirange.py']:src[f]=sha(ROOT/f)
    ident=json.loads(json.dumps(dict(schema='pressure3d-performance-v1',coupling=c.identity,source_sha256=sha(item['folder']/'state.json'),numerical_sources=src,kind=kind,input=which)))
    return c,m,ident,item,extra

def operators(run):
    run=Path(run);mutable(run);p=read(run/'cases/pressure128-D3/profile-8-18.json');profile=p['geometry'];fraction=profile['local_gradient']['seconds']/max(p['process_s']-p['build_s'],1e-30)
    register(run,'S4/candidate-protocol.json',dict(candidate='G2',count=1,source_profile=p,optimistic_whole_process_gradient_removal_fraction=fraction,implementation='exact transverse support ranges in existing raw CSR transpose; no changed values/P',min_gain=.05,max_spread=.05,max_setup_recovery_steps=16,source_steps=[8,16],order=['D30','G20','G21','D31'],cold_geometry_cache=True,statistical_confidence_claim=False))
    if fraction<.05:raise ValueError('candidate not justified by measured hotspot')
    c,m,ident,item,_=initialize(run,'D3',0);base=c.geometry;fast=MultirangeGeometry.from_block(base);rows=[]
    h=history(run/'cases'/case_name('yz'))
    d=np.zeros_like(item['state'].q);d[m.free]=np.random.default_rng(82).normal(size=d[m.free].shape);d*=1e-6/np.linalg.norm(d)
    for name,q in [('25us',h[8]['state'].q),('50us',h[16]['state'].q),('manufactured',d)]:
        a,b=base.evaluate(q),fast.evaluate(q);checks={k:metric(a[k],b[k],1e-10 if k=='volume' else 1e-8,2e-5) for k in ('volume','gradient','H')};b['gradient'][:]=123
        owned=metric(base.evaluate(q)['gradient'],fast.evaluate(q)['gradient'],1e-8,2e-5);rows.append(dict(name=name,checks=checks,owned=owned,passed=all(v['passed'] for v in [*checks.values(),owned])))
    q0=h[8]['state'].q;q1=h[16]['state'].q;G=fast.discrete(q0,q1);dv=fast.evaluate(q1)['volume']-fast.evaluate(q0)['volume'];work=metric(np.einsum('kij,ij->k',G,q1-q0),dv,1e-12,2e-5)
    passed=all(x['passed'] for x in rows) and work['passed'];write(run/'S4/operator-check.json',dict(status='passed_scoped' if passed else 'failed',records=rows,discrete_volume=work,geometry=fast.identity,additional_setup_s=fast.additional_setup_s,profile=profile))
    if not passed:raise ValueError('G2 operator failed')
    print('G2_OPERATOR',fast.identity['old_used_nnz'],fast.identity['used_nnz'],fast.additional_setup_s,flush=True)

def statistics(f):
    out={}
    for region,w in regions(f['X']).items():
        w=np.asarray(w,dtype=float);w=w/w.sum()
        axes=tuple(range(w.ndim))
        out[region]={k:np.tensordot(w,f[k]-f['X'] if k=='x' else f[k],axes=(axes,axes)).tolist() for k in FIELDS}
    return out

def one(run,kind,which):
    run=Path(run);mutable(run)
    if not sharing()['exclusive']:raise RuntimeError('performance requires exclusive GPU')
    tick=time.perf_counter();c,m,ident,item,extra=initialize(run,kind,which);wp.synchronize_device(m.device);build=time.perf_counter()-tick;folder=run/'S4'/f'{kind}{which}'
    if (folder/'identity.json').exists():raise ValueError('paired trial already exists')
    write(folder/'identity.json',ident);snapshot(folder/'source',ident['numerical_sources']);store=GenerationStore(folder,ident);store.save(c.state,[]);cache=CachedProbes(m);c.geometry.cache.clear();c.geometry.profile={}
    before=sharing();wp.synchronize_device(m.device);t=time.perf_counter();row=attempt(run,'S4',folder.name,lambda:advance_publish(c,store,[]));field=statistics(frame(c,cache,c.state));write(folder/'field-statistics.json',field);wp.synchronize_device(m.device);elapsed=time.perf_counter()-t
    write(folder/'measurement.json',dict(advance_s=elapsed,build_s=build,additional_setup_s=extra,geometry=c.geometry.profile,sharing_before=before,sharing_after=sharing(),no_display_frame=True,physical_probe_reconstruction_included=True,resource=m.operator.memory_budget.report()))
    print('G2_TRIAL',kind,which,elapsed,build,extra,flush=True)

def finish(run):
    run=Path(run);records=[]
    for i in (0,1):
        a=run/'S4'/f'D3{i}';b=run/'S4'/f'G2{i}';ha,hb=history(a),history(b);checks=state_checks(ha[-1]['state'],hb[-1]['state'])
        for k in ('reaction_N','total_energy_J','energy_balance_J','pressure_solid_work_J','pressure_fluid_work_J'):checks[k]=metric(ha[-1]['rows'][-1][k],hb[-1]['rows'][-1][k],1e-8 if k=='reaction_N' else 1e-12,2e-5)
        fa,fb=read(a/'field-statistics.json'),read(b/'field-statistics.json')
        for r in fa:
            for k in fa[r]:checks[r+'/'+k]=metric(fa[r][k],fb[r][k],1e-8,2e-5)
        A,B=read(a/'measurement.json'),read(b/'measurement.json');gain=1-B['advance_s']/A['advance_s'];pay=max(0.,B['build_s']-A['build_s'],B['additional_setup_s'])/max(A['advance_s']-B['advance_s'],1e-30)
        records.append(dict(input=i,D3=A,G2=B,gain=gain,setup_recovery_steps=pay,checks=checks,passed=all(x['passed'] for x in checks.values())))
    gains=[x['gain'] for x in records];median=float(np.median(gains));spread=float(np.ptp(gains));exclusive=all(r[k][s]['exclusive'] for r in records for k in ('D3','G2') for s in ('sharing_before','sharing_after'))
    selected=all(x['passed'] for x in records) and exclusive and min(gains)>=0 and median>=.05 and spread<=.05 and max(x['setup_recovery_steps'] for x in records)<=16
    write(run/'S4/paired-performance.json',dict(status='passed_scoped' if selected else 'limited',records=records,median_gain=median,spread=spread,exclusive=exclusive,selected=selected))
    write(run/'S4/backend-decision.json',dict(status='awaiting_continuous' if selected else 'limited',selected=selected,backend='G2' if selected else 'D3',continuous_qualified=False,reason='four-step continuity required' if selected else 'registered performance or setup gate not met; retain D3'))
    update(run,f'S4 G2：两输入整步降幅{gains}，中位{median:.2%}，设置回收{[x["setup_recovery_steps"] for x in records]}步；入选={selected}。');print('G2_DECISION',selected,gains,flush=True)

def continuous(run):
    run=Path(run);d=read(run/'S4/backend-decision.json')
    if not d['selected']:raise ValueError('G2 did not pass performance gate')
    c,m,ident,item,extra=initialize(run,'G2',0);folder=run/'S4/G2-continuous';store=GenerationStore(folder,ident);write(folder/'identity.json',ident);snapshot(folder/'source',ident['numerical_sources']);store.save(c.state,[]);rows=[];base=history(run/'cases'/case_name('yz'));checks=[]
    for i in range(4):
        row=attempt(run,'S4','G2-continuous',lambda:advance_publish(c,store,rows));rows.append(row);ck=state_checks(c.state,base[c.state.step]['state']);checks.append(dict(step=c.state.step,checks=ck,passed=all(x['passed'] for x in ck.values())))
    passed=all(x['passed'] for x in checks);write(run/'S4/continuous-check.json',dict(status='passed_scoped' if passed else 'failed',records=checks))
    if not passed:raise ValueError('G2 continuous differs')
    write(run/'S4/backend-decision.json',dict(d,status='passed_scoped',continuous_qualified=True,reason='paired performance plus four continuous steps passed; 128-cell scoped research only'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['operators','one','finish','continuous']);p.add_argument('--run',type=Path,required=True);p.add_argument('--kind',choices=['D3','G2'],default='D3');p.add_argument('--which',type=int,default=0);a=p.parse_args()
    with serial_lock(a.run):one(a.run,a.kind,a.which) if a.phase=='one' else globals()[a.phase](a.run)
