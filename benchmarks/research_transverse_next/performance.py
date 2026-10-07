"""One registered candidate; four independent-process single steps, no new frames."""
import argparse,time
import numpy as np
import warp as wp
from .provenance import *
from .fixture import setup
from .trajectory import bind,equivalence
from .runtime import attempt,update
from .observables import spatial_statistics
from engine.aniso_phase1.research_transverse_next.device_volume import DeviceVolumeGeometry
from engine.aniso_phase1.research_transverse_next.recovery import SafePublication
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_pressure_window_next.coupling_study import frame
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric,regions

def candidate(g):return DeviceVolumeGeometry.adopt(g)

def sources(ident):
    ident['numerical_sources'].update({k:v for k,v in all_sources().items() if Path(k).name in ('device_volume.py','performance.py')})
    return ident

def static(run):
    run=Path(run);mutable(run);tick=time.perf_counter()
    if read(run/'S4/candidate-protocol.json')['status']!='registered':raise ValueError('candidate not registered')
    c,m,cfg,ident=setup(run,'YZ128',profile=False);sources(ident);g=c.geometry;other=candidate(g);h=history(run/'cases/YZ128');q0=h[8]['state'].q;q1=h[9]['state'].q
    qs=[q0,q1,.5*(q0+q1),h[16]['state'].q];out=[];values=[]
    for i,q in enumerate(qs):
        a=g.evaluate(q);b=other.evaluate(q);z=np.random.default_rng(927+i).normal(size=g.topology.nflux);z*=.2/np.linalg.norm(a['H']@z)
        checks={key:metric(b[key],a[key],1e-12 if key=='volume' else 1e-8,2e-5) for key in ('volume','gradient')}
        checks['H_action']=metric(b['H']@z,a['H']@z,1e-6,2e-5);checks['minJ']=metric(b['min_detF'],a['min_detF'],1e-12,2e-5)
        out.append(dict(state=i,checks=checks,passed=all(x['passed'] for x in checks.values())));values.append(b)
    d=q1-q0;bar=(values[0]['gradient']+4*values[2]['gradient']+values[1]['gradient'])/6;dv=values[1]['volume']-values[0]['volume'];p=np.array(h[8]['state'].child_states['fluid']['pressure_Pa'])
    chain=metric(np.einsum('kij,ij->k',bar,d),dv,1e-12,2e-5)
    pressure=metric(.8*float(p@dv),.8*float(np.einsum('k,kij,ij',p,bar,d)),1e-12,2e-5)
    a=other.evaluate(q0);a['volume'][:]=99;a['gradient'][:]=99;owned=np.array_equal(other.evaluate(q0)['volume'],values[0]['volume']) and np.array_equal(other.evaluate(q0)['gradient'],values[0]['gradient'])
    # A different metadata owner must fail before even returning a cache hit.
    old=other.assembler;other.assembler=object()
    try:other.evaluate(q0)
    except ValueError:invalidated=True
    else:raise ValueError('metadata replacement accepted')
    finally:other.assembler=old
    if not all(x['passed'] for x in out) or not chain['passed'] or not pressure['passed'] or not owned:raise ValueError('device volume operator differs')
    folder=run/'S4/static';bind(folder,ident)
    write(run/'S4/operator-check.json',dict(status='passed_scoped',records=out,static_state_groups=4,Simpson_chain=chain,pressure_work=pressure,owned_returns=owned,metadata_owner_invalidation=invalidated,distinct_q_entries=len(other.cache),implementation=other.implementation,additional_setup_s=other.additional_setup_s,seconds=time.perf_counter()-tick))
    update(run,'S4唯一候选算子通过：GPU单元体积汇总在四个真实状态上与D3一致；完整G/H/P不变，压力功、Simpson体积链、缓存所有权和元数据所有者失效检查通过。')

def one(run,kind,index):
    run=Path(run);mutable(run);tick=time.perf_counter();c,m,cfg,ident=setup(run,'YZ128',profile=False);built=time.perf_counter()-tick
    if read(run/'S4/operator-check.json')['status']!='passed_scoped':raise ValueError('candidate operator not qualified')
    extra=0.
    if kind=='DV':c.core.geometry=candidate(c.geometry);extra=c.geometry.additional_setup_s
    h=history(run/'cases/YZ128');step=(8,16)[index];a,b=h[step],h[step+1];c.restore(a['state']);sources(ident);ident.update(performance_backend=kind,input_state_sha256=sha(a['folder']/'state.json'),input_digest=a['state'].digest())
    folder=run/'S4'/f'{kind}{index}';bind(folder,ident);store=GenerationStore(folder,ident);store.save(c.state,[]);safe=SafePublication(c,store);cache=CachedProbes(m)
    # Both backends start with an empty q cache. Existing shared JIT cache is
    # declared; model construction and candidate installation remain recorded.
    c.geometry.cache.clear();c.geometry.profile={};wp.synchronize_device(m.operator.device);started=time.perf_counter()
    row=attempt(run,'S4',folder.name,safe.advance);f=frame(c,cache,c.state);w=regions(f['X'])['global_domain'];statistics={k:spatial_statistics(f['X'],f[k]-f['X'] if k=='x' else f[k],w) for k in ('x','velocity','PK1','PK1_total','Cauchy_skeleton','Cauchy_total')}
    write(folder/'field-statistics.json',statistics);wp.synchronize_device(m.operator.device);elapsed=time.perf_counter()-started
    checks=equivalence(c.state,b['state'],row,b['rows'][-1]);passed=all(v['passed'] for v in checks.values())
    write(folder/'measurement.json',dict(status='passed_scoped' if passed else 'failed',kind=kind,input_step=step,build_s=built,additional_setup_s=extra,total_setup_s=built+extra,advance_s=elapsed,includes_probe_and_IO=True,checks=checks,geometry=c.geometry.profile,publication=safe.profile,geometry_cache_cold=True,JIT='existing run cache; initial compilation recorded in static process',new_display_frames=0,statistics=statistics,seconds=time.perf_counter()-tick))
    if not passed:raise ValueError('candidate paired step not equivalent')
    print('PAIR',kind,index,elapsed,'setup',built+extra,flush=True)

def review(run):
    run=Path(run);mutable(run);out=[];gains=[]
    for index in (0,1):
        a=read(run/f'S4/D3{index}/measurement.json');b=read(run/f'S4/DV{index}/measurement.json')
        checks={}
        for key in a['statistics']:
            tol=1e-8 if key in ('x','velocity') else 1e-6
            checks[key]={k:metric(a['statistics'][key][k],b['statistics'][key][k],tol,2e-5) for k in ('mean','rms','max_abs')}
        physical=a['status']==b['status']=='passed_scoped' and all(v['passed'] for row in checks.values() for v in row.values())
        gain=1-b['advance_s']/a['advance_s'];saving=a['advance_s']-b['advance_s'];extra=max(0.,b['total_setup_s']-a['total_setup_s']);recovery=extra/saving if saving>0 else None;gains.append(gain)
        out.append(dict(input_step=a['input_step'],D3=a,DV=b,gain_fraction=gain,field_equivalence=checks,physical_passed=physical,additional_total_setup_s=extra,setup_recovery_steps=recovery))
    jobs=[read(run/'processes'/f'pair-{k}{i}.json') for k,i in [('D3',0),('DV',0),('DV',1),('D3',1)]]
    isolated=all(len(j[key].strip().splitlines())==1 for j in jobs for key in ('gpu_before','gpu_after'))
    selected=all(v['physical_passed'] for v in out) and min(gains)>=0 and float(np.median(gains))>=.05 and max(gains)-min(gains)<=.05 and all(v['setup_recovery_steps'] is not None and v['setup_recovery_steps']<=16 for v in out) and isolated
    write(run/'S4/paired-performance.json',dict(status='passed_scoped' if all(v['physical_passed'] for v in out) else 'failed',records=out,median_gain_fraction=float(np.median(gains)),gain_spread=max(gains)-min(gains),GPU_isolated=isolated,selected=selected,order=['D30','DV0','DV1','D31'],actual_single_steps=4,no_full_window_timing_claim=True))
    write(run/'S4/backend-decision.json',dict(status='candidate_requires_continuity' if selected else 'retained',backend='D3 until continuity' if selected else 'D3',selected=selected,reason='paired algebra and all registered speed/setup gates passed; continuity pending' if selected else 'one or more registered whole-step gain/spread/setup gates not met; retain D3',gains=gains,median_gain=float(np.median(gains))))
    if not selected:write(run/'S4/continuous-check.json',dict(status='not_triggered',reason='candidate not selected by paired gate',new_steps=0))
    update(run,f'S4公平四单步完成：两输入收益{gains}，中位{np.median(gains):.2%}；'+('通过选取门槛，待4步连续检查。' if selected else '未满足全部选取条件，保留D3，不追加连续试验。'))
    print('PERFORMANCE_DECISION',selected,gains,flush=True)

def continuous(run):
    run=Path(run);mutable(run)
    if not read(run/'S4/backend-decision.json')['selected']:raise ValueError('candidate not selected')
    c,m,cfg,ident=setup(run,'YZ128',profile=False);c.core.geometry=candidate(c.geometry);h=history(run/'cases/YZ128');c.restore(h[8]['state']);sources(ident);ident.update(performance_backend='DV',input_digest=c.state.digest());folder=run/'S4/continuous';bind(folder,ident);store=GenerationStore(folder,ident);store.save(c.state,[]);safe=SafePublication(c,store);out=[]
    for i in range(9,13):
        row=attempt(run,'S4','DV-continuous',safe.advance);checks=equivalence(c.state,h[i]['state'],row,h[i]['rows'][-1]);out.append(dict(step=i,checks=checks,passed=all(v['passed'] for v in checks.values())))
        if not out[-1]['passed']:break
    from benchmarks.research_continuous_geometry_next.review import balances
    bal=balances(store.history());passed=len(out)==4 and all(v['passed'] for v in out) and bal['passed']
    write(run/'S4/continuous-check.json',dict(status='passed_scoped' if passed else 'limited',records=out,balances=bal,new_steps=len(out),owned_cache_limit=6,cache_entries=len(c.geometry.cache)))
    d=read(run/'S4/backend-decision.json');d.update(status='selected_scoped' if passed else 'retained',selected=passed,backend='DV: device-cell-volume, specified research window' if passed else 'D3',reason='four continuous equivalent steps and physical ledgers passed' if passed else 'continuity failed; retain D3');write(run/'S4/backend-decision.json',d)
    update(run,'S4连续验证：'+d['reason']+'。')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['static','one','review','continuous']);p.add_argument('--run',type=Path,required=True);p.add_argument('--kind',choices=['D3','DV'],default='D3');p.add_argument('--index',type=int,choices=[0,1],default=0);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='static':static(a.run)
        elif a.phase=='one':one(a.run,a.kind,a.index)
        elif a.phase=='review':review(a.run)
        else:continuous(a.run)
