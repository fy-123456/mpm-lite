"""One DV-relative candidate, bounded operator and whole-step qualification."""
import argparse,time
import numpy as np
import warp as wp
from .provenance import *
from .runtime import attempt,update
from .fixture import setup
from .coupling import bind,field_stats,start_store
from engine.aniso_phase1.research_transverse_reference_next.batch_download import BatchDownloadGeometry
from engine.aniso_phase1.research_transverse_next.recovery import SafePublication
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_transverse_next.trajectory import equivalence
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric
from benchmarks.research_pressure_window_next.coupling_study import frame
from benchmarks.research_continuous_geometry_next.review import balances


def source_identity(ident,kind):
    ident['numerical_sources'].update({k:v for k,v in all_sources().items() if Path(k).name in ('batch_download.py','performance.py')})
    ident['performance_backend']=kind
    return ident


def static(run):
    run=Path(run);mutable(run)
    if read(run/'S3/candidate-protocol.json')['status']!='registered':raise ValueError('candidate not registered')
    c,m,cfg,ident=setup(run);g=c.geometry;other=BatchDownloadGeometry.adopt(g);h=history(APP/'cases/YZ128');q0=h[8]['state'].q;q1=h[9]['state'].q;qs=[q0,q1,.5*(q0+q1),h[16]['state'].q]
    # Diagnostic-only synchronization splits the original combined local segment.
    timers={'adjoint_s':0.,'calls':0};originals=[]
    for mp in {id(x[2]):x[2] for x in g.local}.values():
        original=mp.gradient_adjoint;originals.append((mp,original))
        def timed(*args,_original=original,**kwargs):
            wp.synchronize_device(g.op.device);tick=time.perf_counter();v=_original(*args,**kwargs);wp.synchronize_device(g.op.device);timers['adjoint_s']+=time.perf_counter()-tick;timers['calls']+=1;return v
        mp.gradient_adjoint=timed
    tick=time.perf_counter();first=g.evaluate(q0);diagnostic_elapsed=time.perf_counter()-tick
    for mp,original in originals:mp.gradient_adjoint=original
    diagnostic=dict(parent_adjoint_s=timers['adjoint_s'],parent_adjoint_calls=timers['calls'],geometry=g.profile.copy(),elapsed_s=diagnostic_elapsed,unseparated='remaining local time is gather/download/P/host allocations; no exact per-kernel attribution',extra_dynamic_steps=0)
    g.cache.clear();g.profile={};out=[];vals=[]
    for i,q in enumerate(qs):
        a=g.evaluate(q);b=other.evaluate(q);z=np.random.default_rng(117+i).normal(size=g.topology.nflux);z*=.2/np.linalg.norm(a['H']@z)
        checks={k:metric(b[k],a[k],1e-12 if k=='volume' else 1e-8,2e-5) for k in ('volume','gradient')}
        checks['H_action']=metric(b['H']@z,a['H']@z,1e-6,2e-5);checks['minJ']=metric(b['min_detF'],a['min_detF'],1e-12,2e-5)
        vals.append(b);out.append(dict(state=i,checks=checks,passed=all(x['passed'] for x in checks.values())))
    dq=q1-q0;bar=(vals[0]['gradient']+4*vals[2]['gradient']+vals[1]['gradient'])/6;dv=vals[1]['volume']-vals[0]['volume'];p=np.array(h[8]['state'].child_states['fluid']['pressure_Pa'])
    chain=metric(np.einsum('cij,ij->c',bar,dq),dv,1e-12,2e-5)
    pressure=metric(.8*float(p@dv),.8*float(np.einsum('c,cij,ij',p,bar,dq)),1e-12,2e-5)
    owned=other.evaluate(q0);owned['gradient'][:]=99;ownership=np.array_equal(other.evaluate(q0)['gradient'],vals[0]['gradient'])
    saved=other.assembler;other.assembler=object()
    try:other.evaluate(q0)
    except ValueError:rejected=True
    else:rejected=False
    finally:other.assembler=saved
    other.cache.clear();orig=other.local[0][2].gradient_adjoint
    def fail(*args,**kwargs):raise MemoryError('registered local transient failure')
    other.local[0][2].gradient_adjoint=fail
    try:other.evaluate(q0)
    except MemoryError:failed_clean=not other.cache
    else:failed_clean=False
    finally:other.local[0][2].gradient_adjoint=orig
    rebuilt=other.evaluate(q0);retry=metric(rebuilt['gradient'],vals[0]['gradient'],1e-8,2e-5)
    passed=all(x['passed'] for x in out) and chain['passed'] and pressure['passed'] and ownership and rejected and failed_clean and retry['passed']
    bind(run/'S3/static',source_identity(ident,'BD'))
    write(run/'S3/DV-adjoint-profile.json',diagnostic)
    write(run/'S3/operator-check.json',dict(status='passed_scoped' if passed else 'failed',records=out,static_state_groups=4,Simpson_chain=chain,pressure_work=pressure,implementation=other.implementation))
    write(run/'S3/buffer-lifecycle.json',dict(status='passed_scoped' if passed else 'failed',owned_returns=ownership,metadata_owner_rejected=rejected,failed_call_no_cache=failed_clean,retry=retry,persistent_staging=False,extra_dynamic_fault_steps=0,scope='call-local staging failure and rebuild; unchanged transaction owner'))
    if not passed:raise ValueError('batched download operator failed')
    update(run,'S3.3：唯一批量下载候选在4个真实几何状态上通过V/G/H、压力功与Simpson链检查；返回所有权、元数据失效、局部临时失败后重建通过。')
    print('STATIC',passed,diagnostic,flush=True)


def one(run,kind,index):
    run=Path(run);mutable(run)
    if read(run/'S3/operator-check.json')['status']!='passed_scoped':raise ValueError('operator not qualified')
    tick=time.perf_counter();c,m,cfg,ident=setup(run);base_build=time.perf_counter()-tick
    if kind=='BD':c.core.geometry=BatchDownloadGeometry.adopt(c.geometry)
    extra=c.geometry.additional_setup_s if kind=='BD' else 0.
    h=history(APP/'cases/YZ128');step=(8,16)[index];a,b=h[step],h[step+1];ident=source_identity(ident,kind);ident['input_digest']=a['state'].digest()
    folder,store=start_store(run,f'S3/{kind}{index}',c,ident,a);safe=SafePublication(c,store);cache=CachedProbes(m);c.geometry.cache.clear();c.geometry.profile={}
    wp.synchronize_device(m.operator.device);tick=time.perf_counter();row=attempt(run,'S3',folder.name,safe.advance);f=frame(c,cache,c.state);stats=field_stats(f,m.parent.params.fiber_direction);write(folder/'field-statistics.json',stats);wp.synchronize_device(m.operator.device);elapsed=time.perf_counter()-tick
    checks=equivalence(c.state,b['state'],row,b['rows'][-1]);passed=all(x['passed'] for x in checks.values()) and balances(store.history())['passed']
    write(folder/'measurement.json',dict(status='passed_scoped' if passed else 'failed',kind=kind,input_step=step,advance_s=elapsed,build_s=base_build,additional_setup_s=extra,total_setup_s=base_build+extra,includes_probe_and_IO=True,checks=checks,geometry=c.geometry.profile,publication=safe.profile,cache_cold=True,statistics=stats))
    if not passed:raise ValueError('paired step not equivalent')
    print('PAIR',kind,index,elapsed,flush=True)


def review(run):
    run=Path(run);mutable(run);records=[];gains=[]
    for i in (0,1):
        a=read(run/f'S3/DV{i}/measurement.json');b=read(run/f'S3/BD{i}/measurement.json');checks={}
        for region in a['statistics']:
            for key in a['statistics'][region]:
                for stat in ('mean','rms','max_abs'):
                    checks[f'{region}/{key}/{stat}']=metric(a['statistics'][region][key][stat],b['statistics'][region][key][stat],1e-8 if key in ('x','velocity') else 1e-6,2e-5)
        gain=1-b['advance_s']/a['advance_s'];gains.append(gain);saving=a['advance_s']-b['advance_s'];extra=max(0.,b['total_setup_s']-a['total_setup_s']);recovery=extra/saving if saving>0 else None
        records.append(dict(input_step=a['input_step'],DV=a['advance_s'],BD=b['advance_s'],gain_fraction=gain,setup_recovery_steps=recovery,fields=checks,passed=a['status']==b['status']=='passed_scoped' and all(x['passed'] for x in checks.values())))
    jobs=[read(run/f'processes/pair-{k}{i}.json') for k,i in [('DV',0),('BD',0),('BD',1),('DV',1)]]
    isolated=all(not j['gpu_before'].strip() and not j['gpu_after'].strip() for j in jobs)
    selected=all(x['passed'] for x in records) and min(gains)>=0 and np.median(gains)>=.05 and max(gains)-min(gains)<=.05 and all(x['setup_recovery_steps'] is not None and x['setup_recovery_steps']<=16 for x in records) and isolated
    write(run/'S3/paired-performance.json',dict(status='passed_scoped' if all(x['passed'] for x in records) else 'failed',records=records,median_gain_fraction=float(np.median(gains)),isolated=isolated,selected=bool(selected),no_full_cycle_speed_claim=True))
    write(run/'S3/backend-decision.json',dict(status='requires_continuity' if selected else 'retained',selected=False,candidate_eligible=bool(selected),backend='DV',gains=gains,reason='four-step continuity pending' if selected else 'paired speed/spread/setup/isolation gate not met; retain DV'))
    if not selected:write(run/'S3/continuous-check.json',dict(status='not_triggered',new_steps=0,reason='candidate did not meet paired gate'))
    update(run,f'S3.4：DV相对唯一候选的两组单步降幅为{gains}，中位{np.median(gains):.2%}；'+('进入4步连续验证。' if selected else '未满足全部性能门槛，保留DV。'))
    print('DECISION',selected,gains,flush=True)


def continuous(run):
    run=Path(run);mutable(run)
    if not read(run/'S3/backend-decision.json')['candidate_eligible']:raise ValueError('candidate not eligible')
    c,m,cfg,ident=setup(run);c.core.geometry=BatchDownloadGeometry.adopt(c.geometry);h=history(APP/'cases/YZ128');folder,store=start_store(run,'S3/continuous',c,source_identity(ident,'BD'),h[8]);safe=SafePublication(c,store);records=[]
    for i in range(9,13):
        row=attempt(run,'S3','BD-continuous',safe.advance);checks=equivalence(c.state,h[i]['state'],row,h[i]['rows'][-1]);records.append(dict(step=i,checks=checks,passed=all(x['passed'] for x in checks.values())))
        if not records[-1]['passed']:break
    bal=balances(store.history());passed=len(records)==4 and all(x['passed'] for x in records) and bal['passed']
    write(run/'S3/continuous-check.json',dict(status='passed_scoped' if passed else 'limited',records=records,balances=bal,new_steps=len(records)))
    d=read(run/'S3/backend-decision.json');d.update(status='selected_scoped' if passed else 'retained',selected=passed,backend='BD: batched gradient download, specified research window' if passed else 'DV',reason='operator, paired steps and four continuous steps passed' if passed else 'continuity failed');write(run/'S3/backend-decision.json',d)
    update(run,'S3连续验证：'+d['reason'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['static','one','review','continuous']);p.add_argument('--run',type=Path,required=True);p.add_argument('--kind',choices=['DV','BD'],default='DV');p.add_argument('--index',type=int,choices=[0,1],default=0);a=p.parse_args()
    if a.phase=='static':static(a.run)
    elif a.phase=='one':one(a.run,a.kind,a.index)
    elif a.phase=='review':review(a.run)
    else:continuous(a.run)
