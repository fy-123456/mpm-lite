"""One algebraic candidate, certified against current BD, never against old DV."""
import argparse,time
import numpy as np
import warp as wp
from .provenance import *
from .fixture import setup,inputs
from .runtime import attempt,update
from .dynamics import start_store,field_stats
from engine.aniso_phase1.research_coupled_reference_next.factored import FactoredGradientGeometry
from engine.aniso_phase1.research_transverse_next.recovery import SafePublication
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from benchmarks.research_transverse_next.trajectory import equivalence
from benchmarks.research_sequential_next.compare import metric
from benchmarks.research_pressure_window_next.coupling_study import frame
from benchmarks.research_continuous_geometry_next.review import balances

def static(run):
    run=Path(run);mutable(run)
    if read(run/'S3/candidate-protocol.json')['status']!='registered':raise ValueError('candidate unregistered')
    c,m,cfg,ident=setup(run);g=c.geometry;candidate=FactoredGradientGeometry.adopt(g);h=inputs();q0=h[8]['state'].q;q1=h[9]['state'].q;qs=[q0,q1,.5*(q0+q1),h[16]['state'].q];out=[];vals=[]
    for i,q in enumerate(qs):
        a=g.evaluate(q);b=candidate.evaluate(q);z=np.random.default_rng(503+i).normal(size=g.topology.nflux);z*=.2/np.linalg.norm(a['H']@z)
        checks={k:metric(b[k],a[k],1e-12 if k=='volume' else 1e-8,2e-5) for k in ('volume','gradient')}
        checks['H_action']=metric(b['H']@z,a['H']@z,1e-6,2e-5);checks['minJ']=metric(b['min_detF'],a['min_detF'],1e-12,2e-5)
        vals.append(b);out.append(dict(state=i,checks=checks,passed=all(x['passed'] for x in checks.values())))
    dq=q1-q0;bar=(vals[0]['gradient']+4*vals[2]['gradient']+vals[1]['gradient'])/6;dv=vals[1]['volume']-vals[0]['volume'];p=np.array(h[8]['state'].child_states['fluid']['pressure_Pa'])
    chain=metric(np.einsum('cij,ij->c',bar,dq),dv,1e-12,2e-5);pressure=metric(.8*float(p@dv),.8*float(np.einsum('c,cij,ij',p,bar,dq)),1e-12,2e-5)
    owned=candidate.evaluate(q0);owned['gradient'][:]=123;ownership=np.array_equal(candidate.evaluate(q0)['gradient'],vals[0]['gradient'])
    saved=candidate.assembler;candidate.assembler=object()
    try:candidate.evaluate(q0)
    except ValueError:rejected=True
    else:rejected=False
    finally:candidate.assembler=saved
    candidate.cache.clear();mp=candidate.local[0][2];original=mp.gradient_adjoint
    def fail(*a,**k):raise MemoryError('registered call-local adjoint failure')
    mp.gradient_adjoint=fail
    try:candidate.evaluate(q0)
    except MemoryError:failed_clean=not candidate.cache
    else:failed_clean=False
    finally:mp.gradient_adjoint=original
    retry=metric(candidate.evaluate(q0)['gradient'],vals[0]['gradient'],1e-8,2e-5)
    passed=all(x['passed'] for x in out) and chain['passed'] and pressure['passed'] and ownership and rejected and failed_clean and retry['passed']
    write(run/'S3/operator-check.json',dict(status='passed_scoped' if passed else 'failed',records=out,chain=chain,pressure_work=pressure,implementation=candidate.implementation))
    write(run/'S3/buffer-lifecycle.json',dict(status='passed_scoped' if passed else 'failed',owned_returns=ownership,metadata_rejected=rejected,failed_call_no_cache=failed_clean,retry=retry,persistent_buffers=False,dynamic_fault_steps=0))
    update('S3唯一候选FBD：4状态算子、完整P压力功/体积链、返回所有权、元数据失效与局部失败重建 '+('通过。' if passed else '失败。'))
    print('STATIC',passed,flush=True)
    if not passed:raise ValueError('candidate operator failed')

def one(run,kind,index):
    run=Path(run);mutable(run)
    if read(run/'S3/operator-check.json')['status']!='passed_scoped':raise ValueError('operator not qualified')
    tick=time.perf_counter();c,m,cfg,ident=setup(run);base_build=time.perf_counter()-tick
    if kind=='FBD':c.core.geometry=FactoredGradientGeometry.adopt(c.geometry)
    extra=c.geometry.additional_setup_s if kind=='FBD' else 0.;h=inputs();step=(8,16)[index];a,b=h[step],h[step+1]
    ident.update(performance_backend=kind,input_digest=a['state'].digest(),numerical_sources=numerical_sources())
    folder,store=start_store(run,f'S3/{kind}{index}',c,ident,a);safe=SafePublication(c,store);cache=CachedProbes(m)
    # Both implementations get identical one-state warm-up, excluded from timing.
    c.geometry.evaluate(a['state'].q);m.evaluate(a['state'].q);c.geometry.cache.clear();c.geometry.profile={}
    wp.synchronize_device(m.operator.device);tick=time.perf_counter();row=attempt(run,'S3',folder.name,safe.advance);f=frame(c,cache,c.state);stats=field_stats(f,m.parent.params.fiber_direction);write(folder/'field-statistics.json',stats);wp.synchronize_device(m.operator.device);elapsed=time.perf_counter()-tick
    checks=equivalence(c.state,b['state'],row,b['rows'][-1]);bal=balances(store.history());passed=all(x['passed'] for x in checks.values()) and bal['passed']
    write(folder/'measurement.json',dict(status='passed_scoped' if passed else 'failed',kind=kind,input_step=step,advance_s=elapsed,build_s=base_build,additional_setup_s=extra,checks=checks,balances=bal,geometry=c.geometry.profile,publication=safe.profile,statistics=stats,iterations=row['iterations'],includes_probe_and_IO=True,identical_one_state_warmup=True))
    print('PAIR',kind,index,elapsed,flush=True)
    if not passed:raise ValueError('paired step failed')

def review(run):
    run=Path(run);mutable(run);records=[]
    for i in (0,1):
        a=read(run/f'S3/BD{i}/measurement.json');b=read(run/f'S3/FBD{i}/measurement.json');checks={}
        for region in a['statistics']:
            for key in a['statistics'][region]:
                for stat in ('mean','rms','max_abs'):
                    checks[f'{region}/{key}/{stat}']=metric(a['statistics'][region][key][stat],b['statistics'][region][key][stat],1e-8 if key in ('x','velocity') else 1e-6,2e-5)
        gain=1-b['advance_s']/a['advance_s'];saving=a['advance_s']-b['advance_s'];extra=max(0.,b['build_s']+b['additional_setup_s']-a['build_s'])
        records.append(dict(input_step=a['input_step'],BD_s=a['advance_s'],FBD_s=b['advance_s'],gain_fraction=gain,setup_recovery_steps=extra/saving if saving>0 else None,iterations=[a['iterations'],b['iterations']],geometry_calls=[a['geometry']['field']['calls'],b['geometry']['field']['calls']],field_checks=checks,passed=a['status']==b['status']=='passed_scoped' and all(x['passed'] for x in checks.values())))
    jobs=[read(run/f'processes/pair-{kind}{i}.json') for kind,i in [('BD',0),('FBD',0),('FBD',1),('BD',1)]];isolated=all(not j['gpu_before'].strip() and not j['gpu_after'].strip() for j in jobs)
    eligible=all(x['passed'] and x['gain_fraction']>=.05 and x['setup_recovery_steps'] is not None and x['setup_recovery_steps']<=4 for x in records) and isolated
    write(run/'S3/paired-performance.json',dict(status='passed_scoped' if all(x['passed'] for x in records) else 'failed',records=records,isolated=isolated,candidate_eligible=bool(eligible),median_gain_fraction=float(np.median([x['gain_fraction'] for x in records])),scope='two actual inputs, whole single steps only',setup_recovery_limit=4))
    write(run/'S3/backend-decision.json',dict(status='requires_continuity' if eligible else 'retained',selected=False,candidate_eligible=bool(eligible),backend='BD',reason='four-step continuity pending' if eligible else 'speed/setup/isolation gate not met; retain certified BD'))
    if not eligible:write(run/'S3/continuous-check.json',dict(status='not_triggered',new_steps=0,reason='performance gate not met'))
    update('S3公平单步配对：'+str([(x['input_step'],x['gain_fraction'],x['setup_recovery_steps']) for x in records])+'；'+('进入4步连续检查。' if eligible else '未满足采用条件，保留BD。'))
    print('PERFORMANCE_REVIEW',eligible,[(x['gain_fraction'],x['setup_recovery_steps']) for x in records],flush=True)

def continuous(run):
    run=Path(run);mutable(run)
    if not read(run/'S3/backend-decision.json')['candidate_eligible']:raise ValueError('candidate not eligible')
    c,m,cfg,ident=setup(run);c.core.geometry=FactoredGradientGeometry.adopt(c.geometry);ident['performance_backend']='FBD';h=inputs()
    folder,store=start_store(run,'S3/continuous',c,ident,h[8]);safe=SafePublication(c,store);records=[]
    for step in range(9,13):
        row=attempt(run,'S3','FBD-continuous',safe.advance);checks=equivalence(c.state,h[step]['state'],row,h[step]['rows'][-1]);records.append(dict(step=step,checks=checks,passed=all(x['passed'] for x in checks.values())))
        if not records[-1]['passed']:break
    bal=balances(store.history());passed=len(records)==4 and all(x['passed'] for x in records) and bal['passed']
    write(run/'S3/continuous-check.json',dict(status='passed_scoped' if passed else 'limited',records=records,balances=bal,new_steps=len(records)))
    d=read(run/'S3/backend-decision.json');d.update(status='selected_scoped' if passed else 'retained',selected=passed,backend='FBD: factored local adjoint, specified BD research window' if passed else 'BD',reason='operator, two paired steps and four-step continuity passed' if passed else 'continuity failed');write(run/'S3/backend-decision.json',d);update('S3连续检查：'+d['reason'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['static','one','review','continuous']);p.add_argument('--run',type=Path,required=True);p.add_argument('--kind',choices=['BD','FBD'],default='BD');p.add_argument('--index',type=int,choices=[0,1],default=0);a=p.parse_args()
    if a.phase=='one':one(a.run,a.kind,a.index)
    else:globals()[a.phase](a.run)
