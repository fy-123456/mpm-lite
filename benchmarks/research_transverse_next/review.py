"""Read-only engineering comparisons with separate signal and certificate scope."""
import argparse
import numpy as np
from .provenance import *
from .runtime import update
from .observables import modes,spatial_statistics
from .trajectory import FIELDS
from benchmarks.research_pressure3d_next.review import balances,probes
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_pressure_startup_next.coupling_review import fluid,context,compare_fluid
from benchmarks.research_pressure_window_next.candidate_study import fields,good
from benchmarks.research_sequential_next.compare import metric,regions
from benchmarks.research_restoring_rt0_next.grid_transfer import restriction
from engine.aniso_phase1.research_startup_substeps_next.schedule import aggregate,interval_rows,OBSERVATIONS

def contracts(run):
    run=Path(run);mutable(run);static={k:read(run/f'S1/{k}/operator-check.json') for k in ('Y64','Y128')}
    if not all(v['status']=='passed_scoped' for v in static.values()):raise ValueError('static entry failed')
    write(run/'S1/directional-operator-check.json',dict(status='passed_scoped',grids=static,static_state_groups=8,scope='rest independent RT0, one small displacement direction; original full P and q7 retained'))
    write(run/'S1/memory-preflight.json',dict(status='passed_scoped',geometry={k:v['geometry'] for k,v in static.items()},original_guard=.7,cap='min(256 MiB, .02*initial free)',max_RSS_GiB=max(v['peak_RSS_GiB'] for v in static.values())))
    cases=read(run/'S0/initial-condition-protocol.json')['cases'];values={}
    cuts=read(run/'S0/grid-protocol-inherited.json')['cuts'];storage=read(run/'S0/coupled-protocol.json')['parameters']['storage']
    ca=context(cuts['y'],storage);cb=context(cuts['yz'],storage);P,C,Z=restriction(ca['top'],cb['top'])
    a=np.array(cases['Y64']['pressure_Pa']);b=np.array(cases['Y128']['pressure_Pa'])
    checks=dict(pressure=metric(a,P@b,1e-12,2e-5),content=metric(ca['C']*a,C@(cb['C']*b),1e-14,2e-5),storage=metric(cases['Y64']['initial_pressure_storage_J'],cases['Y128']['initial_pressure_storage_J'],1e-14,2e-5))
    if not all(v['passed'] for v in checks.values()):raise ValueError('initial conservative projection differs')
    for name,case in cases.items():
        top=ca['top'] if name=='Y64' else cb['top'];values[name]=dict(modes=modes(top,case['pressure_Pa']),storage_J=case['initial_pressure_storage_J'],delta_storage_J=case['perturbation_energy_J'],content_m3=case['initial_content_sum_m3'])
    log=run/'processes/test-contracts.log'
    if 'Ran 5 tests' not in log.read_text() or '\nOK\n' not in log.read_text():raise ValueError('contract tests missing')
    write(run/'S1/initialization-check.json',dict(status='passed_scoped',cases=values,CPU_contract_log=str(log.relative_to(run)),immutable_tag='transverse_initial_contract',initial_digest_nonrecursive=True,foreign_initial_history_rejected=True))
    write(run/'S1/observable-check.json',dict(status='passed_scoped',cases=values,all_spatial_axes_contracted=True,unresolved_Az_Y64=None,full_tensor_direction=static,contract_tests=str(log.relative_to(run))))
    write(run/'S1/projection-contract.json',dict(status='passed_scoped',checks=checks,pressure='volume average',content='conservative sum',face_flux='oriented volume-rate sum, no extra area factor',probe_ownership='right cell at shared cut, exterior clipped'))
    inherited=APP/'S6/test-report.json'
    write(run/'S1/publication-contract.json',dict(status='passed_scoped',new_CPU_tests=str(log.relative_to(run)),inherited_pending_bad_hash_wrong_model_tests=dict(path=str(inherited),sha256=sha(inherited)),atomic_state_ledger_frame=True,frame_failure_rollback=True,lost_ack_single_commit=True,real_GPU_fault_pending=True))

def case_review(run,case):
    folder=run/'cases'/case;bal=balances(folder);h=history(folder);obs=read(folder/'observables.json');X,d,fp=probes(folder)
    regions_out=[]
    for t,f in sorted(fp.items()):
        out={}
        for name,w in regions(X).items():
            values={k:spatial_statistics(X,f[k]-X if k=='x' else f[k],w) for k in FIELDS}
            values['fiber_skeleton']=spatial_statistics(X,np.einsum('i,...ij,j->...',d,f['PK1'],d),w)
            values['fiber_total']=spatial_statistics(X,np.einsum('i,...ij,j->...',d,f['PK1_total'],d),w);out[name]=values
        regions_out.append(dict(time_s=t,regions=out))
    final=fp[max(fp)];u=final['x']-X
    signal=dict(max_component_displacement_m=np.max(abs(u),axis=(0,1,2)).tolist(),max_component_velocity_m_s=np.max(abs(final['velocity']),axis=(0,1,2)).tolist(),displacement_below_absolute_floor=bool(np.max(abs(u))<5e-5),initial_modes=obs[0]['modes'],final_modes=obs[-1]['modes'],initial_to_final_Ay_Pa=obs[-1]['modes']['Ay']['value_Pa']-obs[0]['modes']['Ay']['value_Pa'],initial_to_final_Az_Pa=None if not obs[0]['modes']['Az']['resolved'] else obs[-1]['modes']['Az']['value_Pa']-obs[0]['modes']['Az']['value_Pa'])
    out=dict(status='passed_scoped' if bal['passed'] else 'failed',balances=bal,signals=signal,regions=regions_out,observations=str((folder/'observables.json').relative_to(run)),physical_window_s=[0,75e-6],scope='nonnegative stable specified response; no dynamic continuum reference')
    write(run/f'S2/{case}-review.json',out)
    if not bal['passed']:raise ValueError(case+' physical hard condition failed')
    return out

def compare(run):
    run=Path(run);mutable(run);reports={name:case_review(run,name) for name in ('Y64','Y128','YZ128')}
    cuts=read(run/'S0/grid-protocol-inherited.json')['cuts'];storage=read(run/'S0/coupled-protocol.json')['parameters']['storage'];ca=context(cuts['y'],storage);cb=context(cuts['yz'],storage)
    a=run/'cases/Y64';b=run/'cases/Y128';va,t,rows=fluid(a);vb,tt,rhs=fluid(b);obs=OBSERVATIONS[:7];aa=aggregate(va,t,obs);bb=aggregate(vb,tt,obs);flow=compare_fluid(ca,cb,aa,bb,obs);P,C,Z=restriction(ca['top'],cb['top'])
    X,d,pa=probes(a);Y,_,pb=probes(b)
    if not np.array_equal(X,Y):raise ValueError('physical probe positions differ')
    ra=interval_rows(rows,obs,average=['reaction_N']);rb=interval_rows(rhs,obs,average=['reaction_N']);solid=[];transverse=[]
    for i,t0 in enumerate(obs[1:],1):
        t0=round(float(t0),14);fa,fb=pa[t0],pb[t0];checks=fields(dict(X=X,**fa),dict(X=X,**fb),d)
        for region,w in regions(X).items():
            checks[region].update({k:metric(fa[k],fb[k],.02,.05,w) for k in ('PK1_total','Cauchy_skeleton','Cauchy_total')})
            checks[region]['fiber_total']=metric(np.einsum('i,...ij,j->...',d,fa['PK1_total'],d),np.einsum('i,...ij,j->...',d,fb['PK1_total'],d),.02,.05,w)
        reaction=metric(ra[i-1]['reaction_N'],rb[i-1]['reaction_N'],1e-4,.05)
        solid.append(dict(time_s=t0,checks=checks,reaction=reaction,passed=good(checks) and reaction['passed'],max_displacement_difference_m=float(np.max(abs(fa['x']-fb['x'])))))
        ma=modes(ca['top'],aa['pressure'][i]);mb=modes(cb['top'],bb['pressure'][i]);ch=dict(Ay=metric(ma['Ay']['value_Pa'],mb['Ay']['value_Pa'],.001,.05),mean=metric(ma['mean_Pa'],mb['mean_Pa'],.001,.05))
        for axis,name in enumerate('xyz'):
            ids=ca['top'].internal[ca['top'].axes[ca['top'].internal]==axis]
            if len(ids):ch['internal_'+name]=metric(aa['flux'][i-1][ids],(Z@bb['flux'][i-1])[ids],1e-10,.05)
        # Pressure is also compared in each physical transverse partition.
        for iy in range(2):
            mask=(np.indices(ca['top'].shape)[1].ravel()==iy);w=ca['top'].V0*mask
            ch[f'pressure_y{iy}']=metric(aa['pressure'][i],P@bb['pressure'][i],.001,.05,w)
        transverse.append(dict(time_s=t0,checks=ch,passed=all(v['passed'] for v in ch.values()),Az_comparison='Y64 unresolved; Y128 diagnostic only',Y128_Az=mb['Az']))
    passed=flow['status']=='passed_scoped' and all(x['passed'] for x in solid+transverse)
    out=dict(status='passed_scoped' if passed else 'limited',flow=flow,solid=solid,transverse=transverse,scope='same continuous Y initial field; z subdivision sensitivity only',full_spatial_accuracy=False,independent_dynamic_solid_reference=False)
    write(run/'S2/transverse-grid-comparison.json',out)
    write(run/'S2/mixed-direction-review.json',dict(reports['YZ128'],comparison_to_Y64_is_error_estimate=False,short_transverse_diffusion_limit='small mode change does not imply solver failure; see initial/final amplitude and static Darcy direction'))
    write(run/'S2/scope-decision.json',dict(status='passed_scoped' if passed else 'limited',stable_cases=list(reports),engineering_z_subdivision_consistency=passed,full_spatial_accuracy=False,temporal_accuracy=False,raw_peak_accuracy=False,mechanical_signal_limited=all(v['signals']['displacement_below_absolute_floor'] for v in reports.values()),dynamic_performance_allowed=True))
    need=not passed
    write(run/'S3/time-entry-decision.json',dict(status='triggered' if need else 'not_triggered',need_half=need,case='Y128' if need else None,reason='important engineering discrepancy may have temporal contamination' if need else 'all registered engineering comparisons and hard conditions pass; no new same-input time reference or application requirement for startup peaks',inherited_uniform_raw_microstep_difference=.1245,inherited_is_not_new_case_error=True,adjacent_physical_changes_not_error_estimates=True))
    update(run,f'S2三轨迹硬条件通过；Y64/Y128工程比较{out["status"]}，流体最大预算比{max(flow["max_budget_ratios"].values()):.5g}。S3时间细分{("触发" if need else "未触发")}；机械响应低于绝对位移验收尺度，不宣称相对精度。')
    print('TRANSVERSE_REVIEW',out['status'],flow['max_budget_ratios'],{k:v['signals'] for k,v in reports.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['contracts','compare']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):contracts(a.run) if a.phase=='contracts' else compare(a.run)
