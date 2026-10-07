"""Two pre-registered CPU startup candidates, fixed observations and raw diagnostics."""
import argparse,time
import numpy as np
from .provenance import *
from engine.aniso_phase1.research_startup_substeps_next.schedule import OBSERVATIONS,make_times,aggregate
from engine.aniso_phase1.research_boundary_reference_next.reference import boundary_grid
from engine.aniso_phase1.research_pressure_startup_next.theta import integrate,schedule
from benchmarks.research_stabilization_boundary_next.pressure import algebra,exact,comparison,bisect
from benchmarks.research_restoring_rt0_next.grid_transfer import restriction

def prepare():
    run=freeze();p=read(APP/'R0/input-contract.json');cuts=bisect(boundary_grid(p['cuts']['coarse'],4e-6))
    register(run,'S0/physical-contract.json',dict(parameters=p['parameters'],cuts=cuts,engineering_times_s=OBSERVATIONS.tolist(),source_m3_s=0.,old_cuts=p['cuts'],old_times_s=p['times_s'],selected_space=read(APP/'selected-space.json'),mass_order=7,material_order=7,physical_input_sha256=sha(APP/'R0/input-contract.json'),tensor_source=dict(path='engine/aniso_phase1/research_local_span_next/rt0.py',sha256=sha(ROOT/'engine/aniso_phase1/research_local_span_next/rt0.py'))))
    tables={c:{label:make_times(c,fine).tolist() for label,fine in [('h',False),('half',True)]} for c in ('U2','U4')}
    register(run,'S1/time-protocol.json',dict(status='registered',tables=tables,observations_s=OBSERVATIONS.tolist(),theta={c:{k:schedule(t,'startup').tolist() for k,t in v.items()} for c,v in tables.items()},method='startup',switch_s=25e-6,refined_until_s=50e-6))
    register(run,'S1/observation-contract.json',dict(status='registered',flux='sum(h*z)/engineering_dt; face flux already m3/s',reaction='sum(h*R)/engineering_dt',fields='endpoint states at explicit indices',energy='sum of committed actual-step ledgers',raw_intervals_retained=True,partial_intervals_not_published=True,raw_accuracy_separate=True))
    register(run,'S1/transaction-contract.json',dict(status='inherited_core_with_new_time_identity',microstep_is_full_solid_fluid_step=True,engineering_interval_is_not_atomic_transaction=True,immutable_actual_times=True,source_zero=True,inherited_fault=dict(path=str(APP/'R4/fault-B0/failure.json'),sha256=sha(APP/'R4/fault-B0/failure.json')),new_fault_required_if_dynamic_branch_runs=True))
    register(run,'S2/screen-protocol.json',dict(status='registered',candidate_order=['U2','U4'],cells=[32,64,128],mother_first_m=4e-6,growth=2.2,reference_fraction=.25,field_rtol=.05,max_CPU_matrices=6,max_spectral_calls=8,max_matrix_steps=512,max_seconds=600))
    PROGRESS.write_text(f'# 启动子步与共享几何实施进展\n\n执行[{PLAN.name}]({PLAN.name})。开工最新BASE为`{APP.name}`，发布SHA `{APP_SHA}`；22源码/22快照/274产物及祖先审计通过。系统盘约7.9GiB，未触发迁移。\n\n结果目录：`{run.relative_to(ROOT)}`。S0已冻结，S1时间表、聚合与事务协议已登记。旧发布保持只读。\n')
    print('RUN',run,flush=True)

def screen(run):
    run=Path(run);mutable(run);begun=time.perf_counter();p=read(run/'S0/physical-contract.json');params=p['parameters'];cuts=p['cuts'];models={};transfers=[]
    for n in (32,64,128):models[n]=algebra(cuts,params);cuts=bisect(cuts)
    for a,b in ((32,64),(64,128)):
        P,C,Z=restriction(models[a]['top'],models[b]['top']);transfers.append(dict(cells=[a,b],divergence_error=float(abs(models[a]['top'].B@Z-C@models[b]['top'].B).max()),volume_error=float(abs(C@models[b]['top'].V0-models[a]['top'].V0).max()),constant_pressure_error=float(abs(P@np.ones(b)-1).max())))
    ev={n:exact(a,params,OBSERVATIONS) for n,a in models.items()};calls=3;ref=comparison(models[64],models[128],ev[64],ev[128],OBSERVATIONS,.25);spatial=comparison(models[32],models[128],ev[32],ev[128],OBSERVATIONS)
    write(run/'S2/reference-check.json',dict(status='passed_scoped' if ref['status']==spatial['status']=='passed_scoped' else 'limited',reference=ref,spatial=spatial,transfers=transfers,x_refinement_only=True,raw_microstep_continuum_reference=False))
    write(run/'S2/reference-binding.json',dict(status='recomputed_same_family',ancestor_screen_sha256=sha(APP/'R2/grid-screen.json'),parameters=params,cuts={str(n):[x.tolist() for x in a['top'].cuts] for n,a in models.items()},observations_s=OBSERVATIONS.tolist(),independent_scalar_reference=dict(path=str(APP/'R1/scalar-reference.json'),sha256=sha(APP/'R1/scalar-reference.json'))))
    baseline_file=APP/'R2/grid-4e-06-uniform16-h.npz'
    with np.load(baseline_file) as f:baseline={k:f[k] for k in ('pressure','flux','cumulative')};baseline_t=f['times']
    baseline=aggregate(baseline,baseline_t,OBSERVATIONS);base_cmp=comparison(models[32],models[32],baseline,ev[32],OBSERVATIONS);write(run/'S2/baseline.json',dict(source=str(baseline_file),sha256=sha(baseline_file),comparison=base_cmp,new_steps=0))
    records=[];selected=None;steps=0
    for candidate in ('U2','U4'):
        pair={};trials=[]
        for label,fine in [('h',False),('half',True)]:
            t=make_times(candidate,fine);v=integrate(models[32],params,t,'startup');steps+=len(t)-1;ex=exact(models[32],params,t);calls+=1;eng=aggregate(v,t,OBSERVATIONS);cmp=comparison(models[32],models[32],eng,ev[32],OBSERVATIONS);raw=comparison(models[32],models[32],v,ex,t);pair[label]=eng
            energy=max(abs(x['energy_balance_J']) for x in v['ledger']);mass=max(x['mass_defect_m3'] for x in v['ledger']);minp=float(v['pressure'].min());hard=bool(minp>=0 and energy<1e-9 and mass<1e-10 and all(x['source_work_J']==0 for x in v['ledger']))
            trials.append(dict(label=label,steps=len(t)-1,engineering=cmp,raw=raw,minimum_pressure_Pa=minp,max_mass_defect_m3=mass,max_energy_balance_J=energy,hard_passed=hard,Dnum_J=sum(x['numerical_dissipation_J'] for x in v['ledger']),Darcy_J=sum(x['darcy_dissipation_J'] for x in v['ledger'])))
            np.savez_compressed(run/'S2'/f'{candidate}-{label}.npz',times=t,pressure=v['pressure'],flux=v['flux'],cumulative=v['cumulative'],exact_pressure=ex['pressure'],exact_flux=ex['flux'],exact_cumulative=ex['cumulative'],engineering_times=OBSERVATIONS,engineering_pressure=eng['pressure'],engineering_flux=eng['flux'],engineering_cumulative=eng['cumulative'],exact_engineering_pressure=ev[32]['pressure'],exact_engineering_flux=ev[32]['flux'],exact_engineering_cumulative=ev[32]['cumulative'],theta=v['theta'],Dnum=[x['numerical_dissipation_J'] for x in v['ledger']])
        temporal=comparison(models[32],models[32],pair['h'],pair['half'],OBSERVATIONS);eligible=ref['status']==spatial['status']==temporal['status']=='passed_scoped' and all(v['hard_passed'] and v['engineering']['status']=='passed_scoped' for v in trials)
        early=max(r['budget_ratios']['face_flux'] for r in trials[0]['engineering']['records'][:4]);old_early=max(r['budget_ratios']['face_flux'] for r in base_cmp['records'][:4]);eligible=eligible and early<old_early
        value=dict(candidate=candidate,trials=trials,temporal=temporal,eligible=eligible,early_face_budget_ratio=early,baseline_early_face_budget_ratio=old_early);records.append(value);write(run/'S2'/f'candidate-{candidate}.json',value)
        print('CANDIDATE',candidate,'engineering',[max(v['engineering']['max_budget_ratios'].values()) for v in trials],'raw',[max(v['raw']['max_budget_ratios'].values()) for v in trials],'eligible',eligible,flush=True)
        if time.perf_counter()-begun>600 or steps>512 or calls>8:raise TimeoutError('CPU screen budget')
        if eligible:selected=candidate;break
    write(run/'S2/screen-accounting.json',dict(new_matrices=3,spectral_calls=calls,matrix_steps=steps,seconds=time.perf_counter()-begun,new_GPU_steps=0))
    register(run,'S2/method-decision.json',dict(status='qualified_engineering_window' if selected else 'limited',candidate=selected,raw_substep_accuracy=all(v['raw']['status']=='passed_scoped' for r in records if r['candidate']==selected for v in r['trials']) if selected else False,scope='fixed skeleton full tensor, boundary32 grid, fixed12.5us observations over0..200us',actual_coupled_accuracy=False,selected_geometry='BASE-bounded-Gauss'))
    if selected:register(run,'S3/coupled-protocol.json',dict(status='registered',candidate=selected,cuts=p['cuts'],parameters=params,observations_s=OBSERVATIONS.tolist(),times={k:make_times(selected,f).tolist() for k,f in [('h',False),('half',True)]},max_attempts=89,method='startup',geometry='BASE-bounded-Gauss'))
    else:write(run/'S3/grid-decision.json',dict(status='not_triggered',reason='both registered candidates fail engineering observation gates',new_dynamic_attempts=0))
    with PROGRESS.open('a') as f:f.write(f'\nS1小型时间/聚合逻辑已实现。S2 CPU筛选完成，选定候选：`{selected}`；原始子步误差与工程观察分列，网格/时间参考结果保存于S2。\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','screen']);p.add_argument('--run',type=Path);a=p.parse_args()
    with serial_lock(a.run):prepare() if a.phase=='prepare' else screen(a.run)
