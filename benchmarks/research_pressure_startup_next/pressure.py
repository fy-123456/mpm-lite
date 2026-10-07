"""Bounded CPU screening; temporal and spatial qualifications stay separate."""
from pathlib import Path
import argparse,time
import numpy as np
from .provenance import *
from benchmarks.research_stabilization_boundary_next.pressure import algebra,exact,comparison,graded,bisect
from engine.aniso_phase1.research_pressure_startup_next.theta import integrate,schedule

def coarse_observations(v,t):
    return dict(pressure=v['pressure'][::2],cumulative=v['cumulative'][::2],flux=np.diff(v['cumulative'][::2],axis=0)/np.diff(t[::2])[:,None])

def screen(run):
    run=Path(run);verify(run);tick=time.perf_counter();protocol=read(OBS/'S3/new-scene-protocol.json');p=protocol['parameters'];ts=np.linspace(0,2e-4,17);fine=np.linspace(0,2e-4,33)
    register(run,'S1/fixed-inputs.json',dict(status='registered',source=str(OBS/'S3/new-scene-protocol.json'),source_sha256=sha(OBS/'S3/new-scene-protocol.json'),parameters=p,families=['original','graded4'],methods=['backward-euler','startup'],times_s=ts.tolist(),fine_times_s=fine.tolist(),switch_time_s=2.5e-5,max_matrix_assemblies=16,max_spectral_calls=64,max_steps=1024,max_seconds=600,engineering_fraction=1.,reference_fraction=.25,first_interval_included=True))
    allrows=[];spaces=[];count=0;calls=0;steps=0;selected=None
    for family in ('original','graded4'):
        cuts=protocol['cuts']['coarse'] if family=='original' else graded(protocol['cuts']['coarse'],4)
        models={};ex={}
        for n in ((16,32) if family=='original' else (16,32,64,128)):
            a=algebra(cuts,p);count+=1;models[n]=a;ex[n]=exact(a,p,fine);calls+=1
            np.savez_compressed(run/'S1'/f'{family}-{n}-algebra.npz',H=a['H'],C=a['C'],L=a['L'],rhs=a['rhs'],eigenvalues=a['lam'],pressure_exact=ex[n]['pressure'],cumulative_exact=ex[n]['cumulative'],times=fine)
            cuts=bisect(cuts)
        ref=None;space=None
        if family=='graded4':
            ref=comparison(models[64],models[128],ex[64],ex[128],fine,.25)
            space=comparison(models[32],models[128],ex[32],ex[128],fine)
        spatial=dict(family=family,reference=ref,space=space,grid_accuracy_passed=bool(ref and ref['status']==space['status']=='passed_scoped'))
        spaces.append(spatial)
        for method in ('backward-euler','startup'):
            gridchecks={}
            for n in (16,32):
                a=models[n];results=[]
                for label,t in [('h',ts),('half',fine)]:
                    val=integrate(a,p,t,method);steps+=len(t)-1
                    ev=ex[n] if label=='half' else coarse_observations(ex[n],fine)
                    cmp=comparison(a,a,val,ev,t)
                    coarse=cmp if label=='h' else comparison(a,a,coarse_observations(val,t),coarse_observations(ev,t),ts)
                    physical=dict(minimum_pressure_Pa=float(val['pressure'].min()),max_mass_defect_m3=max(x['mass_defect_m3'] for x in val['ledger']),max_energy_balance_J=max(abs(x['energy_balance_J']) for x in val['ledger']),numerical_dissipation_J=sum(x['numerical_dissipation_J'] for x in val['ledger']),darcy_dissipation_J=sum(x['darcy_dissipation_J'] for x in val['ledger']))
                    ok=cmp['status']=='passed_scoped' and coarse['status']=='passed_scoped' and physical['minimum_pressure_Pa']>=0 and physical['max_mass_defect_m3']<1e-10 and physical['max_energy_balance_J']<1e-9
                    results.append(dict(label=label,status='passed_scoped' if ok else 'limited',raw_intervals=cmp,engineering_intervals=coarse,physical=physical))
                    np.savez_compressed(run/'S1'/f'{family}-{n}-{method}-{label}.npz',times=t,pressure=val['pressure'],cumulative=val['cumulative'],flux=val['flux'],theta=val['theta'],numerical_dissipation=[x['numerical_dissipation_J'] for x in val['ledger']],darcy_dissipation=[x['darcy_dissipation_J'] for x in val['ledger']])
                gridchecks[str(n)]=dict(passed=all(x['status']=='passed_scoped' for x in results),records=results)
            row=dict(family=family,method=method,grids=gridchecks,score=max(v for n in gridchecks.values() for x in n['records'] for v in x['raw_intervals']['max_budget_ratios'].values()))
            allrows.append(row); print('PRESSURE',family,method,{k:v['passed'] for k,v in gridchecks.items()},'max',row['score'],flush=True)
            if time.perf_counter()-tick>600:raise TimeoutError('pressure screening budget')
    eligible=[r for r in allrows if r['grids']['16']['passed'] and (r['family']=='original' or (r['grids']['32']['passed'] and spaces[1]['grid_accuracy_passed']))]
    if eligible:
        graded_ok=[r for r in eligible if r['family']=='graded4'];choice=min(graded_ok or eligible,key=lambda r:r['score'])
        cuts=protocol['cuts']['coarse'] if choice['family']=='original' else graded(protocol['cuts']['coarse'],4)
        selected=dict(family=choice['family'],method=choice['method'],cuts=dict(coarse=cuts,fine=bisect(cuts)),parameters=p,times_s=ts.tolist(),theta_h=schedule(ts,choice['method']).tolist(),theta_half=schedule(fine,choice['method']).tolist(),fine_eligible=choice['grids']['32']['passed'],grid_accuracy_passed=choice['family']=='graded4' and spaces[1]['grid_accuracy_passed'])
    write(run/'S1/startup-screen.json',dict(status='passed_scoped' if selected else 'limited',records=allrows,spatial=spaces,seconds=time.perf_counter()-tick,matrix_assemblies=count,spectral_calls=calls,matrix_steps=steps))
    write(run/'S1/startup-decision.json',dict(status='passed_scoped' if selected else 'limited',selected=selected,same_grid_time_passed=selected is not None,reference_reliable=spaces[1]['reference']['status']=='passed_scoped',grid_accuracy_passed=bool(selected and selected['grid_accuracy_passed']),no_clipping=True,no_new_dynamic_steps=True))
    if selected:register(run,'S5/selected-protocol.json',dict(status='registered',**selected,max_attempts=100))
    else:write(run/'S5/coupling-decision.json',dict(status='not_triggered',reason='neither frozen startup strategy meets raw and engineering same-grid flow budgets',new_dynamic_steps=0,production_C_E_integration=False,coupled_q5=False))
    contract='''# Pressure theta work contract
The original solid AVF, full M7/q7, current full-tensor H and exact volume discrete gradient are retained.
p_theta=(1-theta)*p0+theta*p1. Solid pressure force=-alpha*Gbar.T*p_theta.
Mass residual=alpha*DeltaV+C*DeltaP+h*B*z-h*s. Darcy residual=H*z-B.T*p_theta+g_b.
Pressure work cancels cell by cell because Gbar*Deltaq=DeltaV.
C=diag(storage*V0) is constant. D_num=(theta-0.5)*DeltaP.T*C*DeltaP is nonnegative and separate from h*z.T*H*z.
Energy balance=DeltaE+D_Darcy+D_num-Wext-h*p_theta.T*s+h*g_b.T*z.
Pressure blocks: -h*alpha*theta*Gbar.T and -theta*B.T. The existing geometry quasi-Newton approximation and true residual checks remain.
The solid AVF path error and residual work are recorded separately. Energy closure alone is not an accuracy certificate.
New theta schedule and cumulative D_num must enter the method identity, checkpoint and full transaction. New methods start at the common t=0 initial state.
This contract does not certify actual coupled execution; see the conditional coupling decision.
'''
    (run/'S1/discrete-work-contract.md').write_text(contract)
    write(run/'S1/equation-protocol.json',dict(status='derived_pending_coupled' if selected else 'derived_fixed_skeleton_only',constant_capacity=True,pressure_work='same p_theta on both sides',numerical_dissipation='(theta-.5)*DeltaP.T*C*DeltaP',new_state_fields=['theta_schedule_identity','cumulative_numerical_dissipation_J'],solid_AVF_unchanged=True,contract_sha256=sha(run/'S1/discrete-work-contract.md')))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    with serial_lock(args.run):screen(args.run)
