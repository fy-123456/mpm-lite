"""Full-loading F45 refinement followed by a history-gradient intervention."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import warp as wp
from benchmarks import aniso_mainline as baseline
from benchmarks import aniso_boundary_impulse as previous
from benchmarks.aniso_gradient_control import GradientControlLedger
from demos.aniso import Scene
from utils.resource_guard import prepare_warp_cache,inspect_storage

ROOT=baseline.ROOT
DEFAULT=ROOT/'docs/results/lite-aniso-mainline/v4'
TESTS=previous.TESTS+['tests.test_aniso_gradient_control']
LEVELS={'coarse':.001,'fine':.0005,'finest':.00025}
MODES=('baseline','pic_history')


def hashes():
    h=previous.hashes()
    for name in ('benchmarks/aniso_f45_refinement.py','benchmarks/aniso_gradient_control.py','tests/test_aniso_gradient_control.py'):
        h[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return h


def verify(p):
    current=hashes();changed=[n for n,h in p['source_sha256'].items() if current.get(n)!=h]
    if changed:raise RuntimeError('frozen sources changed: '+str(changed))


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve existing protocol')
    configs={f'{mode}-{level}':asdict(replace(baseline.BASE,kf=200.,fiber_angle=45.,dt=dt,
        reaction_force_atol=1e-7,boundary_impulse_transfer=True)) for mode in MODES for level,dt in LEVELS.items()}
    p=dict(name='lite-f45-full-refinement-v4',frozen_at=datetime.now(timezone.utc).isoformat(),source_sha256=hashes(),
        configs=configs,modes={name:name.rsplit('-',1)[0] for name in configs},device='cpu',precision='float64',
        duration=.5,endpoint_displacement_m=.005,tests=TESTS,snapshot_times=[.05,.1,.25,.5],
        run_count=6,total_steps=7000,stages=['baseline','pic_history'],
        acceptance={'time_interval':[.05,.5],'relative_force_difference_max':.05,'force_rms_floor_N':.001,
                    'force_error_max_N':1e-7,'particle_grid_gap_max_kg_m_s':1e-14,
                    'grid_grip_velocity_max_m_s':1e-12,'particle_grip_error_max_m':.001,
                    'snapshot_identity_max':1e-12,'prefix_v3_state_max':1e-12,'all_tests_no_skips':True},
        trend='all three curves on common coarse times; rho=D(fine,finest)/D(coarse,fine); p=log2(1/rho); not proof of asymptotic convergence',
        control='only particle F uses analytic derivative of the PIC position interpolation; keep x/v/APIC G/center force/tangent formulas unchanged',
        snapshots='same-state production versus PIC-history; frozen gradient/covariance/moving-weight decomposition; finite nonlinear stress and energy interventions',
        cuda={'status':'not_run','reason':'CPU validation; visible A100 already occupied'},
        resource={'max_cpu_processes':2,'threads_each':1,'storage':asdict(inspect_storage())},
        previous_manifests={v:hashlib.sha256((ROOT/f'docs/results/lite-aniso-mainline/{v}/artifact-sha256.json').read_bytes()).hexdigest() for v in ('v1','v2','v3')})
    baseline.write_json(out/'protocol.json',p)


def worker(out,p,name):
    dest=out/'cases'/name;mode=p['modes'][name]
    class AuditedScene(Scene):
        def __init__(self,config,device):
            super().__init__(config,device);self.solver.energy_ledger=GradientControlLedger(mode)
            self.audit_steps={round(t/config.dt) for t in p['snapshot_times']}
        def step(self):
            ledger=self.solver.energy_ledger;ledger.audit_next=self.solver.sim_steps+1 in self.audit_steps
            ok=super().step()
            if ok and ledger.audit_next:
                step=self.solver.sim_steps
                np.savez_compressed(dest/f'audit-{step:05d}.npz',**ledger.snapshot)
                baseline.write_json(dest/f'audit-{step:05d}.json',ledger.audit)
            return ok
    baseline.Scene=AuditedScene
    return baseline.run_case(out,name,p['configs'][name],p)


def run(out,p,stage,jobs):
    if not json.loads((out/'tests.json').read_text())['required_checks_passed']:raise RuntimeError('tests must pass')
    if stage=='pic_history':
        trend=json.loads((out/'trend-baseline.json').read_text())
        if not trend['all_physical_checks_passed']:raise RuntimeError('baseline physical checks must pass before control')
    names=[n for n in p['configs'] if p['modes'][n]==stage]
    def launch(name):
        log=out/(name+'.log')
        if log.exists() or (out/'cases'/name).exists():raise RuntimeError('preserve existing case '+name)
        with log.open('w') as stream:
            proc=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_f45_refinement','worker','--output',str(out),'--case',name],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        print(name,'exit',proc.returncode,flush=True);return dict(case=name,exit_code=proc.returncode)
    with ThreadPoolExecutor(max_workers=jobs) as pool:records=list(pool.map(launch,names))
    baseline.write_json(out/f'batch-{stage}.json',records)
    return all(r['exit_code']==0 for r in records)


def read_cases(out,names):
    data={};checks={};audits={}
    for name in names:
        path=out/'cases'/name;status=json.loads((path/'status.json').read_text())
        if not status['run_completed']:
            checks[name]={'passed':False,'status':status};continue
        rows=[json.loads(line) for line in (path/'steps.jsonl').read_text().splitlines()];data[name]=rows
        c=dict(steps=len(rows),max_grid_force_error_N=max(abs(r['momentum_balance_error']) for r in rows),
            max_particle_force_error_N=max(r['particle_momentum_balance_error_norm'] for r in rows),
            max_particle_grid_gap_kg_m_s=max(r['particle_grid_momentum_gap_norm'] for r in rows),
            max_force_residual_bound_N=max(r['reaction_force_residual_bound'] for r in rows),
            min_particle_J=min(r['min_particle_det_F'] for r in rows),min_center_J=min(r['min_det_F'] for r in rows),
            max_grid_grip_velocity_error=max(r['grid_grip_velocity_error'] for r in rows),
            max_particle_grip_error=max(abs(r['measured_grip_displacement']-r['displacement']) for r in rows),
            max_energy_budget_closure_J=max(abs(r['budget_closure']) for r in rows),
            all_nonlinear_targets_pass=all(r['last_residual_norm']<=r['newton_residual_target'] for r in rows),
            final_reaction_N=rows[-1]['right_force'],final_loading_work_J=rows[-1]['loading_work'])
        c['passed']=bool(c['max_grid_force_error_N']<=1e-7 and c['max_particle_force_error_N']<=1e-7
            and c['max_particle_grid_gap_kg_m_s']<=1e-14 and c['max_force_residual_bound_N']<=1e-7
            and c['min_particle_J']>0 and c['min_center_J']>0 and c['max_grid_grip_velocity_error']<=1e-12
            and c['max_particle_grip_error']<=.001 and c['all_nonlinear_targets_pass'])
        checks[name]=c
        audits[name]=[json.loads(f.read_text()) for f in sorted(path.glob('audit-*.json'))]
    return data,checks,audits


def refinement(data,mode):
    names=[f'{mode}-{level}' for level in LEVELS]
    if not all(n in data for n in names):return {'complete':False}
    t=np.array([r['time'] for r in data[names[0]]]);t=t[(t>=.05-1e-12)&(t<=.5+1e-12)]
    rms=lambda a:float(np.sqrt(np.trapezoid(a*a,t)/(t[-1]-t[0])))
    result={'complete':True,'components':{}}
    for key in ('right_force','right_elastic_force','right_inertial_force'):
        curves=[np.interp(t,[r['time'] for r in data[n]],[r[key] for r in data[n]]) for n in names]
        d01=curves[0]-curves[1];d12=curves[1]-curves[2];a=rms(d01);b=rms(d12)
        rho=b/max(a,1e-300)
        result['components'][key]=dict(coarse_fine_rms_N=a,fine_finest_rms_N=b,
            coarse_fine_relative=a/max(rms(curves[1]),.001),fine_finest_relative=b/max(rms(curves[2]),.001),
            finest_rms_N=rms(curves[2]),rho=rho,observed_order=float(-np.log2(max(rho,1e-300))),
            differences_cosine=float(np.trapezoid(d01*d12,t)/(t[-1]-t[0])/max(a*b,1e-300)),
            decreasing=bool(b<a),passes_5_percent=bool(b/max(rms(curves[2]),.001)<=.05))
    return result


def prefix_check(out,data):
    result={}
    for level in ('coarse','fine'):
        name='baseline-'+level;old=ROOT/f'docs/results/lite-aniso-mainline/v3/cases/F45-{level}-on'
        oldrows=[json.loads(x) for x in (old/'steps.jsonl').read_text().splitlines()]
        rows=data[name][:len(oldrows)]
        err=max(abs(a['right_force']-b['right_force']) for a,b in zip(rows,oldrows))
        with np.load(out/'cases'/name/'frames.npz') as new,np.load(old/'frames.npz') as prior:
            errors={'x':0.,'F':0.};matches=0
            for i,t in enumerate(new['time']):
                j=np.flatnonzero(abs(prior['time']-t)<1e-12)
                if not len(j):continue
                matches+=1
                for key in errors:errors[key]=max(errors[key],float(np.max(abs(new[key][i]-prior[key][j[0]]))))
        result[level]=dict(reaction_max_error_N=err,matched_frames=matches,state_errors=errors,
            passed=bool(err<=1e-12 and max(errors.values())<=1e-12 and matches>=5))
    return result


def trend(out,p):
    data,checks,audits=read_cases(out,[n for n in p['configs'] if p['modes'][n]=='baseline'])
    result=dict(checks=checks,refinement=refinement(data,'baseline'),audits=audits,
                all_physical_checks_passed=all(c['passed'] for c in checks.values()))
    if len(data)==3:result['v3_prefix']=prefix_check(out,data)
    baseline.write_json(out/'trend-baseline.json',result)
    print(json.dumps(result['refinement'],indent=2))
    return result['all_physical_checks_passed']


def analyze(out,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data,checks,audits=read_cases(out,p['configs']);trends={mode:refinement(data,mode) for mode in MODES}
    audit_checks={name:dict(count=len(rows),passed=len(rows)==4 and all(r['frozen_decomposition_max_error']<=1e-12 and r['pic_advection_oracle_max_error']<=1e-12 for r in rows)) for name,rows in audits.items()}
    prefix=prefix_check(out,data) if all('baseline-'+l in data for l in ('coarse','fine')) else {}
    same_dt={}
    for level in LEVELS:
        bn='baseline-'+level;cn='pic_history-'+level
        if bn not in data or cn not in data:continue
        t=np.array([r['time'] for r in data[bn]]);mask=t>=.05-1e-12;t=t[mask]
        r0=np.array([r['right_force'] for r in data[bn]])[mask];r1=np.array([r['right_force'] for r in data[cn]])[mask]
        rms=lambda a:float(np.sqrt(np.trapezoid(a*a,t)/(t[-1]-t[0])))
        same_dt[level]=dict(reaction_difference_rms_N=rms(r1-r0),relative_to_baseline=rms(r1-r0)/max(rms(r0),.001))
    physical=all(c['passed'] for c in checks.values())
    time_pass=all(t['complete'] and t['components']['right_force']['passes_5_percent'] for t in trends.values())
    required=json.loads((out/'tests.json').read_text())['required_checks_passed']
    summary=dict(required_checks_passed=required,checks=checks,refinement=trends,audits=audits,audit_checks=audit_checks,
        same_dt_control_effect=same_dt,v3_prefix=prefix,physical_checks_passed=physical,
        diagnostics_passed=all(c['passed'] for c in audit_checks.values()) and len(audit_checks)==6,
        time_sensitivity_passed=time_pass,experiment_completed=len(data)==6,
        production_default_changed=False,spatial_accuracy='not_verified',cuda='not_run')
    summary['all_acceptance_passed']=bool(required and physical and time_pass and summary['diagnostics_passed'] and all(c['passed'] for c in prefix.values()))
    baseline.write_json(out/'summary.json',summary)
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for i,mode in enumerate(MODES):
        for level in LEVELS:
            rows=data.get(mode+'-'+level)
            if not rows:continue
            for j,key in enumerate(('right_force','right_elastic_force','right_inertial_force')):
                axes[i,j].plot([r['displacement'] for r in rows],[r[key] for r in rows],label=level)
        for j,title in enumerate(('total','elastic','inertial')):
            axes[i,j].set(xlabel='command displacement (m)',ylabel='reaction (N)',title=mode+' / '+title)
            axes[i,j].legend();axes[i,j].grid(alpha=.25)
    fig.savefig(out/'full-reaction-refinement.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for level in LEVELS:
        rows=audits.get('baseline-'+level,[])
        if not rows:continue
        t=[r['time_end'] for r in rows]
        for j,key in enumerate(('gradient_gap_rate_rms','moving_weight_rate_rms','gradient_gap_stress_effect_rms_Pa')):
            axes[j].plot(t,[r[key] for r in rows],'-o',label=level)
    for ax,title,unit in zip(axes,('frozen gradient gap / dt','moving weights / dt','stress effect of gradient gap'),('1/s','1/s','Pa')):
        ax.set(xlabel='time (s)',ylabel=unit,title=title);ax.legend();ax.grid(alpha=.25)
    fig.savefig(out/'gradient-history-audit.png',dpi=160);plt.close(fig)
    return summary['all_acceptance_passed']


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('freeze','tests','run','worker','trend','analyze'))
    parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--stage',choices=MODES,default='baseline')
    parser.add_argument('--jobs',type=int,choices=(1,2),default=2);parser.add_argument('--case')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache')
    if args.action=='freeze':freeze(out);return
    p=json.loads((out/'protocol.json').read_text());verify(p)
    if args.action=='tests':ok=baseline.tests(out,p)
    elif args.action=='run':ok=run(out,p,args.stage,args.jobs)
    elif args.action=='worker':ok=worker(out,p,args.case)
    elif args.action=='trend':ok=trend(out,p)
    else:ok=analyze(out,p)
    raise SystemExit(0 if ok else 2)


if __name__=='__main__':main()
