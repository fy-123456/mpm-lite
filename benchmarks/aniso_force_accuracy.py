"""Bounded force-tolerance, frozen-transfer, and equal-displacement loading audit."""
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
from demos.aniso import Config,Scene
from engine.aniso_phase1.transfer_audit import TransferAuditLedger
from utils.resource_guard import prepare_warp_cache,inspect_storage

ROOT=baseline.ROOT
DEFAULT=ROOT/'docs/results/lite-aniso-mainline/v2'
TESTS=baseline.TESTS+['tests.test_aniso_force_accuracy']


def hashes():
    result=baseline.hashes()
    for name in ('benchmarks/aniso_force_accuracy.py','tests/test_aniso_force_accuracy.py'):
        result[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


def verify(p):
    current=hashes()
    changed=[name for name,h in p['source_sha256'].items() if current.get(name)!=h]
    if changed:raise RuntimeError('frozen source changed: '+str(changed))


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('protocol exists; preserve prior results')
    configs={}
    for rate,T,speed in [('fast',.5,.01),('slow',1.,.005)]:
        for case,(kf,angle) in baseline.CASES.items():
            for level,dt in [('coarse',.001),('fine',.0005)]:
                configs[f'{rate}-{case}-{level}']=asdict(replace(baseline.BASE,
                    kf=kf,fiber_angle=angle,dt=dt,loading_time=T,loading_speed=speed,reaction_force_atol=1e-7))
    p=dict(name='lite-aniso-force-accuracy-v2',frozen_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashes(),configs=configs,device='cpu',precision='float64',tests=TESTS,
        baseline='docs/results/lite-aniso-mainline/v1',
        baseline_summary_sha256=hashlib.sha256((baseline.DEFAULT_OUTPUT/'summary.json').read_bytes()).hexdigest(),
        acceptance={'time_error_max':.05,'force_rms_floor_N':.001,'phase_interval':[.1,1.],
                    'momentum_balance_max_N':1e-7,'grid_grip_velocity_max_m_s':1e-12,
                    'particle_grip_error_max_m':.001,'direction_factor':2.,
                    'snapshot_identity_max':1e-12,'required_tests_no_skips':True},
        force_stopping={'resultant_atol_N':1e-7,'node_bound':'all active nodes >= free nodes',
                        'nonlinear_cap':'0.5*dt*epsilon_R/sqrt(N_active)',
                        'linear_cap':'0.1*effective_nonlinear_target; cap both relative and absolute CG tolerances',
                        'small_update_cannot_bypass':True},
        diagnostics={'all_cases':'stage linear momentum and existing energy ledger',
                     'F45_snapshot_phases':[.2,.5,1.],
                     'frozen_identity':'W Fnew-Fcenter_new = dt*(W(Gp Fp)-W(Gp)W(Fp)) + dt*(W(Gp)-Gc)W(Fp)',
                     'scope':'center quadrature, uniform A0, particle_resample; no solver substitution'},
        resource={'max_cpu_processes':3,'threads_each':1,'storage':asdict(inspect_storage())},
        cuda={'status':'not_run','reason':'only visible A100 occupied; CPU experiment'},
        run_count=16,total_steps=18000,endpoint_displacement_m=.005)
    baseline.write_json(out/'protocol.json',p)


def worker(out,p,name):
    config=p['configs'][name];dest=out/'cases'/name
    class AuditedScene(Scene):
        def __init__(self,config,device):
            super().__init__(config,device)
            self.solver.energy_ledger=TransferAuditLedger()
            self.audit_steps={round(config.loading_time*phase/config.dt) for phase in (.2,.5,1.)} if config.fiber_angle==45 else set()
        def step(self):
            ledger=self.solver.energy_ledger
            ledger.audit_next=self.solver.sim_steps+1 in self.audit_steps
            success=super().step()
            if success and ledger.audit_next:
                step=self.solver.sim_steps
                np.savez_compressed(dest/f'audit-{step:05d}.npz',**ledger.snapshot)
                baseline.write_json(dest/f'audit-{step:05d}.json',ledger.audit)
            return success
    baseline.Scene=AuditedScene
    return baseline.run_case(out,name,config,{**p,'duration':config['loading_time']})


def run(out,p,jobs):
    if not json.loads((out/'tests.json').read_text())['required_checks_passed']:raise RuntimeError('tests must pass')
    def launch(name):
        log=out/(name+'.log')
        if log.exists() or (out/'cases'/name).exists():raise RuntimeError('case exists: '+name)
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_force_accuracy','worker',
                '--output',str(out),'--case',name],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        print(name,'exit',result.returncode,flush=True)
        return dict(case=name,exit_code=result.returncode)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        records=list(pool.map(launch,p['configs']))
    baseline.write_json(out/'batch.json',records)
    return all(r['exit_code']==0 for r in records)


def analyze(out,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data={};statuses={};checks={};audits={}
    for name,cfg in p['configs'].items():
        dest=out/'cases'/name
        status=json.loads((dest/'status.json').read_text()) if (dest/'status.json').exists() else {'run_completed':False}
        statuses[name]=status
        if not status['run_completed']:continue
        rows=[json.loads(line) for line in (dest/'steps.jsonl').read_text().splitlines()];data[name]=rows
        checks[name]=dict(max_force_bound_N=max(r['reaction_force_residual_bound'] for r in rows),
            max_momentum_error_N=max(abs(r['momentum_balance_error']) for r in rows),
            max_grid_grip_velocity_error=max(r['grid_grip_velocity_error'] for r in rows),
            max_particle_grip_error_m=max(abs(r['measured_grip_displacement']-r['displacement']) for r in rows),
            min_particle_det_F=min(r['min_particle_det_F'] for r in rows),min_center_det_F=min(r['min_det_F'] for r in rows),
            max_newton_residual=max(r['last_residual_norm'] for r in rows),
            max_newton_iterations=max(r['newton_iterations'] for r in rows),
            all_residuals_below_effective_target=all(r['last_residual_norm']<=r['newton_residual_target'] for r in rows))
        check=checks[name]
        check['passed']=bool(check['max_force_bound_N']<=1e-7 and check['max_momentum_error_N']<=1e-7
            and check['max_grid_grip_velocity_error']<=1e-12 and check['max_particle_grip_error_m']<=.001
            and check['min_particle_det_F']>0 and check['min_center_det_F']>0 and check['all_residuals_below_effective_target'])
        snapshots=[json.loads(f.read_text()) for f in sorted(dest.glob('audit-*.json'))]
        if snapshots:audits[name]=snapshots
    sensitivity={};directions={};baseline_delta={};components={}
    for rate in ('fast','slow'):
        sensitivity[rate]={};directions[rate]={};aligned={};errors={}
        for case in baseline.CASES:
            cn,fn=f'{rate}-{case}-coarse',f'{rate}-{case}-fine'
            if cn not in data or fn not in data:continue
            c,f=data[cn],data[fn];T=p['configs'][cn]['loading_time']
            t=np.array([r['time'] for r in c]);mask=(t>=.1*T-1e-12)&(t<=T+1e-12);t=t[mask]
            rms=lambda x:float(np.sqrt(np.trapezoid(x*x,t)/(t[-1]-t[0])))
            components[rate+'-'+case]={}
            for key in ('right_force','right_elastic_force','right_inertial_force'):
                rf=np.interp(t,[r['time'] for r in f],[r[key] for r in f]);delta=np.array([r[key] for r in c])[mask]-rf
                components[rate+'-'+case][key]={'fine_rms_N':rms(rf),'difference_rms_N':rms(delta),
                    'relative':rms(delta)/max(rms(rf),.001)}
                if key=='right_force':aligned[case]=rf;errors[case]=rms(delta)
            d=components[rate+'-'+case]
            sensitivity[rate][case]={**d['right_force'],'passed':d['right_force']['relative']<=.05,
                'inertial_to_total_rms':d['right_inertial_force']['fine_rms_N']/max(d['right_force']['fine_rms_N'],.001)}
            if rate=='fast':
                old_path=baseline.DEFAULT_OUTPUT/'cases'/f'{case}-fine'/'steps.jsonl'
                if old_path.exists():
                    old=[json.loads(line) for line in old_path.read_text().splitlines()]
                    old_r=np.interp(t,[r['time'] for r in old],[r['right_force'] for r in old])
                    baseline_delta[case]={'reaction_change_rms_N':rms(aligned[case]-old_r),
                        'max_momentum_error_before_N':max(abs(r['momentum_balance_error']) for r in old),
                        'max_momentum_error_after_N':checks[fn]['max_momentum_error_N']}
        import itertools
        for a,b in itertools.combinations(aligned,2):
            difference=rms(aligned[a]-aligned[b]);threshold=2*(errors[a]+errors[b])
            directions[rate][a+'-'+b]={'difference_rms_N':difference,'threshold_N':threshold,'resolved':difference>threshold}
    all_done=len(data)==16 and all(r['run_completed'] for r in statuses.values())
    tests=json.loads((out/'tests.json').read_text())
    audit_pass=len(audits)==4 and all(len(rows)==3 and all(r['resample_oracle_F_max']<=1e-12
        and r['resample_oracle_volume_max']<=1e-12 and r['particle_update_max_error']<=1e-12
        and r['frozen_decomposition_max_error']<=1e-12 for r in rows) for rows in audits.values())
    time_pass={rate:len(values)==4 and all(v['passed'] for v in values.values()) for rate,values in sensitivity.items()}
    summary=dict(run_completed=all_done,required_checks_passed=tests['required_checks_passed'],
        force_checks_passed=len(checks)==16 and all(r['passed'] for r in checks.values()),
        frozen_audits_passed=audit_pass,time_sensitivity_passed=time_pass,cases=statuses,checks=checks,
        time_sensitivity=sensitivity,direction_pairs=directions,tolerance_effect=baseline_delta,
        reaction_components=components,transfer_audits=audits,cuda=p['cuda'],space_accuracy='not_verified')
    summary['all_acceptance_passed']=bool(all_done and summary['required_checks_passed'] and summary['force_checks_passed'] and audit_pass and all(time_pass.values()))
    baseline.write_json(out/'summary.json',summary)
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for ri,rate in enumerate(('fast','slow')):
        for case,color in zip(baseline.CASES,('black','tab:red','tab:orange','tab:blue')):
            for level,style in [('coarse','--'),('fine','-')]:
                rows=data.get(f'{rate}-{case}-{level}')
                if not rows:continue
                for ax,key in zip(axes[ri],('right_force','right_elastic_force','right_inertial_force')):
                    ax.plot([r['displacement'] for r in rows],[r[key] for r in rows],style,color=color,label=f'{case} {level}')
        for ax,title in zip(axes[ri],('Total reaction','Elastic reaction','Inertial reaction')):
            ax.set(xlabel='Command displacement (m)',ylabel='Right reaction (N)',title=f'{rate}: {title}');ax.grid(alpha=.3)
        axes[ri,0].legend(fontsize=7)
    fig.savefig(out/'reaction-rate-comparison.png',dpi=160);plt.close(fig)
    return summary['all_acceptance_passed']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','tests','run','worker','analyze'))
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--jobs',type=int,default=3,choices=(1,2,3))
    parser.add_argument('--case')
    a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache')
    if a.action=='freeze':freeze(out);return
    p=json.loads((out/'protocol.json').read_text());verify(p)
    if a.action=='tests':ok=baseline.tests(out,p)
    elif a.action=='run':ok=run(out,p,a.jobs)
    elif a.action=='worker':ok=worker(out,p,a.case)
    else:ok=analyze(out,p)
    if not ok:raise SystemExit(2 if a.action=='analyze' else 1)


if __name__=='__main__':main()
