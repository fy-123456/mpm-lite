"""Frozen v9 experiment: variational projected center history versus v7/v8.

Run freeze, tests, probes, run, analyze in that order. Existing records are
never overwritten; each subprocess uses a single CPU thread.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import warp as wp
from demos.aniso import Config, Scene
from benchmarks import aniso_mainline as base
from benchmarks import aniso_affine_consistency as prior
from benchmarks.aniso_flip_time import scaled_flip
from benchmarks.aniso_apic_frequency import Oracle
from engine.aniso_phase1.projected_history import frozen_maps
from engine.aniso_phase1.diagnostics import EnergyLedger, center_snapshot
from engine.aniso_phase1.transfer_audit import center_weights, average, tensor_rms
from engine.aniso_phase1.constitutive import pk1
from engine.aniso_phase1.tensile import grid_values
from engine.sp_grid import B
from utils.resource_guard import prepare_warp_cache, inspect_storage

ROOT=base.ROOT
DEFAULT=ROOT/'docs/results/lite-aniso-mainline/v9'
LEVELS={'coarse':.001,'fine':.0005,'finest':.00025,'fourth':.000125}
TESTS=prior.TESTS+['tests.test_aniso_resample_diagnostic','tests.test_aniso_projected_history']


def write(path,value):
    base.write_json(path,value)


def hashes():
    files=set(base.hashes())
    files.update(str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_aniso_*.py'))
    files.add('benchmarks/aniso_projected_history.py')
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sorted(files)}


def check_archive():
    p=ROOT/'docs/results/lite-aniso-mainline/v8/artifact-sha256.json'
    manifest=json.loads(p.read_text())
    bad=[f for f,h in manifest.items() if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h]
    if bad:raise RuntimeError('v8 archive changed: '+str(bad))
    return {'files':len(manifest),'manifest_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve protocol')
    cfg=json.loads((ROOT/'docs/results/lite-aniso-mainline/v8/cases/incremental-fourth/config.json').read_text())
    configs={}
    for level,dt in LEVELS.items():
        configs['projected-'+level]={**cfg,'dt':dt,'flip_ratio':scaled_flip(dt),'history_consistency':'projected_center'}
    configs['anchor-coarse']={**cfg,'dt':.001,'flip_ratio':.9,'history_consistency':'standard'}
    write(out/'protocol.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),source_sha256=hashes(),
        configs=configs,device='cpu',precision='float64',duration=.5,snapshot_times=[.05,.1,.25,.5],
        tests=TESTS,cuda={'status':'not_run','reason':'controlled CPU float64 comparison'},
        environment={'python':sys.version,'numpy':np.__version__,'warp':wp.__version__,'platform':platform.platform()},
        archive=check_archive(),max_cpu_processes=2,total_steps=8000,
        acceptance={'force_difference_max':.05,'force_floor_N':.001,'interval':[.05,.5],
          'absolute_difference_must_decrease':True,'force_balance_max_N':1e-7,'oracle_max':1e-12,
          'same_state_eta_reduction_target':.5,'anchor_max':1e-12,
          'reliable_time_trend':'last two observed orders >= 0.5; a screening criterion, not a proof of asymptotic convergence'},
        scope='one variational center-map intervention; original particle/APIC transfers, mass, geometry and constitutive law retained'))


def verify(protocol):
    if hashes()!=protocol['source_sha256']:raise RuntimeError('frozen v9 source changed')
    check_archive()


class HistoryLedger(EnergyLedger):
    def __init__(self):
        super().__init__();self.audit_next=False

    def begin(self,s):
        super().begin(s)
        if self.audit_next:
            self.before={k:getattr(s,'ptc_'+k).numpy().copy() for k in ('x','F','v','C','A0','vol0','m')}

    def finish(self,s):
        super().finish(s)
        if not self.audit_next:return
        coords,V,_,active=center_snapshot(s)
        Fc=s.aniso_committed_F[:,:int(s.bcn)].numpy()[0].reshape(-1,3,3)[active].copy()
        Lc=s.center_G[:int(s.bcn)].numpy().reshape(-1,3,3)[active].copy()
        x,F,L=s.ptc_x.numpy().copy(),s.ptc_F.numpy().copy(),s.ptc_L.numpy().copy()
        W,_=center_weights(self.before['x'],self.before['vol0'],coords,s.dx)
        W1,V1=center_weights(x,self.before['vol0'],coords,s.dx)
        oldF=average(W,self.before['F']);oldL=average(W,L)
        cov=s.dt*(average(W,L@self.before['F'])-oldL@oldF)
        grad=s.dt*(oldL-Lc)@oldF
        frozen=average(W,F);rebuilt=average(W1,F);moving=rebuilt-frozen
        A=s.aniso_A0[:,:int(s.bcn)].numpy()[0].reshape(-1,3,3)[active].copy()
        P=lambda a:np.array([pk1(f,t,s.aniso_params) for f,t in zip(a,A)])
        self.audit=dict(time=s.sim_time,history_consistency=getattr(s,'history_consistency','standard'),
            frozen_F_max=float(np.max(abs(frozen-Fc))),frozen_eta=tensor_rms(frozen-Fc,V)/s.dt,
            total_eta=tensor_rms(rebuilt-Fc,V)/s.dt,moving_eta=tensor_rms(moving,V)/s.dt,
            baseline_covariance_eta=tensor_rms(cov,V)/s.dt,baseline_gradient_eta=tensor_rms(grad,V)/s.dt,
            particle_F_update_max=float(np.max(abs(F-(np.eye(3)+s.dt*L)@self.before['F']))),
            moving_decomposition_max=float(np.max(abs(rebuilt-Fc-(frozen-Fc)-moving))),
            rebuilt_stress_difference_rms_Pa=tensor_rms(P(rebuilt)-P(Fc),V))
        self.snapshot=dict(coords=coords,volume=V,new_volume=V1,center_F_committed=Fc,center_G=Lc,
            center_A0=A,particle_x_before=self.before['x'],particle_F_before=self.before['F'],
            particle_velocity_before=self.before['v'],particle_C_before=self.before['C'],
            particle_x_after=x,particle_F_after=F,particle_L_after=L,particle_C_after=s.ptc_C.numpy().copy(),
            particle_velocity_after=s.ptc_v.numpy().copy(),particle_A0=self.before['A0'],
            particle_volume=self.before['vol0'],particle_mass=self.before['m'],grid_nodes=self.nodes.copy(),
            grid_velocity_raw=self.raw_v.copy(),grid_velocity_new=grid_values(s,s.grid_v_new,self.nodes),
            F_frozen=frozen,F_rebuilt=rebuilt,covariance=cov,gradient=grad,moving=moving)


def worker(out,p,name):
    dest=out/'cases'/name
    if dest.exists():raise RuntimeError('preserve '+str(dest))
    dest.mkdir(parents=True)
    cfg=p['configs'][name];write(dest/'config.json',cfg)
    count=round(p['duration']/cfg['dt']);status=dict(run_completed=False,requested_steps=count,completed_steps=0,error=None)
    start=time.monotonic();scene=None;frames=[];Fs=[];times=[]
    try:
        scene=Scene(Config(**cfg),p['device']);s=scene.solver;s.energy_ledger=HistoryLedger()
        audit_steps={round(t/cfg['dt']) for t in p['snapshot_times']}
        frames.append(s.ptc_x.numpy().copy());Fs.append(s.ptc_F.numpy().copy());times.append(0.)
        with (dest/'steps.jsonl').open('x',buffering=1) as log:
            for step in range(1,count+1):
                prepare_warp_cache('/tmp/mpm-lite-warp-cache')
                s.energy_ledger.audit_next=step in audit_steps
                if not scene.step():raise RuntimeError(f'step {step} failed without commit')
                row=scene.metrics();nodes=s.energy_ledger.nodes
                v=grid_values(s,s.grid_v_new,nodes);grip=(nodes[:,0]*s.dx<=.25)|(nodes[:,0]*s.dx>=.75)
                expected=np.zeros_like(v);expected[nodes[:,0]*s.dx>=.75,0]=scene.loading_rows[-1]['loading_velocity']
                row.update(min_particle_det_F=float(np.min(np.linalg.det(s.ptc_F.numpy()))),
                    grid_grip_velocity_error=float(np.max(abs(v[grip]-expected[grip]))))
                log.write(json.dumps(row,allow_nan=False)+'\n');status['completed_steps']=step
                if s.energy_ledger.audit_next:
                    write(dest/f'audit-{step:05d}.json',s.energy_ledger.audit)
                    np.savez_compressed(dest/f'audit-{step:05d}.npz',**s.energy_ledger.snapshot)
                if step % max(1,count//20)==0:
                    frames.append(s.ptc_x.numpy().copy());Fs.append(s.ptc_F.numpy().copy());times.append(s.sim_time)
                if step % 200==0:print(name,step,'/',count,flush=True)
        status.update(run_completed=True,sim_time=s.sim_time,last_step_stats=s.last_step_stats)
    except Exception:
        status['error']=traceback.format_exc();print(status['error'],flush=True)
    finally:
        status['wall_seconds']=time.monotonic()-start;write(dest/'status.json',status)
        np.savez_compressed(dest/'frames.npz',time=times,x=frames,F=Fs)
        if scene is not None:(dest/'trajectory.csv').write_text(scene.solver.energy_ledger.csv())
    return status['run_completed']


def probes(out,p):
    if (out/'same-state.json').exists():raise RuntimeError('preserve probes')
    records=[]
    for level,dt in LEVELS.items():
        version='v8' if level=='fourth' else 'v7'
        folder=ROOT/f'docs/results/lite-aniso-mainline/{version}/cases/incremental-{level}'
        for path in sorted(folder.glob('audit-*.npz')):
            with np.load(path) as z:
                o=Oracle(z['particle_x_before'],z['particle_mass'],.125)
                maps,F0,V,W,S,D=frozen_maps(z['particle_x_before'],z['particle_volume'],z['particle_F_before'],o.c,o.nodes,.125)
                node_lookup={tuple(n):i for i,n in enumerate(z['grid_nodes'])}
                v=z['grid_velocity_new'][[node_lookup[tuple(n)] for n in o.nodes]]
                trial=F0+dt*np.stack([m@v for m in maps],axis=2)
                fp=z['particle_F_after'];mean=np.asarray(W@fp.reshape(len(fp),9)).reshape(-1,3,3)
                c_lookup={tuple(c):i for i,c in enumerate(z['coords'])}
                old=z['center_F_committed'][[c_lookup[tuple(c)] for c in o.c]]
                W1,V1=center_weights(z['particle_x_after'],z['particle_volume'],o.c,.125)
                rebuilt=average(W1,fp)
                eta_old=tensor_rms(rebuilt-old,V)/dt;eta_new=tensor_rms(rebuilt-trial,V)/dt
                records.append(dict(level=level,snapshot=path.name,time=dt*int(path.stem.split('-')[-1]),
                    fixed_weight_max=float(np.max(abs(trial-mean))),baseline_total_eta=eta_old,
                    candidate_total_eta=eta_new,ratio=eta_new/max(eta_old,1e-30)))
    passed=len(records)==16 and all(r['fixed_weight_max']<=1e-12 and r['ratio']<=.5 for r in records)
    write(out/'same-state.json',dict(passed=passed,records=records,note='identical saved velocity/geometry; not a new force solve'))
    return passed


def run(out,p,jobs):
    if not json.loads((out/'tests.json').read_text())['required_checks_passed']:raise RuntimeError('tests must pass')
    if not json.loads((out/'same-state.json').read_text())['passed']:raise RuntimeError('same-state checks must pass')
    def launch(name):
        with (out/(name+'.log')).open('x') as log:
            r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_projected_history','worker','--case',name,'--output',str(out)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        print(name,'exit',r.returncode,flush=True);return dict(case=name,exit_code=r.returncode)
    names=['projected-fourth','anchor-coarse','projected-finest','projected-coarse','projected-fine']
    with ThreadPoolExecutor(max_workers=jobs) as pool:result=list(pool.map(launch,names))
    write(out/'batch.json',result)
    return all(r['exit_code']==0 for r in result)


def read_case(folder):
    status=json.loads((folder/'status.json').read_text())
    rows=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()]
    check=dict(completed=status['run_completed'],steps=len(rows),
        min_particle_J=min(r['min_particle_det_F'] for r in rows),min_center_J=min(r['min_det_F'] for r in rows),
        max_particle_force_error_N=max(r['particle_momentum_balance_error_norm'] for r in rows),
        max_grid_force_error_N=max(abs(r['momentum_balance_error']) for r in rows),
        max_particle_grid_gap=max(r['particle_grid_momentum_gap_norm'] for r in rows),
        max_grip_velocity_error=max(r['grid_grip_velocity_error'] for r in rows),
        max_grip_displacement_error=max(abs(r['measured_grip_displacement']-r['displacement']) for r in rows),
        all_residual_targets_pass=all(r['last_residual_norm']<=r['newton_residual_target'] for r in rows),
        max_budget_closure=max(abs(r['budget_closure']) for r in rows))
    check['passed']=bool(check['completed'] and check['min_particle_J']>0 and check['min_center_J']>0
        and check['max_particle_force_error_N']<=1e-7 and check['max_grid_force_error_N']<=1e-7
        and check['max_particle_grid_gap']<=1e-14 and check['max_grip_velocity_error']<=1e-12
        and check['max_grip_displacement_error']<=.001 and check['all_residual_targets_pass'])
    return rows,check


def analyze(out,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data={};checks={};audits=[]
    for name in p['configs']:
        data[name],checks[name]=read_case(out/'cases'/name)
        audits.extend(json.loads(f.read_text())|{'case':name} for f in sorted((out/'cases'/name).glob('audit-*.json')))
    for level in LEVELS:
        version='v8' if level=='fourth' else 'v7'
        data['baseline-'+level],_=read_case(ROOT/f'docs/results/lite-aniso-mainline/{version}/cases/incremental-{level}')
    t=np.array([r['time'] for r in data['baseline-coarse']]);t=t[(t>=.05-1e-12)&(t<=.5+1e-12)]
    rms=lambda a:float(np.sqrt(np.trapezoid(a*a,t)/(t[-1]-t[0])))
    trends={}
    for mode in ('baseline','projected'):
        trends[mode]={}
        for key in ('right_force','right_elastic_force','right_inertial_force'):
            curves=[np.interp(t,[r['time'] for r in data[mode+'-'+l]],[r[key] for r in data[mode+'-'+l]]) for l in LEVELS]
            pairs=[]
            for i in range(3):
                err=rms(curves[i]-curves[i+1]);r=dict(pair=list(LEVELS)[i]+'-'+list(LEVELS)[i+1],absolute_rms_N=err,relative=err/max(rms(curves[i+1]),.001))
                if i:
                    rho=err/pairs[-1]['absolute_rms_N'];r.update(rho=rho,observed_order=float(-np.log2(rho)))
                pairs.append(r)
            trends[mode][key]=pairs
    anchor=max(abs(a['right_force']-b['right_force']) for a,b in zip(data['anchor-coarse'],data['baseline-coarse']))
    last=trends['projected']['right_force'][-1]
    time_ok=last['relative']<=.05 and last['rho']<1
    reliable=all(r['observed_order']>=.5 for r in trends['projected']['right_force'][1:])
    diag_ok=all(r['particle_F_update_max']<=1e-12 and r['moving_decomposition_max']<=1e-12 and
        (not r['case'].startswith('projected') or r['frozen_F_max']<=1e-12) for r in audits)
    result=dict(run_completed=all(c['completed'] for c in checks.values()),checks=checks,refinement=trends,
        anchor_max_N=anchor,anchor_passed=anchor<=1e-12,diagnostics_passed=diag_ok,audits=audits,
        physical_checks_passed=all(c['passed'] for c in checks.values()),time_threshold_passed=time_ok,
        reliable_time_trend_screen_passed=reliable,default_changed=False,spatial_accuracy='separate experiment',cuda='not_run')
    result['all_frozen_numeric_checks_passed']=bool(result['physical_checks_passed'] and diag_ok and time_ok and anchor<=1e-12)
    write(out/'summary.json',result)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for l in LEVELS:
        for ax,key in zip(axes,('right_force','right_elastic_force','right_inertial_force')):
            rows=data['projected-'+l];ax.plot([r['time'] for r in rows],[r[key] for r in rows],label=l)
            ax.set(title=key,xlabel='time (s)',ylabel='reaction (N)');ax.grid(alpha=.25)
    for ax in axes:ax.legend()
    fig.savefig(out/'projected-reactions.png',dpi=160);plt.close(fig)
    print(json.dumps({k:result[k] for k in ('time_threshold_passed','reliable_time_trend_screen_passed','physical_checks_passed','anchor_max_N')},indent=2))
    return result['all_frozen_numeric_checks_passed']


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=('freeze','tests','probes','run','worker','analyze'))
    parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--case')
    parser.add_argument('--jobs',type=int,choices=(1,2),default=2)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache')
    if args.action=='freeze':freeze(out);return
    p=json.loads((out/'protocol.json').read_text());verify(p)
    if args.action=='tests':ok=base.tests(out,p)
    elif args.action=='probes':ok=probes(out,p)
    elif args.action=='worker':ok=worker(out,p,args.case)
    elif args.action=='run':ok=run(out,p,args.jobs)
    else:ok=analyze(out,p)
    raise SystemExit(0 if ok else 2)


if __name__=='__main__':main()
