"""Validate physical-time FLIP scaling against immutable fixed-beta F45 records."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import warp as wp
from benchmarks import aniso_mainline as baseline
from benchmarks import aniso_f45_refinement as previous
from utils.resource_guard import prepare_warp_cache,inspect_storage

ROOT=baseline.ROOT
DEFAULT=ROOT/'docs/results/lite-aniso-mainline/v5'
REFERENCE=previous.DEFAULT
TESTS=previous.TESTS+['tests.test_aniso_flip_time']


def scaled_flip(dt,beta0=.9,dt0=.001):
    """Match the reference per-step exponential mixing over physical time."""
    if not all(math.isfinite(x) for x in (dt,beta0,dt0)) or dt<=0 or dt0<=0 or not 0<=beta0<=1:
        raise ValueError('positive finite dt/dt0 and beta0 in [0,1] required')
    return beta0**(dt/dt0)


def hashes():
    result=previous.hashes()
    for name in ('benchmarks/aniso_flip_time.py','tests/test_aniso_flip_time.py'):
        result[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


def verify_reference():
    old=json.loads((REFERENCE/'protocol.json').read_text());previous.verify(old)
    manifest=json.loads((REFERENCE/'artifact-sha256.json').read_text())
    changed=[name for name,h in manifest.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h]
    if changed:raise RuntimeError('v4 reference artifacts changed: '+str(changed))
    return len(manifest)


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve previous protocol')
    count=verify_reference();old=json.loads((REFERENCE/'protocol.json').read_text())
    configs={}
    for level,dt in previous.LEVELS.items():
        cfg=old['configs']['baseline-'+level].copy();cfg['flip_ratio']=scaled_flip(dt)
        configs['calibrated-'+level]=cfg
    p=dict(name='lite-f45-physical-time-flip-v5',frozen_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashes(),configs=configs,modes={name:'baseline' for name in configs},
        device='cpu',precision='float64',duration=.5,endpoint_displacement_m=.005,
        beta_rule='beta(dt)=0.9**(dt/0.001)',reference_dt=.001,reference_beta=.9,
        tests=TESTS,snapshot_times=old['snapshot_times'],run_count=3,total_steps=3500,
        reference=str(REFERENCE.relative_to(ROOT)),reference_files_verified=count,
        reference_manifest_sha256=hashlib.sha256((REFERENCE/'artifact-sha256.json').read_bytes()).hexdigest(),
        acceptance={'time_interval':[.05,.5],'relative_force_difference_max':.05,'force_rms_floor_N':.001,
                    'trend':'fine-finest absolute RMS difference smaller than coarse-fine',
                    'coarse_reference_max':1e-12,'force_error_max_N':1e-7,
                    'particle_grid_gap_max_kg_m_s':1e-14,'grid_grip_velocity_max_m_s':1e-12,
                    'particle_grip_error_max_m':.001,'snapshot_identity_max':1e-12,'all_tests_no_skips':True},
        cuda={'status':'not_run','reason':'CPU float64 controlled validation'},
        resource={'max_cpu_processes':2,'threads_each':1,'storage':asdict(inspect_storage())},
        scope='only Config.flip_ratio changes; original gradient/history/forces/APIC/position/loading stay unchanged; reuse verified v4 fixed-beta cases')
    baseline.write_json(out/'protocol.json',p)


def verify(p):
    current=hashes();changed=[name for name,h in p['source_sha256'].items() if current.get(name)!=h]
    if changed:raise RuntimeError('frozen source changed: '+str(changed))
    if hashlib.sha256((REFERENCE/'artifact-sha256.json').read_bytes()).hexdigest()!=p['reference_manifest_sha256']:
        raise RuntimeError('reference manifest changed')


def run(out,p,jobs):
    if not json.loads((out/'tests.json').read_text())['required_checks_passed']:raise RuntimeError('tests must pass')
    verify_reference()
    def launch(name):
        log=out/(name+'.log')
        if log.exists() or (out/'cases'/name).exists():raise RuntimeError('preserve existing case '+name)
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_flip_time','worker','--output',str(out),'--case',name],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        print(name,'exit',result.returncode,flush=True);return dict(case=name,exit_code=result.returncode)
    with ThreadPoolExecutor(max_workers=jobs) as pool:records=list(pool.map(launch,p['configs']))
    baseline.write_json(out/'batch.json',records)
    return all(r['exit_code']==0 for r in records)


def analyze(out,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verify_reference()
    data,checks,audits=previous.read_cases(out,p['configs'])
    ref,_,_=previous.read_cases(REFERENCE,['baseline-'+l for l in previous.LEVELS])
    fixed={'fixed-'+name.split('-')[-1]:rows for name,rows in ref.items()}
    trends={'fixed':previous.refinement(fixed,'fixed'),'calibrated':previous.refinement(data,'calibrated')}
    anchor={};energy={};effects={}
    if 'calibrated-coarse' in data:
        a=data['calibrated-coarse'];b=ref['baseline-coarse']
        anchor['reaction_max_error_N']=max(abs(x['right_force']-y['right_force']) for x,y in zip(a,b))
        with np.load(out/'cases/calibrated-coarse/frames.npz') as x,np.load(REFERENCE/'cases/baseline-coarse/frames.npz') as y:
            anchor['x_max_error']=float(np.max(abs(x['x']-y['x'])))
            anchor['F_max_error']=float(np.max(abs(x['F']-y['F'])))
        anchor['passed']=all(v<=1e-12 for v in anchor.values())
    for name,rows in {**fixed,**data}.items():
        signed={k:sum(r[k] for r in rows) for k in ('p2c_delta','c2g_delta','g2p_delta','state_transport_delta')}
        energy[name]=dict(signed_cumulative_J=signed,kinetic_transfer_net_J=sum(signed[k] for k in ('p2c_delta','c2g_delta','g2p_delta')),loading_work_J=rows[-1]['loading_work'])
    for level in previous.LEVELS:
        key='calibrated-'+level
        if key not in data:continue
        rows=data[key];t=np.array([r['time'] for r in rows]);mask=t>=.05-1e-12;t=t[mask]
        a=np.array([r['right_force'] for r in rows])[mask];b=np.array([r['right_force'] for r in fixed['fixed-'+level]])[mask]
        rms=lambda a:float(np.sqrt(np.trapezoid(a*a,t)/(t[-1]-t[0])))
        effects[level]=dict(reaction_change_rms_N=rms(a-b),relative_to_fixed=rms(a-b)/max(rms(b),.001))
    audit_checks={name:len(rows)==4 and all(r['frozen_decomposition_max_error']<=1e-12 and r['pic_advection_oracle_max_error']<=1e-12 for r in rows) for name,rows in audits.items()}
    new=trends['calibrated'];complete=len(data)==3
    time_pass=bool(new['complete'] and new['components']['right_force']['passes_5_percent'])
    decreasing=bool(new['complete'] and new['components']['right_force']['decreasing'])
    summary=dict(experiment_completed=complete,required_checks_passed=json.loads((out/'tests.json').read_text())['required_checks_passed'],
        checks=checks,refinement=trends,anchor=anchor,audits=audits,audit_checks=audit_checks,
        physical_checks_passed=complete and all(c['passed'] for c in checks.values()),
        diagnostics_passed=len(audit_checks)==3 and all(audit_checks.values()),
        time_sensitivity_passed=time_pass,decreasing_difference=decreasing,
        same_dt_effect=effects,energy=energy,production_default_changed=False,spatial_accuracy='not_verified',cuda='not_run')
    summary['all_acceptance_passed']=bool(summary['required_checks_passed'] and summary['physical_checks_passed'] and summary['diagnostics_passed'] and anchor.get('passed',False) and time_pass and decreasing)
    baseline.write_json(out/'summary.json',summary)
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for i,(mode,group) in enumerate((('fixed',fixed),('calibrated',data))):
        for level in previous.LEVELS:
            rows=group.get(mode+'-'+level)
            if not rows:continue
            for j,key in enumerate(('right_force','right_elastic_force','right_inertial_force')):
                axes[i,j].plot([r['displacement'] for r in rows],[r[key] for r in rows],label=level)
        for j,title in enumerate(('total','elastic','inertial')):
            axes[i,j].set(xlabel='command displacement (m)',ylabel='reaction (N)',title=mode+' / '+title)
            axes[i,j].legend();axes[i,j].grid(alpha=.25)
    fig.savefig(out/'flip-time-reaction.png',dpi=160);plt.close(fig)
    return summary['all_acceptance_passed']


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('freeze','tests','run','worker','analyze'))
    parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--jobs',type=int,choices=(1,2),default=2);parser.add_argument('--case')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache')
    if args.action=='freeze':freeze(out);return
    p=json.loads((out/'protocol.json').read_text());verify(p)
    if args.action=='tests':ok=baseline.tests(out,p)
    elif args.action=='run':ok=run(out,p,args.jobs)
    elif args.action=='worker':ok=previous.worker(out,p,args.case)
    else:ok=analyze(out,p)
    raise SystemExit(0 if ok else 2)


if __name__=='__main__':main()
