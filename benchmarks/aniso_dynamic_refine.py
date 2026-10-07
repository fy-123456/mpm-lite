"""Additional common dt for separating dynamic space and time differences.

The initial 16 trajectories and their failed gates are retained unchanged.
Eight dt=0.125 ms trajectories extend every direction/grid to three levels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from benchmarks.aniso_projected_history import worker,read_case
from benchmarks.aniso_dynamic_space import DEFAULT as SOURCE,CASES,hashes as previous_hashes,frame,stress,relative
from benchmarks.aniso_flip_time import scaled_flip
from benchmarks.aniso_mainline import write_json

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'docs/results/lite-aniso-mainline/v9-dynamic-refined'


def hashes():
    h=previous_hashes();h['benchmarks/aniso_dynamic_refine.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();return h


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve protocol')
    old=json.loads((SOURCE/'protocol.json').read_text());summary=json.loads((SOURCE/'summary.json').read_text())
    if not summary['completed']:raise RuntimeError('initial trajectories must be complete')
    configs={}
    for label in CASES:
        for grid in (9,17):
            cfg=old['configs'][f'{label}-g{grid}-fine'].copy();cfg.update(dt=.000125,flip_ratio=scaled_flip(.000125))
            configs[f'{label}-g{grid}-finest']=cfg
    files={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in SOURCE.rglob('*') if f.is_file()}
    write_json(out/'protocol.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),source_sha256=hashes(),
        configs=configs,reference_sha256=files,device='cpu',precision='float64',duration=.05,snapshot_times=[.005,.01,.025,.05],
        total_steps=3200,jobs=2,thresholds=old['thresholds'],
        reason='Initial two-level time checks exceed 2% in some directions; retain all failures and add one common finer dt for all groups',
        selection='all four directions and both grids; not selected by favorable results',
        initial_time_checks=summary['time_checks'],maximum_additional_levels=1))


def verify(p):
    if p['source_sha256']!=hashes():raise RuntimeError('frozen refinement source changed')
    bad=[f for f,h in p['reference_sha256'].items() if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h]
    if bad:raise RuntimeError('initial results changed: '+str(bad))


def run(out,p):
    def launch(name):
        with (out/(name+'.log')).open('x') as log:
            r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_dynamic_refine','worker','--case',name,'--output',str(out)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        print(name,'exit',r.returncode,flush=True);return dict(case=name,exit_code=r.returncode)
    with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(launch,p['configs']))
    write_json(out/'batch.json',records);return all(r['exit_code']==0 for r in records)


def compare(a,b,t):
    sample=lambda rows:np.interp(t,[r['time'] for r in rows],[r['right_force'] for r in rows])
    x,y=sample(a),sample(b);rms=lambda v:float(np.sqrt(np.trapezoid(v*v,t)/(t[-1]-t[0])))
    return dict(absolute_RMS_N=rms(x-y),relative=rms(x-y)/max(rms(y),.001))


def analyze(out,p):
    data={};checks={};times=[];spaces=[]
    for label in CASES:
        for grid in (9,17):
            names=[f'{label}-g{grid}-{l}' for l in ('coarse','fine','finest')]
            for name,root in zip(names,(SOURCE,SOURCE,out)):data[name],checks[name]=read_case(root/'cases'/name)
            t=np.array([r['time'] for r in data[names[0]]]);t=t[(t>=.005-1e-12)&(t<=.05+1e-12)]
            pairs=[compare(data[names[0]],data[names[1]],t),compare(data[names[1]],data[names[2]],t)]
            rho=pairs[1]['absolute_RMS_N']/pairs[0]['absolute_RMS_N']
            _,F0=frame(SOURCE/'cases'/names[1]);_,F1=frame(out/'cases'/names[2]);cfg=p['configs'][names[2]]
            r=dict(case=label,grid=grid,reaction=pairs[1],previous_reaction=pairs[0],rho=rho,observed_order=float(-np.log2(rho)),
                F_minus_I_relative=relative(F0-np.eye(3),F1-np.eye(3)),P_relative=relative(stress(F0,cfg),stress(F1,cfg)),
                energy_relative=abs(data[names[1]][-1]['elastic']-data[names[2]][-1]['elastic'])/max(abs(data[names[2]][-1]['elastic']),1e-20))
            r['passed']=r['reaction']['relative']<=.02 and max(r['F_minus_I_relative'],r['P_relative'],r['energy_relative'])<=.02
            times.append(r)
        a=f'{label}-g9-finest';b=f'{label}-g17-finest'
        X0,F0=frame(out/'cases'/a);X1,F1=frame(out/'cases'/b)
        axes=tuple(np.unique(X1[:,k]) for k in range(3));mapped=RegularGridInterpolator(axes,F1.reshape(tuple(map(len,axes))+(3,3)),bounds_error=True)(X0)
        t=np.array([r['time'] for r in data[f'{label}-g9-coarse']]);t=t[(t>=.005-1e-12)&(t<=.05+1e-12)]
        cfg=p['configs'][a]
        spaces.append(dict(case=label,reaction=compare(data[a],data[b],t),
            F_minus_I_relative=relative(F0-np.eye(3),mapped-np.eye(3)),P_relative=relative(stress(F0,cfg),stress(mapped,cfg)),
            energy_relative=abs(data[a][-1]['elastic']-data[b][-1]['elastic'])/max(abs(data[b][-1]['elastic']),1e-20),
            time_separation_passed=all(r['passed'] for r in times if r['case']==label),interpolation_points=len(X0)))
    result=dict(completed=all(c['completed'] for c in checks.values()),checks=checks,time_checks=times,space_checks=spaces,
        physical_checks_passed=all(c['passed'] for c in checks.values()),time_separation_passed=all(r['passed'] for r in times),
        space_convergence_established=False,reference_data_unchanged=True,default_changed=False,cuda='not_run',
        added_trajectories=8,total_trajectories=24,total_steps=5600,
        scope='0.05 s complete finite-strain dynamic loading; explicit 2% time gate, one spatial refinement, sampled field differences')
    write_json(out/'summary.json',result)
    print(json.dumps({k:result[k] for k in ('completed','physical_checks_passed','time_separation_passed')},indent=2))
    return result['physical_checks_passed'] and result['time_separation_passed']


def main():
    import warp as wp
    wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('freeze','run','worker','analyze'))
    parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--case');a=parser.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    if a.action=='freeze':freeze(out);return
    p=json.loads((out/'protocol.json').read_text());verify(p)
    ok=worker(out,p,a.case) if a.action=='worker' else run(out,p) if a.action=='run' else analyze(out,p)
    raise SystemExit(0 if ok else 2)


if __name__=='__main__':main()
