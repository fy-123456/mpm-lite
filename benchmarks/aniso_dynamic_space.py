"""Bounded finite-strain space/time separation for ISO/F0/F45/F90.

Two physical grids and two dt per grid; full 0.05 s monotone loading to 5 mm.
This is a faster dynamic load than the 0.5 s F45 temporal study, never merged
with that study. Particle and quadrature sweeps are handled by static_space.
"""
import argparse
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
from benchmarks.aniso_flip_time import scaled_flip
from benchmarks.aniso_mainline import write_json
from engine.aniso_phase1.consistent_transfer import material_response
from demos.aniso import Config

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'docs/results/lite-aniso-mainline/v9-dynamic-space'
CASES={'ISO':(0.,0.),'F0':(200.,0.),'F45':(200.,45.),'F90':(200.,90.)}


def hashes():
    names=['benchmarks/aniso_dynamic_space.py','benchmarks/aniso_projected_history.py',
        'engine/aniso_phase1/projected_history.py','engine/aniso_phase1/solver.py',
        'engine/aniso_phase1/tensile.py','demos/aniso.py']
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in names}


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve protocol')
    source=ROOT/'docs/results/lite-aniso-mainline/v9'
    prior=json.loads((source/'protocol.json').read_text())
    assert json.loads((source/'tests.json').read_text())['required_checks_passed']
    configs={}
    for label,(kf,angle) in CASES.items():
        for grid in (9,17):
            for level,dt in [('coarse',.0005),('fine',.00025)]:
                cfg=prior['configs']['projected-coarse'].copy()
                cfg.update(grid=grid,kf=kf,fiber_angle=angle,dt=dt,flip_ratio=scaled_flip(dt),
                           loading_time=.05,loading_speed=.1)
                configs[f'{label}-g{grid}-{level}']=cfg
    write_json(out/'protocol.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),source_sha256=hashes(),
        configs=configs,device='cpu',precision='float64',duration=.05,snapshot_times=[.005,.01,.025,.05],
        endpoint_displacement_m=.005,run_count=16,total_steps=2400,jobs=1,
        tests_reused={'path':str(source/'tests.json'),'count':69,'source_checked':True},
        thresholds={'time_reaction_relative':.02,'time_F_P_energy_relative':.02,'force_balance_N':1e-7},
        space_metric='same 192 coarse-grid material sample positions; trilinear interpolation of fine-grid particle fields; finite sampling diagnostic, not a continuum norm proof',
        scope='finite-strain dynamic complement; full monotone 0.05 s load; one h refinement; no asymptotic space-order claim'))


def run(out,p):
    records=[]
    for name in p['configs']:
        with (out/(name+'.log')).open('x') as log:
            r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_dynamic_space','worker','--output',str(out),'--case',name],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        records.append(dict(case=name,exit_code=r.returncode));print(name,'exit',r.returncode,flush=True)
    write_json(out/'batch.json',records)
    return all(r['exit_code']==0 for r in records)


def frame(folder):
    with np.load(folder/'frames.npz') as z:return z['x'][0].copy(),z['F'][-1].copy()


def stress(F,cfg):
    params=Config(**cfg).params;A=np.broadcast_to(params.A0,F.shape)
    return material_response(F,A,params)[1]


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-20))


def curve_difference(a,b):
    t=np.array([r['time'] for r in a]);t=t[(t>=.005-1e-12)&(t<=.05+1e-12)]
    sample=lambda rows:np.interp(t,[r['time'] for r in rows],[r['right_force'] for r in rows])
    ca,cb=sample(a),sample(b);rms=lambda c:float(np.sqrt(np.trapezoid(c*c,t)/(t[-1]-t[0])))
    return dict(absolute_RMS_N=rms(ca-cb),relative=rms(ca-cb)/max(rms(cb),.001))


def analyze(out,p):
    rows={};checks={};times=[];spaces=[]
    for name in p['configs']:rows[name],checks[name]=read_case(out/'cases'/name)
    for label in CASES:
        for grid in (9,17):
            coarse=f'{label}-g{grid}-coarse';fine=f'{label}-g{grid}-fine'
            _,Fc=frame(out/'cases'/coarse);_,Ff=frame(out/'cases'/fine)
            pc,pf=stress(Fc,p['configs'][coarse]),stress(Ff,p['configs'][fine])
            r=dict(case=label,grid=grid,reaction=curve_difference(rows[coarse],rows[fine]),
                F_minus_I_relative=relative(Fc-np.eye(3),Ff-np.eye(3)),P_relative=relative(pc,pf),
                energy_relative=abs(rows[coarse][-1]['elastic']-rows[fine][-1]['elastic'])/max(abs(rows[fine][-1]['elastic']),1e-20))
            r['passed']=r['reaction']['relative']<=.02 and max(r['F_minus_I_relative'],r['P_relative'],r['energy_relative'])<=.02
            times.append(r)
        coarse=f'{label}-g9-fine';fine=f'{label}-g17-fine'
        Xc,Fc=frame(out/'cases'/coarse);Xf,Ff=frame(out/'cases'/fine)
        axes=tuple(np.unique(Xf[:,d]) for d in range(3));shape=tuple(map(len,axes))
        mapped=RegularGridInterpolator(axes,Ff.reshape(shape+(3,3)),bounds_error=True)(Xc)
        pc,pf=stress(Fc,p['configs'][coarse]),stress(mapped,p['configs'][fine])
        spaces.append(dict(case=label,reaction=curve_difference(rows[coarse],rows[fine]),
            F_minus_I_relative=relative(Fc-np.eye(3),mapped-np.eye(3)),P_relative=relative(pc,pf),
            energy_relative=abs(rows[coarse][-1]['elastic']-rows[fine][-1]['elastic'])/max(abs(rows[fine][-1]['elastic']),1e-20),
            time_separation_passed=all(t['passed'] for t in times if t['case']==label),
            interpolation_points=len(Xc)))
    result=dict(completed=all(c['completed'] for c in checks.values()),checks=checks,time_checks=times,space_checks=spaces,
        physical_checks_passed=all(c['passed'] for c in checks.values()),
        time_separation_passed=all(r['passed'] for r in times),space_convergence_established=False,
        scope=p['scope'],cuda='not_run',default_changed=False)
    write_json(out/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,label in zip(axes.flat,CASES):
        for grid in (9,17):
            for level in ('coarse','fine'):
                r=rows[f'{label}-g{grid}-{level}']
                ax.plot([x['time'] for x in r],[x['right_force'] for x in r],label=f'g{grid} {level}',ls='--' if level=='coarse' else '-')
        ax.set(title=label,xlabel='time (s)',ylabel='reaction (N)');ax.legend();ax.grid(alpha=.25)
    fig.savefig(out/'dynamic-directions.png',dpi=160);plt.close(fig)
    print(json.dumps({k:result[k] for k in ('completed','physical_checks_passed','time_separation_passed')},indent=2))
    # A failed accuracy gate is an experimental result, not a missing trajectory.
    return result['physical_checks_passed'] and result['time_separation_passed']


def main():
    import warp as wp
    wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('freeze','run','worker','analyze'))
    parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--case');a=parser.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    if a.action=='freeze':freeze(out);return
    p=json.loads((out/'protocol.json').read_text())
    if p['source_sha256']!=hashes():raise RuntimeError('frozen dynamic spatial source changed')
    ok=worker(out,p,a.case) if a.action=='worker' else run(out,p) if a.action=='run' else analyze(out,p)
    raise SystemExit(0 if ok else 2)


if __name__=='__main__':main()
