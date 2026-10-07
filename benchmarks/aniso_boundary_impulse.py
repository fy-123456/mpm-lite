"""Bounded opt-in boundary impulse validation and manufactured gradient audit."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import warp as wp
from benchmarks import aniso_mainline as baseline
from benchmarks.aniso_transfer_probes import manufactured_probe
from utils.resource_guard import prepare_warp_cache, inspect_storage

ROOT=baseline.ROOT
DEFAULT=ROOT/'docs/results/lite-aniso-mainline/v3'
TESTS=baseline.TESTS+['tests.test_aniso_force_accuracy','tests.test_aniso_boundary_impulse']


def hashes():
    result=baseline.hashes()
    for name in ('benchmarks/aniso_boundary_impulse.py','benchmarks/aniso_transfer_probes.py',
                 'tests/test_aniso_force_accuracy.py','tests/test_aniso_boundary_impulse.py'):
        result[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    return result


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve existing frozen protocol')
    configs={}
    for case,kf,angle in [('ISO',0.,0.),('F45',200.,45.)]:
        for level,dt in [('coarse',.001),('fine',.0005)]:
            for enabled in (False,True):
                configs[f'{case}-{level}-'+('on' if enabled else 'off')]=asdict(replace(baseline.BASE,
                    kf=kf,fiber_angle=angle,dt=dt,reaction_force_atol=1e-7,boundary_impulse_transfer=enabled))
    p=dict(name='lite-aniso-boundary-impulse-v3',frozen_at=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashes(),configs=configs,device='cpu',precision='float64',tests=TESTS,
        duration=.1,loading_time=.5,endpoint_note='short prefix of unchanged v2 fast loading; not full displacement acceptance',
        acceptance={'force_error_max_N':1e-7,'particle_grid_gap_max_kg_m_s':1e-14,
                    'grid_grip_velocity_max_m_s':1e-12,'particle_grip_error_max_m':.001,
                    'legacy_disabled_state_max_error':1e-12,
                    'legacy_F_comparison':'positive-time snapshots only; v2 initial F has confirmed live-array alias','all_required_tests_no_skips':True},
        analytic={'cells':[8,16,32],'waves':[1,2],'offsets':[.13,.73],
                  'fields':['constant','affine','sine'],'fd_oracle_max':1e-9,
                  'refinement_ratio_max':.4,'scope':'complete support; prescribed velocity, no material solve'},
        run_count=8,total_steps=1200,
        cuda={'status':'not_run','reason':'bounded CPU float64 validation'},
        scope='single-material center quadrature; boundary fix only, no gradient algorithm change',
        resource={'max_cpu_processes':2,'threads_each':1,'storage':asdict(inspect_storage())},
        previous_manifests={v:hashlib.sha256((ROOT/f'docs/results/lite-aniso-mainline/{v}/artifact-sha256.json').read_bytes()).hexdigest() for v in ('v1','v2')})
    baseline.write_json(out/'protocol.json',p)


def verify(p):
    current=hashes()
    changed=[name for name,h in p['source_sha256'].items() if current.get(name)!=h]
    if changed:raise RuntimeError('frozen source changed: '+str(changed))


def probes(out):
    target=out/'analytic-gradients.json'
    if target.exists():raise RuntimeError('preserve existing probes')
    rows=[manufactured_probe(n,w,o) for n in (8,16,32) for w in (1,2) for o in (.13,.73)]
    rows += [manufactured_probe(8,offset=o,field=f) for f in ('constant','affine') for o in (.13,.73)]
    baseline.write_json(target,rows)


def run(out,p,jobs):
    if not json.loads((out/'tests.json').read_text())['required_checks_passed']:raise RuntimeError('tests must pass first')
    def launch(name):
        log=out/(name+'.log')
        if log.exists() or (out/'cases'/name).exists():raise RuntimeError('preserve existing case '+name)
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_boundary_impulse','worker',
                '--output',str(out),'--case',name],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        print(name,'exit',result.returncode,flush=True)
        return dict(case=name,exit_code=result.returncode)
    with ThreadPoolExecutor(max_workers=jobs) as pool:records=list(pool.map(launch,p['configs']))
    baseline.write_json(out/'batch.json',records)
    return all(r['exit_code']==0 for r in records)


def analyze(out,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data={};checks={};legacy={};sensitivity={}
    for name,cfg in p['configs'].items():
        dest=out/'cases'/name
        status=json.loads((dest/'status.json').read_text())
        if not status['run_completed']:raise RuntimeError('incomplete '+name)
        rows=[json.loads(line) for line in (dest/'steps.jsonl').read_text().splitlines()];data[name]=rows
        c=dict(steps=len(rows),max_grid_force_error_N=max(abs(r['momentum_balance_error']) for r in rows),
            max_particle_force_error_N=max(r['particle_momentum_balance_error_norm'] for r in rows),
            rms_particle_force_error_N=float(np.sqrt(np.mean([r['particle_momentum_balance_error_norm']**2 for r in rows]))),
            max_particle_grid_gap_kg_m_s=max(r['particle_grid_momentum_gap_norm'] for r in rows),
            min_particle_J=min(r['min_particle_det_F'] for r in rows),min_center_J=min(r['min_det_F'] for r in rows),
            max_grid_grip_velocity_error=max(r['grid_grip_velocity_error'] for r in rows),
            max_particle_grip_error=max(abs(r['measured_grip_displacement']-r['displacement']) for r in rows),
            nonlinear_targets_pass=all(r['last_residual_norm']<=r['newton_residual_target'] for r in rows))
        c['basic_pass']=bool(c['max_grid_force_error_N']<=1e-7 and c['min_particle_J']>0 and c['min_center_J']>0
            and c['max_grid_grip_velocity_error']<=1e-12 and c['max_particle_grip_error']<=.001 and c['nonlinear_targets_pass'])
        c['particle_pass']=bool(c['max_particle_force_error_N']<=1e-7 and c['max_particle_grid_gap_kg_m_s']<=1e-14)
        checks[name]=c
        if not cfg['boundary_impulse_transfer']:
            case,level,_=name.split('-');old=ROOT/f'docs/results/lite-aniso-mainline/v2/cases/fast-{case}-{level}'
            oldrows=[json.loads(line) for line in (old/'steps.jsonl').read_text().splitlines()][:len(rows)]
            force_error=max(abs(a['right_force']-b['right_force']) for a,b in zip(rows,oldrows))
            current=np.load(dest/'frames.npz');previous=np.load(old/'frames.npz')
            errors={};matched=0
            for i,t in enumerate(current['time']):
                j=np.flatnonzero(abs(previous['time']-t)<1e-12)
                if not len(j):continue
                matched+=1
                for key in ('x','F'):
                    # v2 initial F aliases its final state; preserve original evidence.
                    if key == 'F' and abs(t) < 1e-12:continue
                    errors[key]=max(errors.get(key,0.),float(np.max(abs(current[key][i]-previous[key][j[0]]))))
            legacy[name]=dict(max_reaction_error_N=force_error,matched_frames=matched,state_max_errors=errors,
                excluded_v2_initial_F='old runner retained a live CPU array; initial F equals final F',
                passed=bool(force_error<=1e-12 and matched>=2 and max(errors.values())<=1e-12))
    for case in ('ISO','F45'):
        for mode in ('off','on'):
            coarse=data[f'{case}-coarse-{mode}'];fine=data[f'{case}-fine-{mode}']
            t=np.array([r['time'] for r in coarse]);mask=t>=.01-1e-12;t=t[mask]
            rf=np.interp(t,[r['time'] for r in fine],[r['right_force'] for r in fine])
            rc=np.array([r['right_force'] for r in coarse])[mask]
            rms=lambda a:float(np.sqrt(np.trapezoid(a*a,t)/(t[-1]-t[0])))
            sensitivity[f'{case}-{mode}']=dict(relative=rms(rc-rf)/max(rms(rf),.001),absolute_rms_N=rms(rc-rf),
                note='exploratory prefix t=.01..0.1; not comparable to v2 full-trajectory acceptance')
    summary=dict(required_checks_passed=json.loads((out/'tests.json').read_text())['required_checks_passed'],
        checks=checks,disabled_vs_v2=legacy,short_time_sensitivity=sensitivity,
        all_acceptance_passed=all(c['basic_pass'] and (c['particle_pass'] or name.endswith('-off')) for name,c in checks.items())
            and all(c['passed'] for c in legacy.values()),
        full_trajectory_accuracy='not_tested',gradient_algorithm='unchanged',cuda='not_tested')
    summary['all_acceptance_passed'] &= summary['required_checks_passed']
    baseline.write_json(out/'summary.json',summary)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for i,case in enumerate(('ISO','F45')):
        for mode in ('off','on'):
            rows=data[f'{case}-fine-{mode}'];t=[r['time'] for r in rows]
            axes[i,0].plot(t,[r['right_force'] for r in rows],label=mode)
            axes[i,1].semilogy(t,[max(r['particle_momentum_balance_error_norm'],1e-16) for r in rows],label=mode)
        axes[i,0].set_ylabel(case+' reaction (N)');axes[i,1].set_ylabel(case+' particle force error (N)')
        axes[i,1].axhline(1e-7,color='k',ls=':',label='tolerance')
        for ax in axes[i]:ax.set_xlabel('time (s)');ax.legend();ax.grid(alpha=.25)
    fig.savefig(out/'boundary-impulse-comparison.png',dpi=160);plt.close(fig)
    rows=json.loads((out/'analytic-gradients.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for w in (1,2):
        r=[a for a in rows if a['field']=='sine' and a['waves']==w and a['offset']==.13]
        axes[0].loglog([1/a['cells'] for a in r],[a['particle_gradient_rms_error'] for a in r],'-o',label=f'{w} wave(s)')
        axes[1].plot([a['cells'] for a in r],[a['particle_gradient_amplitude_gain'] for a in r],'-o',label=f'{w} wave(s)')
    axes[0].set(xlabel='dx (m)',ylabel='particle gradient RMS error (1/s)')
    axes[1].set(xlabel='cells per axis',ylabel='gradient amplitude gain');axes[1].axhline(1,color='k',ls=':')
    for ax in axes:ax.legend();ax.grid(alpha=.25)
    fig.savefig(out/'analytic-gradient-convergence.png',dpi=160);plt.close(fig)
    return summary['all_acceptance_passed']


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('action',choices=('freeze','tests','probes','run','worker','analyze'))
    parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--jobs',type=int,choices=(1,2),default=2)
    parser.add_argument('--case');args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache')
    if args.action=='freeze':freeze(out);return
    p=json.loads((out/'protocol.json').read_text());verify(p)
    if args.action=='tests':ok=baseline.tests(out,p)
    elif args.action=='probes':probes(out);ok=True
    elif args.action=='run':ok=run(out,p,args.jobs)
    elif args.action=='worker':ok=baseline.run_case(out,args.case,p['configs'][args.case],p)
    else:ok=analyze(out,p)
    raise SystemExit(0 if ok else 2)


if __name__=='__main__':main()
