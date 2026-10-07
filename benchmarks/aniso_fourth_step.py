"""Fourth time-step F45 run and same-state material history resampling diagnosis."""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import asdict
import argparse
import hashlib
import json
import numpy as np
import warp as wp
from benchmarks import aniso_affine_consistency as previous
from benchmarks import aniso_mainline as base
from benchmarks import aniso_f45_refinement as refine
from benchmarks.aniso_flip_time import scaled_flip
from benchmarks.aniso_resample_diagnostic import decompose,replay
from demos.aniso import Config
from utils.resource_guard import prepare_warp_cache,inspect_storage
ROOT=base.ROOT;DEFAULT=ROOT/'docs/results/lite-aniso-mainline/v8'
LEVELS=['coarse','fine','finest','fourth']


def hashes():
    h=previous.hashes()
    for f in ('benchmarks/aniso_fourth_step.py','benchmarks/aniso_resample_diagnostic.py','tests/test_aniso_resample_diagnostic.py'):
        h[f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    return h


def references():
    previous.verify(json.loads((previous.DEFAULT/'protocol.json').read_text()))
    count=previous.references()
    for f,h in json.loads((previous.DEFAULT/'artifact-sha256.json').read_text()).items():
        if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h:raise RuntimeError('reference changed '+f)
        count+=1
    return count


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve protocol')
    old=json.loads((previous.DEFAULT/'protocol.json').read_text())
    cfg=old['configs']['incremental-finest'].copy();cfg.update(dt=.000125,flip_ratio=scaled_flip(.000125))
    base.write_json(out/'protocol.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),source_sha256=hashes(),
        configs={'incremental-fourth':cfg},duration=.5,endpoint_displacement_m=.005,snapshot_times=[.05,.1,.25,.5],
        device='cpu',precision='float64',tests=['tests.test_aniso_resample_diagnostic'],
        unchanged_production_regression={'source_verified':True,'previous_tests':61,'results':str((previous.DEFAULT/'tests.json').relative_to(ROOT))},
        reference_files_verified=references(),run_count=1,total_steps=4000,
        acceptance=dict(relative_force_max=.05,force_floor_N=.001,interval=[.05,.5],
            decreasing_absolute_difference=True,force_error_N=1e-7,oracle_max=1e-12,
            note='report adjacent ratios and observed orders; passing 5% alone does not establish convergence'),
        diagnosis='four common physical end times; covariance + gradient mismatch + moving weights; ordered nonlinear stress/energy effects; actual repeated rebuild on frozen particles',
        cuda={'status':'not_run','reason':'CPU float64 controlled verification'},resource=asdict(inspect_storage()),
        production_changed=False))


def verify(p):
    if hashes()!=p['source_sha256']:raise RuntimeError('frozen source changed')


def diagnostic(out,p,reference_only=False):
    folder=out/('reference-diagnosis' if reference_only else 'diagnosis')
    if folder.exists():raise RuntimeError('preserve diagnostic results')
    folder.mkdir();records=[]
    levels=LEVELS[:3] if reference_only else LEVELS
    for level in levels:
        root=previous.DEFAULT if level!='fourth' else out
        name='incremental-'+level;cfg=json.loads((root/'cases'/name/'config.json').read_text())
        params=Config(**cfg).params
        for path in sorted((root/'cases'/name).glob('audit-*.npz')):
            with np.load(path) as z:r,arrays=decompose(z,cfg['dt'],params);q=replay(z,cfg,arrays)
            r.update(case=name,dt=cfg['dt'],time=int(path.stem.split('-')[-1])*cfg['dt'],replay=q)
            assert max(r['checks'].values())<=1e-12 and max(q.values())<=1e-12
            np.savez_compressed(folder/(name+'-'+path.name),**arrays)
            records.append(r);print('diagnosed',name,path.name,flush=True)
    assert len(records)==len(levels)*4
    base.write_json(folder/'summary.json',dict(passed=True,records=records,production_changed=False))
    return True


def analyze(out,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    new,checks,audits=refine.read_cases(out,p['configs'])
    old,oldchecks,_=refine.read_cases(previous.DEFAULT,['incremental-'+l for l in LEVELS[:3]])
    data={**old,**new};complete=len(data)==4;trends={}
    t=np.array([r['time'] for r in data['incremental-coarse']]);t=t[(t>=.05-1e-12)&(t<=.5+1e-12)]
    rms=lambda a:float(np.sqrt(np.trapezoid(a*a,t)/(t[-1]-t[0])))
    for key in ('right_force','right_elastic_force','right_inertial_force'):
        curves=[np.interp(t,[r['time'] for r in data['incremental-'+l]],[r[key] for r in data['incremental-'+l]]) for l in LEVELS]
        pairs=[]
        for i in range(3):
            error=rms(curves[i]-curves[i+1]);denom=max(rms(curves[i+1]),.001)
            r=dict(pair=LEVELS[i]+'-'+LEVELS[i+1],absolute_rms_N=error,relative=error/denom)
            if i:
                rho=error/pairs[-1]['absolute_rms_N'];r.update(rho=rho,observed_order=float(-np.log2(rho)))
            pairs.append(r)
        trends[key]=pairs
    diagnostic_record=json.loads((out/'diagnosis/summary.json').read_text())
    physical=complete and all(c['passed'] for c in {**oldchecks,**checks}.values())
    total=trends['right_force'];numeric=total[-1]['relative']<=.05 and total[-1]['rho']<1
    summary=dict(experiment_completed=complete,refinement=trends,checks=checks,
        physical_checks_passed=physical,diagnostics_passed=diagnostic_record['passed'],
        required_checks_passed=json.loads((out/'tests.json').read_text())['required_checks_passed'],
        time_threshold_passed=numeric,production_changed=False,spatial_accuracy='not_verified',cuda='not_run')
    summary['all_frozen_numeric_checks_passed']=bool(physical and numeric and summary['diagnostics_passed'] and summary['required_checks_passed'])
    base.write_json(out/'summary.json',summary)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for l in LEVELS:
        rows=data['incremental-'+l]
        for ax,key in zip(axes,trends):
            ax.plot([r['time'] for r in rows],[r[key] for r in rows],label=l)
            ax.set(xlabel='time (s)',ylabel='reaction (N)',title=key);ax.grid(alpha=.25)
    for ax in axes:ax.legend()
    fig.savefig(out/'four-step-reactions.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))
    return summary['all_frozen_numeric_checks_passed']


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('freeze','tests','run','diagnostic','reference-diagnostic','analyze'))
    parser.add_argument('--output',type=Path,default=DEFAULT);args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache')
    if args.action=='freeze':freeze(out);return
    p=json.loads((out/'protocol.json').read_text());verify(p)
    if args.action=='tests':ok=base.tests(out,p)
    elif args.action=='run':
        if not json.loads((out/'tests.json').read_text())['required_checks_passed']:raise RuntimeError('tests must pass')
        references();ok=previous.worker(out,p,'incremental-fourth')
    elif args.action in ('diagnostic','reference-diagnostic'):ok=diagnostic(out,p,args.action=='reference-diagnostic')
    else:ok=analyze(out,p)
    raise SystemExit(0 if ok else 2)


if __name__=='__main__':main()
