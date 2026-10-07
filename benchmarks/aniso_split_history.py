"""v12: retain affine velocity history, keeping the v11 static energy unchanged."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warp as wp
from benchmarks import aniso_mainline as base
from benchmarks import aniso_selective_history as v11
ROOT=v11.ROOT;OUT=ROOT/'docs/results/lite-aniso-mainline/v12'
TESTS=v11.TESTS+['tests.test_aniso_split_affine']


def hashes():
    paths=set(v11.hashes())|{'benchmarks/aniso_split_history.py','benchmarks/aniso_split_check.py','benchmarks/aniso_time_compare.py','benchmarks/aniso_transfer_attribution.py'}
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sorted(paths)}


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve protocol')
    old=json.loads((v11.OUT/'protocol.json').read_text());p={**old,'source_sha256':hashes(),'tests':TESTS}
    p['configs']={n:{**c,'affine_flip_ratio':1.} for n,c in old['configs'].items()}
    p.update(purpose='retain full incremental affine velocity C; velocity keeps calibrated PIC relaxation; material F still uses unblended L',
        static='v11 energy, reference reconstruction and coefficients unchanged; byte-hash proof, inherited full static study plus rerun real-kernel massless tests',
        candidate_gate='all 95 regression tests without skips; v11 static energy sources identical and sealed static rank/solve gate valid',
        time_gate={'R_F_P_U_relative':.02,'full_space_time_P_relative':.02,'observed_order_min':.5,'F45_stress_reduction_target':.5},
        baseline_inputs={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [v11.OUT/'artifact-sha256.json',v11.OUT/'protocol.json',v11.OUT/'static-summary.json',v11.OUT/'time-field-audit.json']},
        max_cpu_processes=8,default_changed=False,accuracy_certified=False)
    base.write_json(out/'protocol.json',p)


def static_gate(out,p):
    assert json.loads((out/'tests.json').read_text())['required_checks_passed']
    old=json.loads((v11.OUT/'protocol.json').read_text())
    files=['engine/aniso_phase1/selective_patch.py','engine/aniso_phase1/residual_history.py','engine/aniso_phase1/projected_history.py','engine/aniso_phase1/constitutive.py','engine/aniso_phase1/enhancements.py','benchmarks/aniso_residual_gate.py','benchmarks/aniso_selective_history.py']
    checks={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==old['source_sha256'][f] for f in files};assert all(checks.values())
    for f,h in p['baseline_inputs'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h
    s=json.loads((v11.OUT/'static-summary.json').read_text());assert s['loading_allowed']
    r=next(r for r in s['records'] if r['case']=='F45' and r['grid']==33 and r['mode']=='selective_patch')
    result=dict(passed=True,unchanged_static_source=checks,inherited_tensile_cases=len(s['records']),inherited_beam_cases=len(s['beams']),F45_grid33=r,
                newly_rerun='95 tests including real Warp (K-M)/dt^2 versus assembled static K for ISO/F0/F45/F90; rigid/affine/quadratic modes; energy/residual/tangent; same-input F/positions/energy invariance under split return',
                reason='affine return occurs after grid nonlinear solve; its coefficient never enters potential, residual, tangent, reference reconstruction or static assemblers')
    base.write_json(out/'static-preservation.json',result)


def worker(out,p,name):return v11.worker(out,p,name)


def run(out,p,jobs):
    assert json.loads((out/'tests.json').read_text())['required_checks_passed']
    assert json.loads((out/'static-preservation.json').read_text())['passed']
    def launch(name):
        with (out/(name+'.log')).open('x') as log:
            r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_split_history','worker','--case',name,'--output',str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        print(name,r.returncode,flush=True);return dict(case=name,exit_code=r.returncode)
    names=[label+'-'+level for level in reversed(v11.LEVELS) for label in ('F45','ISO','F0','F90')]
    with ThreadPoolExecutor(max_workers=jobs) as pool:r=list(pool.map(launch,names))
    base.write_json(out/'batch.json',r);return all(x['exit_code']==0 for x in r)


def analyze(out,p):
    from benchmarks.aniso_time_compare import compare
    r={label:compare(out,p['configs'],label,out/(label+'-comparison.json')) for label in ('ISO','F0','F45','F90')}
    baseline=compare(v11.OUT,json.loads((v11.OUT/'protocol.json').read_text())['configs'])
    a=baseline['pairs'][-1];b=r['F45']['pairs'][-1]
    improvement={k:1-b[k]/a[k] for k in ('P_relative','P_terminal_relative','reaction_relative','F_relative','energy_relative')}
    result=dict(completed=all(x['completed'] for x in r.values()),physical_checks_passed=all(x['physical_passed'] for x in r.values()),
        trajectories=len(p['configs']),steps=sum(c['steps'] for x in r.values() for c in x['checks'].values()),
        history_closure_max=max(x['history_closure_max'] for x in r.values()),particle_commit_max=max(x['particle_commit_max'] for x in r.values()),
        time_threshold_passed=all(x['time_2pct_passed'] for x in r.values()),reliable_time_trend=all(x['orders_passed'] for x in r.values()),
        F45_improvement_fraction=improvement,F45_stress_reduction_target_passed=min(improvement[k] for k in ('P_relative','P_terminal_relative'))>=.5,
        cases={label:{k:x[k] for k in ('physical_passed','time_2pct_passed','orders_passed','pairs')} for label,x in r.items()},
        default_changed=False,overall_accuracy_accepted=False)
    base.write_json(out/'summary.json',result);print(json.dumps(result,indent=2));return result['physical_checks_passed']


def main():
    wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
    a=argparse.ArgumentParser(__doc__);a.add_argument('action',choices=('freeze','tests','static','run','worker','analyze'));a.add_argument('--output',type=Path,default=OUT);a.add_argument('--case');a.add_argument('--jobs',type=int,choices=(1,2,4,8),default=8);args=a.parse_args();out=args.output;out.mkdir(exist_ok=True,parents=True)
    if args.action=='freeze':freeze(out);return
    p=json.loads((out/'protocol.json').read_text());assert hashes()==p['source_sha256'],'frozen source changed'
    if args.action=='tests':ok=base.tests(out,p)
    elif args.action=='static':static_gate(out,p);ok=True
    elif args.action=='worker':ok=worker(out,p,args.case)
    elif args.action=='run':ok=run(out,p,args.jobs)
    else:ok=analyze(out,p)
    raise SystemExit(0 if ok else 2)

if __name__=='__main__':main()
