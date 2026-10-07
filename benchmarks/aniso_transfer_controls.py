"""Controlled F45 velocity/affine damping ablations on the unchanged v11 solver.

Only this experiment replaces finish_transfer in each isolated worker process.
No material, history, stabilization, Newton or boundary formula is replaced.
"""
import argparse,json,os,subprocess,sys,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import warp as wp
from benchmarks import aniso_selective_history as v11
from benchmarks import aniso_residual_history as v10
from benchmarks import aniso_mainline as base
from demos.aniso import Scene
import engine.aniso_phase1.affine_transfer as transfer
ROOT=v11.ROOT;OUT=ROOT/'docs/results/lite-aniso-mainline/v12-transfer-controls'
ORIGINAL_FINISH=transfer.finish_transfer


def finish_separate(s):
    beta=s.flip_ratio
    try:
        s.flip_ratio=s.control_affine_beta
        ORIGINAL_FINISH(s)
    finally:s.flip_ratio=beta


def hashes():
    return {**v11.hashes(),'benchmarks/aniso_transfer_controls.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve protocol')
    old=json.loads((v11.OUT/'protocol.json').read_text());configs={};affine={}
    for mode in ('velocity_only','affine_only'):
        for level,dt in v11.LEVELS.items():
            c=old['configs']['F45-'+level].copy();beta=c['flip_ratio']
            name=mode+'-'+level;c['flip_ratio']=1. if mode=='velocity_only' else beta
            configs[name]=c;affine[name]=beta if mode=='velocity_only' else 1.
    p={**old,'source_sha256':hashes(),'configs':configs,'affine_beta':affine,
       'purpose':'2x2 factorial: remove velocity PIC relaxation or affine PIC relaxation, separately; baseline and both-removed are recorded separately',
       'acceptance':{'P_space_time_relative':.02,'P_terminal_relative':.02,'R_F_U_relative':.02,'all_orders_min':.5,'same_state_F_L_x_force_energy_invariant':True},
       'dynamic_cases':8,'dynamic_steps':15000,'max_cpu_processes':8}
    base.write_json(out/'protocol.json',p)


def worker(out,p,name):
    class ControlScene(Scene):
        def __init__(self,cfg,device):
            super().__init__(cfg,device);self.solver.control_affine_beta=p['affine_beta'][name]
    v10.Scene=ControlScene;transfer.finish_transfer=finish_separate
    return v11.worker(out,p,name)


def precheck(out,p):
    import tests.test_aniso_affine_transfer as t
    saved=t.finish_transfer;errs=[]
    for beta in (.9,.9**.5,.9**.25,.9**.125):
        for b in (0,1,2):
            ref=t.transfer_probe(beta=beta,boundary=b)
            def finish(s):s.control_affine_beta=1.;finish_separate(s)
            t.finish_transfer=finish
            r=t.transfer_probe(beta=beta,boundary=b)
            t.finish_transfer=saved
            expected=ref['L']+(ref['expected_C']-ref['L'])/beta
            err=max(float(np.max(abs(r['C']-expected))),*(float(np.max(abs(r[k]-ref[k]))) for k in ('v','L','F')))
            assert err<1e-12 and r['momentum_error']<1e-12
            errs.append(err)
    # v11 invariance/static tests cover the unchanged material/patch energies;
    # these checks additionally run the real split-return transfer kernels.
    base.write_json(out/'prechecks.json',dict(passed=True,actual_split_transfer_cases=len(errs),max_error=max(errs),static_source_unchanged=True))


def run(out,p,jobs):
    assert json.loads((out/'prechecks.json').read_text())['passed']
    assert json.loads((v11.OUT/'static-summary.json').read_text())['loading_allowed']
    def launch(name):
        with (out/(name+'.log')).open('x') as log:
            r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_transfer_controls','worker','--case',name,'--output',str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        print(name,r.returncode,flush=True);return dict(case=name,exit_code=r.returncode)
    names=[mode+'-'+l for l in reversed(v11.LEVELS) for mode in ('velocity_only','affine_only')]
    with ThreadPoolExecutor(max_workers=jobs) as pool:r=list(pool.map(launch,names))
    base.write_json(out/'batch.json',r);return all(x['exit_code']==0 for x in r)


def main():
    wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
    a=argparse.ArgumentParser(__doc__);a.add_argument('action',choices=('freeze','precheck','worker','run','analyze'));a.add_argument('--output',type=Path,default=OUT);a.add_argument('--case');a.add_argument('--jobs',type=int,default=8);args=a.parse_args();out=args.output;out.mkdir(exist_ok=True,parents=True)
    if args.action=='freeze':freeze(out);return
    p=json.loads((out/'protocol.json').read_text());assert hashes()==p['source_sha256']
    if args.action=='precheck':precheck(out,p);return
    if args.action=='analyze':
        from benchmarks.aniso_time_compare import compare
        for mode in ('velocity_only','affine_only'):
            r=compare(out,p['configs'],mode,out/(mode+'-comparison.json'));print(mode,json.dumps({k:v for k,v in r.items() if k!='checks'},indent=2))
        return
    ok=worker(out,p,args.case) if args.action=='worker' else run(out,p,args.jobs)
    raise SystemExit(0 if ok else 2)

if __name__=='__main__':main()
