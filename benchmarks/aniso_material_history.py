"""v14 material-carried stabilization and extended four-dt recovery test."""
import argparse,hashlib,os,sys,subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import warp as wp
from benchmarks import aniso_unresolved_history as v13
ROOT=v13.ROOT;BASE=v13.BASE;OUT=BASE/'v14';LEVELS=v13.LEVELS
TESTS=v13.TESTS+['tests.test_aniso_material_patch']
load=v13.load;write=v13.write

class MaterialLedger(v13.StageLedger):
    def finish(self,s):
        super().finish(s)
        if s.stabilization=='material_patch' and self.audit_next:
            e=s.enhancements
            self.snapshot.update(patch_origin=e.last_origin.copy(),marker_after=e.origin.numpy().copy(),carrier_ids=e.carrier_ids.numpy().copy(),carrier_weights=e.carrier_weights.numpy().copy())
            expected=e.last_origin+s.dt*(e.N@self.snapshot['native_velocity'])
            self.audit['carrier_commit_max']=float(np.max(abs(expected-e.origin.numpy())))
            self.audit['carrier_affine_error']=e.reconstruction_stats['carrier_affine_error']


def hashes():
    files=set(v13.hashes())|{'benchmarks/aniso_material_history.py','benchmarks/aniso_material_check.py'}
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sorted(files)}


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve protocol')
    old=load(BASE/'v13/protocol.json');configs={}
    for label,kind in [('baseline','selective_patch'),('material','material_patch')]:
        for level in LEVELS:configs[label+'-'+level]={**old['configs']['null-'+level],'stabilization':kind}
    write(out/'protocol.json',dict(source_sha256=hashes(),configs=configs,tests=TESTS,device='cpu',precision='float64',cuda='not_run',duration=1.6,
        ramp=[0.,.5],hold=[.5,.6],unload=[.6,1.1],final_hold=[1.1,1.6],peak_displacement=.005,snapshot_times=[.05,.25,.5,.55,.6,.85,1.1,1.2,1.4,1.6],frame_interval=.025,total_cases=8,total_steps=48000,
        intervention='carry fixed material patch projector and reference weights; update virtual positions by Q1 grid velocity; no energy offset or coefficient fitting; strict-null velocity control identical in both arms',
        gates=dict(required_tests=109,force_balance=1e-7,history_closure=1e-12,stage_budget=1e-12,rebuild_energy=1e-14,relative_time=.02,min_observed_order=.5),
        scope='F45 full four-dt cycle with five-times longer final hold; separate static spatial/particle-density checks and splitting counterfactuals; defaults unchanged'))


def main():
    wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache';a=argparse.ArgumentParser(__doc__);a.add_argument('action',choices=('freeze','tests','run','worker'));a.add_argument('--output',type=v13.Path,default=OUT);a.add_argument('--case');a.add_argument('--jobs',type=int,default=8);args=a.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.action=='freeze':freeze(out);return
    p=load(out/'protocol.json');assert hashes()==p['source_sha256'],'frozen source changed'
    if args.action=='tests':ok=v13.base.tests(out,p)
    elif args.action=='worker':v13.StageLedger=MaterialLedger;ok=v13.worker(out,p,args.case)
    else:
        t=load(out/'tests.json');assert t['required_checks_passed'] and t['tests_run']==109
        def launch(name):
            with (out/(name+'.log')).open('x') as f:r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_material_history','worker','--case',name,'--output',str(out)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
            print(name,r.returncode,flush=True);return dict(case=name,exit_code=r.returncode)
        names=[m+'-'+l for l in reversed(LEVELS) for m in ('material','baseline')]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:r=list(pool.map(launch,names))
        write(out/'batch.json',r);ok=all(x['exit_code']==0 for x in r)
    raise SystemExit(0 if ok else 2)

if __name__=='__main__':main()
