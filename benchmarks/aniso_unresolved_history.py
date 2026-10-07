"""v13: stage energy audit and four-dt stretch/hold/unload verification."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import warp as wp
from demos.aniso import Config,Scene
from benchmarks import aniso_mainline as base
from benchmarks import aniso_split_history as v12
from benchmarks.aniso_selective_history import PatchLedger,LEVELS
from engine.aniso_phase1.diagnostics import particle_kinetic
from engine.aniso_phase1.tensile import grid_values
import engine.aniso_phase1.tensile as tensile
from utils.resource_guard import prepare_warp_cache
ROOT=v12.ROOT;BASE=ROOT/'docs/results/lite-aniso-mainline';OUT=BASE/'v13'
TESTS=v12.TESTS+['tests.test_aniso_unresolved_velocity']
MODES={'baseline':'none','null':'null','weak':'weak'}


def load(path):return json.loads(path.read_text())
def write(path,value):base.write_json(path,value)


def displacement(t,*args,**kwargs):
    t=float(t)
    if t<=0 or t>=1.1:return 0.
    if t<.5:return .0025*(1-np.cos(np.pi*t/.5))
    if t<=.6:return .005
    return .0025*(1+np.cos(np.pi*(t-.6)/.5))


def phase(t):
    return 'ramp' if t<=.5+1e-10 else 'hold' if t<=.6+1e-10 else 'unload' if t<=1.1+1e-10 else 'final_hold'


class StageLedger(PatchLedger):
    def begin(self,s):
        self.old_x=s.ptc_x.numpy().copy()
        super().begin(s)
    def finish(self,s):
        super().finish(s);row=self.rows[-1]
        if s._dissipation_unfiltered is None:v,C=s.ptc_v.numpy(),s.ptc_C.numpy()
        else:v,C=s._dissipation_unfiltered
        m=s.ptc_m.numpy();fixed=sum(particle_kinetic(self.old_x,v,C,m,s.dx));advected=sum(particle_kinetic(s.ptc_x.numpy(),v,C,m,s.dx))
        row['g2p_fixed_geometry_delta']=fixed-self.k_final_grid
        row['kinetic_metric_change']=advected-fixed
        row['selective_dissipation_delta']=row['kinetic']-advected
        row['transfer_roundtrip_delta']=row['p2c_delta']+row['c2g_delta']+row['g2p_fixed_geometry_delta']
        terms=('transfer_roundtrip_delta','boundary_projection_delta','solve_delta','final_projection_damping_delta','kinetic_metric_change','selective_dissipation_delta','stabilization_rebuild_delta')
        row['stage_budget_error']=row['delta_mechanical']-sum(row[k] for k in terms)
        if self.audit_next:
            self.snapshot.update(particle_velocity_unfiltered=v.copy(),particle_C_unfiltered=C.copy())
            self.audit.update({k:row[k] for k in terms},stage_budget_error=row['stage_budget_error'])


def hashes():
    paths=set(v12.hashes())|{'benchmarks/aniso_unresolved_history.py','benchmarks/aniso_unresolved_check.py','benchmarks/aniso_unresolved_screen.py'}
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sorted(paths)}


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve protocol')
    old=load(BASE/'v12/protocol.json');configs={}
    for label,mode in MODES.items():
        for level in LEVELS:configs[label+'-'+level]={**old['configs']['F45-'+level],'velocity_dissipation':mode}
    write(out/'protocol.json',dict(source_sha256=hashes(),configs=configs,tests=TESTS,device='cpu',precision='float64',cuda='not_run',
        duration=1.2,ramp=[0.,.5],hold=[.5,.6],unload=[.6,1.1],final_hold=[1.1,1.2],peak_displacement=.005,
        snapshot_times=[.05,.25,.5,.55,.6,.85,1.1,1.2],frame_interval=.025,modes=MODES,total_cases=12,total_steps=54000,
        coefficient='visibility energy fraction 0.1; rate=sqrt(mu/rho)/h; no reaction fitting; strict-null projection is a stronger zero-mode control',
        gates=dict(required_tests=103,force_balance=1e-7,history_closure=1e-12,stage_budget=1e-12,relative_time=.02,min_observed_order=.5,
                   improvement='compare load stress sensitivity, hold energy growth/drift and unload residual; report every failed gate; no automatic promotion'),
        default_changed=False,scope='F45 complete cyclic protocol; inherited four-direction static studies plus rerun actual massless/invariance tests; small CPU SVD prototype, not a spatial-convergence study'))


def static(out,p):
    assert load(out/'tests.json')['required_checks_passed']
    old=load(BASE/'v12/protocol.json');names=['engine/aniso_phase1/selective_patch.py','engine/aniso_phase1/residual_history.py','engine/aniso_phase1/projected_history.py','engine/aniso_phase1/constitutive.py','engine/aniso_phase1/enhancements.py']
    checks={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==old['source_sha256'][f] for f in names};assert all(checks.values())
    assert load(BASE/'v11/static-summary.json')['loading_allowed']
    write(out/'static-preservation.json',dict(passed=True,unchanged_energy_sources=checks,inherited_tensile_cases=24,inherited_beam_cases=16,
        tests_rerun='actual Warp massless K for ISO/F0/F45/F90, energy/residual/tangent, rigid/affine/quadratic and local F history; same-input post-solve filter leaves grid solve, F, x, L and potential identical',
        no_static_stiffness_added=True,filter='separate kinetic contraction after solve; fallible SVD before particle commit'))


def worker(out,p,name):
    dest=out/'cases'/name;dest.mkdir(parents=True,exist_ok=False);cfg=p['configs'][name];write(dest/'config.json',cfg)
    status=dict(run_completed=False,requested_steps=round(p['duration']/cfg['dt']),completed_steps=0,error=None);frames=[];Fs=[];vel=[];Cs=[];times=[];scene=None;start=time.monotonic()
    tensile.loading_displacement=displacement
    try:
        scene=Scene(Config(**cfg),'cpu');s=scene.solver;s.energy_ledger=StageLedger()
        def frame():
            frames.append(s.ptc_x.numpy().copy());Fs.append(s.ptc_F.numpy().copy());vel.append(s.ptc_v.numpy().copy());Cs.append(s.ptc_C.numpy().copy());times.append(s.sim_time)
        frame();audits={round(t/cfg['dt']) for t in p['snapshot_times']};every=round(p['frame_interval']/cfg['dt'])
        with (dest/'steps.jsonl').open('x',buffering=1) as log:
            for step in range(1,status['requested_steps']+1):
                prepare_warp_cache('/tmp/mpm-lite-warp-cache');s.energy_ledger.audit_next=step in audits
                if not scene.step():raise RuntimeError('step failed without commit '+str(step)+' '+str(s.last_step_stats))
                row=scene.metrics();nodes=s.energy_ledger.nodes;v=grid_values(s,s.grid_v_new,nodes);grip=(nodes[:,0]*s.dx<=.25)|(nodes[:,0]*s.dx>=.75);expected=np.zeros_like(v);expected[nodes[:,0]*s.dx>=.75,0]=row['loading_velocity']
                row.update(phase=phase(s.sim_time),min_particle_det_F=float(np.linalg.det(s.ptc_F.numpy()).min()),grid_grip_velocity_error=float(np.max(abs(v[grip]-expected[grip]))))
                log.write(json.dumps(row,allow_nan=False)+'\n');status['completed_steps']=step
                if step in audits:
                    write(dest/f'audit-{step:05d}.json',s.energy_ledger.audit);np.savez_compressed(dest/f'audit-{step:05d}.npz',**s.energy_ledger.snapshot)
                if step%every==0:frame()
        status.update(run_completed=True,sim_time=s.sim_time,last_step_stats=s.last_step_stats)
    except Exception:
        status['error']=traceback.format_exc();print(status['error'],flush=True)
    finally:
        status['wall_seconds']=time.monotonic()-start;write(dest/'status.json',status);np.savez_compressed(dest/'frames.npz',time=times,x=frames,F=Fs,v=vel,C=Cs)
    return status['run_completed']


def run(out,p,jobs):
    assert load(out/'tests.json')['required_checks_passed'] and load(out/'static-preservation.json')['passed']
    def launch(name):
        with (out/(name+'.log')).open('x') as log:
            r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_unresolved_history','worker','--case',name,'--output',str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        print(name,r.returncode,flush=True);return dict(case=name,exit_code=r.returncode)
    names=[mode+'-'+level for level in reversed(LEVELS) for mode in ('weak','baseline','null')]
    with ThreadPoolExecutor(max_workers=jobs) as pool:r=list(pool.map(launch,names))
    write(out/'batch.json',r);return all(x['exit_code']==0 for x in r)


def main():
    wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache';a=argparse.ArgumentParser(__doc__);a.add_argument('action',choices=('freeze','tests','static','run','worker'));a.add_argument('--output',type=Path,default=OUT);a.add_argument('--case');a.add_argument('--jobs',type=int,choices=(1,2,4,8,12),default=8);args=a.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.action=='freeze':freeze(out);return
    p=load(out/'protocol.json');assert hashes()==p['source_sha256'],'frozen source changed'
    if args.action=='tests':ok=base.tests(out,p)
    elif args.action=='static':static(out,p);ok=True
    elif args.action=='worker':ok=worker(out,p,args.case)
    else:ok=run(out,p,args.jobs)
    raise SystemExit(0 if ok else 2)

if __name__=='__main__':main()
