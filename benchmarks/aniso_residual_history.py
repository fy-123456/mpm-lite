"""v10: local-history energy, massless candidate gate, and gated short loading."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import warp as wp
from demos.aniso import Config, Scene
from benchmarks import aniso_mainline as base
from benchmarks.aniso_projected_history import TESTS as OLD_TESTS, HistoryLedger, read_case
from benchmarks.aniso_boundary_reference import CASES, hessian
from benchmarks.aniso_residual_gate import geometry, matrix, rank_gate, static_solve, MODES
from benchmarks.aniso_static_space import center_interpolation
from engine.aniso_phase1.boundary_reference import LO, HI, gradient as q1_gradient
from engine.aniso_phase1.diagnostics import ParticleEnergyLedger
from engine.aniso_phase1.tensile import grid_values
from utils.resource_guard import prepare_warp_cache

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'docs/results/lite-aniso-mainline/v10'
TESTS=OLD_TESTS+['tests.test_aniso_residual_history', 'tests.test_aniso_corotated',
                 'benchmarks.aniso_static_space.StaticSpaceTests']


def hashes():
    paths=set(base.hashes())
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT/'tests').glob('test_aniso_*.py'))
    paths.update(['benchmarks/aniso_residual_history.py','benchmarks/aniso_residual_gate.py',
                  'benchmarks/aniso_q2_reference.py','benchmarks/aniso_boundary_reference.py'])
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sorted(paths)}


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve protocol')
    configs={}
    old=json.loads((ROOT/'docs/results/lite-aniso-mainline/v9-dynamic-space/protocol.json').read_text())
    for label in CASES:
        for level in ('coarse','fine'):
            cfg=old['configs'][f'{label}-g17-{level}'].copy()
            cfg.update(history_consistency='residual_center',stabilization='corotated',stabilization_strength=1.)
            configs[f'{label}-g17-{level}']=cfg
    base.write_json(out/'protocol.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),source_sha256=hashes(),
        device='cpu',precision='float64',tests=TESTS,cuda={'status':'not_run','reason':'CPU controlled audit'},
        static_grids=[9,17,33],samples=2,directions=CASES,modes=MODES,beam_grids=[17,33],
        candidate_gate='all required tests; positive massless spectra for all directions and static/beam grids before any loading batch',
        history='exact individual particle energy and W-averaged center commit',
        stabilization='existing objective corotated energy; exact rotation derivatives; coefficient fixed at 1',
        not_calibrated_to_reaction=True,default_changed=False,configs=configs,duration=.05,
        snapshot_times=[.005,.01,.025,.05],dynamic_steps=1200,dynamic_cases=8,
        acceptance=dict(residual=1e-8,rank_relative=1e-9,affine=1e-12,history=1e-12,
            force_balance_N=1e-7,time_relative=.02,space_relative=.05),
        reference='Q1 grid129 for ISO/F0/F90; Q2 64 cells/unit for F45; reference uncertainty reported separately',
        long_loading='not included; only complete 0.05 s diagnostic load after candidate gate'))


def evaluation(label):
    if label=='F45':
        from benchmarks.aniso_q2_reference import gradient
        folder=ROOT/'docs/results/lite-aniso-mainline/v10-reference-q2/cases'
        with np.load(folder/'F45-q2-n64.npz') as z:u=z['u'].copy()
        record=json.loads((folder/'F45-q2-n64.json').read_text());divisions=64;order=3
        evaluate=lambda X:gradient(X,64,u)
    else:
        folder=ROOT/'docs/results/lite-aniso-mainline/v10-reference/cases'
        with np.load(folder/f'{label}-hard-g129.npz') as z:u=z['u'].copy()
        record=json.loads((folder/f'{label}-hard-g129.json').read_text());divisions=128;order=2
        evaluate=lambda X:q1_gradient(X,129,u)
    counts=np.rint((HI-LO)*divisions).astype(int)
    cells=np.array(list(itertools.product(*[range(n) for n in counts])))
    z,w=np.polynomial.legendre.leggauss(order);z,w=(z+1)/2,w/2
    local=np.array(list(itertools.product(z,repeat=3)))
    X=LO+(cells[:,None,:]+local).reshape(-1,3)/divisions
    V=np.tile(np.prod(np.array(list(itertools.product(w,repeat=3))),axis=1)/divisions**3,len(cells))
    L=evaluate(X);P=(L.reshape(-1,9)@hessian(label).T).reshape(-1,3,3)
    return X,V,L,P,record


def static(out,protocol):
    if not json.loads((out/'tests.json').read_text())['required_checks_passed']:raise RuntimeError('tests required')
    dest=out/'static';dest.mkdir(exist_ok=False);records=[];beams=[]
    files=[]
    for folder in ('v10-reference','v10-reference-q2'):
        for f in (ROOT/f'docs/results/lite-aniso-mainline/{folder}/cases').glob('*'):files.append(f)
    base.write_json(out/'static-inputs.json',{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    for label in CASES:
        X,V,Lref,Pref,reference=evaluation(label);H=hessian(label)
        masks=dict(whole=np.ones(len(X),bool),near_grip=np.minimum(abs(X[:,0]-.25),abs(X[:,0]-.75))<.0625,
                   interior=(X[:,0]>.3125)&(X[:,0]<.6875),deep_interior=(X[:,0]>.375)&(X[:,0]<.625))
        norm=lambda q,mask:float(np.sqrt(np.sum(V[mask,None,None]*q[mask]**2)))
        for grid in protocol['static_grids']:
            g=geometry(grid);interpolation=center_interpolation(X,g['centers'],1/(grid-1))
            for mode in MODES:
                name=f'{label}-g{grid}-{mode}';start=time.monotonic();K=matrix(g,H,mode)
                gate=rank_gate(g,K);u,r=static_solve(g,K,gate['passed'])
                Lc=np.stack([D@u for D in g['D']],axis=2)
                L=(interpolation@Lc.reshape(-1,9)).reshape(-1,3,3);P=(L.reshape(-1,9)@H.T).reshape(-1,3,3)
                regions={key:dict(F_relative=norm(L-Lref,mask)/max(norm(Lref,mask),1e-30),
                    P_relative=norm(P-Pref,mask)/max(norm(Pref,mask),1e-30)) for key,mask in masks.items()}
                refenergy=reference.get('elastic_J',reference.get('energy_J'))
                r.update(case=label,grid=grid,mode=mode,gate=gate,regions=regions,
                    reaction_reference_N=reference['reaction_N'],reaction_relative=abs(r['reaction_N']-reference['reaction_N'])/abs(reference['reaction_N']),
                    energy_relative=abs(r['energy_J']-refenergy)/abs(refenergy),seconds=time.monotonic()-start)
                records.append(r);base.write_json(dest/(name+'.json'),r);np.savez_compressed(dest/(name+'.npz'),nodes=g['nodes'],u=u)
                print(name,'rank',gate['passed'],'R',r['reaction_N'],'gap',r['reaction_relative'],'solved',r['solved'],flush=True)
            del interpolation;gc.collect()
    for grid in protocol['beam_grids']:
        g=geometry(grid,True)
        for label in CASES:
            for mode in MODES:
                K=matrix(g,hessian(label),mode);gate=rank_gate(g,K,True)
                r=dict(grid=grid,case=label,mode=mode,gate=gate)
                if gate['passed']:
                    _,solution=static_solve(g,K,True);r.update(solution)
                else:r['static_solve']='blocked_by_zero_stiffness_gate'
                beams.append(r)
                print('beam',grid,label,mode,'rank',gate['passed'],'null',gate['zero_or_soft_modes_observed'],flush=True)
    candidates={mode:all(r['gate']['passed'] for r in records+beams if r['mode']==mode) for mode in MODES}
    result=dict(completed=True,records=records,beams=beams,candidate_static_gate=candidates,
        all_compatible_tensile_solves_passed=all(r['solved'] for r in records),
        no_mass_or_diagonal_stiffness_added=True)
    base.write_json(out/'static-summary.json',result);return True


class ResidualLedger(HistoryLedger,ParticleEnergyLedger):
    def finish(self,s):
        super().finish(s)
        if self.audit_next:
            e=s.enhancements
            self.snapshot.update(stabilizer_ids=e.reference_ids.numpy(),stabilizer_F0=e.reference_F0.numpy(),
                stabilizer_B=e.reference_B.numpy(),native_nodes=s.mapped_nodes,
                native_velocity=grid_values(s,s.grid_v_new,s.mapped_nodes),native_center_V=s.mapped_V.numpy(),
                native_center_A=s.mapped_A.numpy())


def worker(out,p,name):
    dest=out/'cases'/name;dest.mkdir(parents=True,exist_ok=False);cfg=p['configs'][name]
    base.write_json(dest/'config.json',cfg);count=round(p['duration']/cfg['dt'])
    status=dict(run_completed=False,requested_steps=count,completed_steps=0,error=None);start=time.monotonic()
    frames=[];Fs=[];times=[];scene=None
    try:
        scene=Scene(Config(**cfg),'cpu');s=scene.solver;s.energy_ledger=ResidualLedger()
        frames.append(s.ptc_x.numpy().copy());Fs.append(s.ptc_F.numpy().copy());times.append(0.)
        audits={round(t/cfg['dt']) for t in p['snapshot_times']}
        with (dest/'steps.jsonl').open('x',buffering=1) as log:
            for step in range(1,count+1):
                prepare_warp_cache('/tmp/mpm-lite-warp-cache');s.energy_ledger.audit_next=step in audits
                if not scene.step():raise RuntimeError('step failed without commit '+str(step))
                row=scene.metrics();nodes=s.energy_ledger.nodes;v=grid_values(s,s.grid_v_new,nodes)
                grip=(nodes[:,0]*s.dx<=.25)|(nodes[:,0]*s.dx>=.75);expected=np.zeros_like(v)
                expected[nodes[:,0]*s.dx>=.75,0]=scene.loading_rows[-1]['loading_velocity']
                row.update(min_particle_det_F=float(np.min(np.linalg.det(s.ptc_F.numpy()))),
                    grid_grip_velocity_error=float(np.max(abs(v[grip]-expected[grip]))))
                log.write(json.dumps(row,allow_nan=False)+'\n');status['completed_steps']=step
                if s.energy_ledger.audit_next:
                    base.write_json(dest/f'audit-{step:05d}.json',s.energy_ledger.audit)
                    np.savez_compressed(dest/f'audit-{step:05d}.npz',**s.energy_ledger.snapshot)
                if step % (count//20)==0:
                    frames.append(s.ptc_x.numpy().copy());Fs.append(s.ptc_F.numpy().copy());times.append(s.sim_time)
        status.update(run_completed=True,sim_time=s.sim_time,last_step_stats=s.last_step_stats)
    except Exception:
        status['error']=traceback.format_exc();print(status['error'],flush=True)
    finally:
        status['wall_seconds']=time.monotonic()-start;base.write_json(dest/'status.json',status)
        np.savez_compressed(dest/'frames.npz',time=times,x=frames,F=Fs)
        if scene is not None:(dest/'trajectory.csv').write_text(scene.solver.energy_ledger.csv())
    return status['run_completed']


def run(out,p):
    gate=json.loads((out/'static-summary.json').read_text())
    if not gate['candidate_static_gate']['residual_corotated']:raise RuntimeError('candidate failed static gate; loading blocked')
    if not json.loads((out/'tests.json').read_text())['required_checks_passed']:raise RuntimeError('tests required')
    records=[]
    for name in p['configs']:
        with (out/(name+'.log')).open('x') as log:
            r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_residual_history','worker','--case',name,'--output',str(out)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        records.append(dict(case=name,exit_code=r.returncode));print(name,r.returncode,flush=True)
        if r.returncode:break
    base.write_json(out/'batch.json',records);return len(records)==8 and all(r['exit_code']==0 for r in records)


def analyze(out,p):
    from benchmarks.aniso_dynamic_space import frame,stress,relative
    from benchmarks.aniso_dynamic_refine import compare
    data={};checks={};time_checks=[]
    for name in p['configs']:data[name],checks[name]=read_case(out/'cases'/name)
    for label in CASES:
        a,b=f'{label}-g17-coarse',f'{label}-g17-fine'
        t=np.array([r['time'] for r in data[a]]);t=t[t>=.005-1e-12]
        _,Fa=frame(out/'cases'/a);_,Fb=frame(out/'cases'/b);cfg=p['configs'][b]
        r=dict(case=label,reaction=compare(data[a],data[b],t),F_relative=relative(Fa-np.eye(3),Fb-np.eye(3)),
            P_relative=relative(stress(Fa,cfg),stress(Fb,cfg)),
            energy_relative=abs(data[a][-1]['elastic']-data[b][-1]['elastic'])/abs(data[b][-1]['elastic']))
        r['passed']=max(r['reaction']['relative'],r['F_relative'],r['P_relative'],r['energy_relative'])<=.02
        time_checks.append(r)
    audits=[json.loads(f.read_text()) for f in (out/'cases').glob('*/audit-*.json')]
    result=dict(completed=all(r['completed'] for r in checks.values()),checks=checks,time_checks=time_checks,
        physical_checks_passed=all(r['passed'] for r in checks.values()),
        time_sensitivity_passed=all(r['passed'] for r in time_checks),
        history_closure_max=max(r['frozen_F_max'] for r in audits),
        particle_trial_commit_max=max(r['particle_F_update_max'] for r in audits),
        long_loading_validated=False,default_changed=False,accuracy_certified=False)
    base.write_json(out/'summary.json',result)
    print(json.dumps({k:result[k] for k in ('completed','physical_checks_passed','time_sensitivity_passed','history_closure_max')},indent=2))
    return result['physical_checks_passed']


def main():
    wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('freeze','tests','static','run','worker','analyze'))
    p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--case');a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    if a.action=='freeze':freeze(out);return
    protocol=json.loads((out/'protocol.json').read_text())
    if protocol['source_sha256']!=hashes():raise RuntimeError('frozen candidate source changed')
    ok=base.tests(out,protocol) if a.action=='tests' else static(out,protocol) if a.action=='static' else run(out,protocol) if a.action=='run' else worker(out,protocol,a.case) if a.action=='worker' else analyze(out,protocol)
    raise SystemExit(0 if ok else 2)


if __name__=='__main__':main()
