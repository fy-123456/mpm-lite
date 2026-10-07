"""Bounded 0.05 s hold diagnostic after F45 ramp; verify restart first."""
from pathlib import Path
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import argparse,json,os,subprocess,sys,time
import numpy as np
import warp as wp
from demos.aniso import Config,Scene
from benchmarks.aniso_selective_history import PatchLedger
from benchmarks.aniso_mainline import write_json
from benchmarks.aniso_apic_frequency import Oracle
from benchmarks.aniso_dynamic_space import stress,relative
import engine.aniso_phase1.tensile as tensile
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'docs/results/lite-aniso-mainline';OUT=BASE/'v12/hold'


def read(p):return json.loads(p.read_text())


def load_scene(cfg,z,which):
    scene=Scene(Config(**cfg),'cpu');s=scene.solver
    for a,b in [('x','x'),('F','F'),('v','velocity'),('C','C')]:getattr(s,'ptc_'+a).assign(z['particle_'+b+'_'+which])
    s.ptc_reference_x.assign(z['particle_reference_x']);s.sim_time=.5-cfg['dt'] if which=='before' else .5
    s.energy_ledger=PatchLedger();s.energy_ledger.audit_next=False
    return scene


def worker(name):
    folder=BASE/('v11' if name=='baseline' else 'v12-transfer-controls');prefix='F45' if name=='baseline' else 'affine_only';case=prefix+'-fourth'
    cfg=read(folder/'protocol.json')['configs'][case]
    if name!='baseline':cfg={**cfg,'affine_flip_ratio':1.}
    with np.load(folder/'cases'/case/'audit-04000.npz') as f:z={k:f[k].copy() for k in f.files}
    scene=load_scene(cfg,z,'before');assert scene.step()
    err=max(float(np.max(abs(getattr(scene.solver,'ptc_'+a).numpy()-z['particle_'+b+'_after']))) for a,b in [('x','x'),('F','F'),('v','velocity'),('C','C')])
    assert err<1e-11,err
    import gc
    del scene;gc.collect()
    scene=load_scene(cfg,z,'after');s=scene.solver;original=tensile.loading_displacement
    tensile.loading_displacement=lambda t,*args,**kwargs:original(min(t,.5),*args,**kwargs)
    F0=s.ptc_F.numpy().copy();P0=stress(F0,cfg);rows=[];count=round(.05/cfg['dt']);start=time.monotonic()
    dest=OUT/name;dest.mkdir(parents=True,exist_ok=False)
    write_json(dest/'config.json',cfg)
    def sampled():
        o=Oracle(s.ptc_x.numpy(),s.ptc_m.numpy(),s.dx);v=o.apply(s.ptc_v.numpy(),s.ptc_C.numpy(),0.)['grid_v'];return float(.5*np.sum(o.mn[:,None]*v*v))
    initial=dict(kinetic_translation=s.energy_ledger.kinetic(s)[0],kinetic_affine=s.energy_ledger.kinetic(s)[1],raw_grid_kinetic=sampled())
    with (dest/'steps.jsonl').open('x',buffering=1) as log:
        for step in range(count):
            assert scene.step(),s.last_step_stats
            row=scene.metrics();row['min_particle_det_F']=float(np.linalg.det(s.ptc_F.numpy()).min());log.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
    F=s.ptc_F.numpy();P=stress(F,cfg);final=dict(kinetic_translation=rows[-1]['kinetic_translation'],kinetic_affine=rows[-1]['kinetic_affine'],raw_grid_kinetic=sampled(),elastic=rows[-1]['elastic'],reaction=rows[-1]['right_force'])
    result=dict(completed=True,steps=count,restart_replay_max=err,initial=initial,final=final,affine_kinetic_ratio=final['kinetic_affine']/initial['kinetic_affine'],stress_drift_relative=relative(P0,P),max_force_error=max(r['particle_momentum_balance_error_norm'] for r in rows),min_particle_det_F=min(r['min_particle_det_F'] for r in rows),max_boundary_speed=max(abs(r['loading_velocity']) for r in rows),wall_seconds=time.monotonic()-start,scope='0.05 s zero-grip-velocity hold after restart at t=0.5; not a long-term stability or time-convergence certificate')
    assert result['max_force_error']<1e-7 and result['min_particle_det_F']>0 and result['max_boundary_speed']==0
    write_json(dest/'summary.json',result);np.savez_compressed(dest/'end.npz',x=s.ptc_x.numpy(),F=F,v=s.ptc_v.numpy(),C=s.ptc_C.numpy());print(name,result,flush=True)


def main():
    wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache';p=argparse.ArgumentParser(__doc__);p.add_argument('--case');a=p.parse_args()
    if a.case:worker(a.case);return
    OUT.mkdir(exist_ok=False)
    write_json(OUT/'protocol.json',dict(duration=.05,dt=.000125,cases=['baseline','candidate'],steps_each=400,inputs='sealed v11 and full four-step v12 affine-only control terminal snapshots',purpose='diagnose observed rise in affine kinetic energy; no candidate change'))
    def launch(name):
        with (OUT/(name+'.log')).open('x') as log:
            r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_split_hold','--case',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        assert r.returncode==0,(name,r.returncode)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(launch,('baseline','candidate')))
    write_json(OUT/'summary.json',dict(completed=True,records={name:read(OUT/name/'summary.json') for name in ('baseline','candidate')}))

if __name__=='__main__':main()
