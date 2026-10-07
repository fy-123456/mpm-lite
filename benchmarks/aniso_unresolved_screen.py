"""Identical-state hold screen and old v12 energy-stage attribution."""
import json,os,sys,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import warp as wp
from benchmarks.aniso_unresolved_history import ROOT,BASE,StageLedger,write,load
from benchmarks.aniso_split_hold import load_scene
from benchmarks.aniso_dynamic_space import stress,relative
from engine.aniso_phase1.unresolved_velocity import VelocityFilter
from engine.aniso_phase1.diagnostics import particle_kinetic
import engine.aniso_phase1.tensile as tensile
OUT=BASE/'v13-screen'


def diagnose():
    result={}
    for name in ('baseline','candidate'):
        rows=[json.loads(l) for l in (BASE/'v12/hold'/name/'steps.jsonl').read_text().splitlines()]
        keys=['p2c_delta','c2g_delta','boundary_projection_delta','solve_delta','g2p_delta','stabilization_rebuild_delta','delta_mechanical','budget_closure']
        sums={k:sum(r[k] for r in rows[1:]) for k in keys};sums['transfer_roundtrip_delta']=sums['p2c_delta']+sums['c2g_delta']+sums['g2p_delta'];sums['endpoint_change']=rows[-1]['mechanical']-rows[0]['mechanical'];result[name]=sums
    write(OUT/'v12-energy-attribution.json',dict(records=result,steps_each=399,interval=[.500125,.55],reason='exclude restart initialization; endpoints and all counted stages agree; large forward/backward transfer energies must be combined before attributing net growth'))
    cfg=load(BASE/'v12/protocol.json')['configs']['F45-fourth'];records=[]
    for file in sorted((BASE/'v12/cases').glob('F45-*/audit-*.npz')):
        with np.load(file) as z:
            x=z['particle_x_after'];v=z['particle_velocity_after'];C=z['particle_C_after'];m=z['particle_mass'];dt=load(file.parent/'config.json')['dt']
            for mode in ('null','weak'):
                p=VelocityFilter(x,m,.125,dt,np.sqrt(10)/.125,mode);a,b,d=p.apply(v,C);records.append(dict(case=file.parent.name,snapshot=file.name,**d))
    write(OUT/'same-input-spectral.json',dict(records=records,scope='16 frozen v12 states, 32 interventions: same positions, F and potential; measure only the velocity dissipation intervention'))


def worker(mode):
    cfg={**load(BASE/'v12/protocol.json')['configs']['F45-fourth'],'velocity_dissipation':mode}
    with np.load(BASE/'v12/cases/F45-fourth/audit-04000.npz') as f:z={k:f[k].copy() for k in f.files}
    scene=load_scene(cfg,z,'after');s=scene.solver;s.energy_ledger=StageLedger();original=tensile.loading_displacement;tensile.loading_displacement=lambda t,*a,**k:original(min(t,.5),*a,**k)
    out=OUT/mode;out.mkdir(exist_ok=False);P0=stress(s.ptc_F.numpy(),cfg);rows=[];start=time.monotonic()
    with (out/'steps.jsonl').open('x',buffering=1) as log:
        for step in range(1,401):
            s.energy_ledger.audit_next=step in (1,200,400);assert scene.step(),s.last_step_stats
            r=scene.metrics();log.write(json.dumps(r,allow_nan=False)+'\n');rows.append(r)
            if s.energy_ledger.audit_next:
                write(out/f'audit-{step:05d}.json',s.energy_ledger.audit);np.savez_compressed(out/f'audit-{step:05d}.npz',**s.energy_ledger.snapshot)
    P=stress(s.ptc_F.numpy(),cfg);initial=sum(particle_kinetic(z['particle_x_after'],z['particle_velocity_after'],z['particle_C_after'],z['particle_mass'],.125))
    keys=('transfer_roundtrip_delta','boundary_projection_delta','solve_delta','kinetic_metric_change','selective_dissipation_delta','stabilization_rebuild_delta','delta_mechanical')
    result=dict(completed=True,steps=400,mode=mode,stress_change=relative(P0,P),initial_kinetic=initial,final_kinetic=rows[-1]['kinetic'],final_affine_kinetic=rows[-1]['kinetic_affine'],final_elastic=rows[-1]['elastic'],final_reaction=rows[-1]['right_force'],stages={k:sum(r[k] for r in rows[1:]) for k in keys},stage_budget_max=max(abs(r['stage_budget_error']) for r in rows),momentum_error_max=max(r['particle_momentum_balance_error_norm'] for r in rows),wall_seconds=time.monotonic()-start)
    assert result['stage_budget_max']<1e-12 and result['momentum_error_max']<1e-7
    write(out/'summary.json',result);print(mode,result,flush=True)


def main():
    wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
    if len(sys.argv)>1:worker(sys.argv[1]);return
    OUT.mkdir(exist_ok=False);diagnose()
    write(OUT/'protocol.json',dict(input_snapshot='v12/cases/F45-fourth/audit-04000.npz',duration=.05,dt=.000125,modes=['none','null','weak'],same_input=True,coefficient='rate=sqrt(10)/0.125; weak energy visibility cutoff=.1; null removes all exact-null content',source='v13 formal protocol hashes'))
    def launch(mode):
        with (OUT/(mode+'.log')).open('x') as log:r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_unresolved_screen',mode],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        return dict(mode=mode,exit_code=r.returncode)
    with ThreadPoolExecutor(max_workers=3) as pool:r=list(pool.map(launch,('none','null','weak')))
    write(OUT/'batch.json',r);assert all(x['exit_code']==0 for x in r)

if __name__=='__main__':main()
