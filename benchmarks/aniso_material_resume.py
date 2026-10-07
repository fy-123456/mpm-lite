"""Resume storage-paused v14 cases, verifying overlapping saved trajectories.

No frozen solver or acceptance tolerance is modified. Failed attempts remain in
v14-storage-attempt. The complete canonical trajectory keeps its original prefix
and regenerates the suffix from a material-history checkpoint.
"""
import hashlib,json,os,shutil,subprocess,sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import warp as wp
from benchmarks.aniso_material_history import ROOT,BASE,OUT,load,write,hashes,MaterialLedger
from benchmarks.aniso_unresolved_history import displacement,phase
from demos.aniso import Config,Scene
from engine.aniso_phase1.diagnostics import energy_density
from engine.aniso_phase1.tensile import grid_values
import engine.aniso_phase1.tensile as tensile
from utils.resource_guard import prepare_warp_cache
PAUSED=BASE/'v14-storage-attempt'

def worker(name):
    p=load(OUT/'protocol.json');assert hashes()==p['source_sha256'];cfg=p['configs'][name];src=PAUSED/'cases'/name;dest=OUT/'cases'/name;dest.mkdir(exist_ok=False)
    oldtext=(src/'steps.jsonl').read_text().splitlines();oldrows=[json.loads(l) for l in oldtext];start_step=max(int(f.stem.split('-')[-1]) for f in src.glob('audit-*.npz'));old_end=len(oldrows)
    path=src/f'audit-{start_step:05d}.npz'
    with np.load(path) as f:z={k:f[k].copy() for k in f.files}
    with np.load(src/'frames.npz') as f:previous={k:f[k].copy() for k in f.files}
    t0=oldrows[start_step-1]['time'];keep=previous['time']<=t0+1e-10;frames={k:list(a[keep]) for k,a in previous.items()};write(dest/'config.json',cfg)
    for f in src.glob('audit-*'):
        if int(f.stem.split('-')[-1])<=start_step:shutil.copyfile(f,dest/f.name)
    tensile.loading_displacement=displacement;scene=Scene(Config(**cfg),'cpu');s=scene.solver
    for a,b in [('x','x'),('F','F'),('v','velocity'),('C','C')]:getattr(s,'ptc_'+a).assign(z['particle_'+b+'_after'])
    s.ptc_reference_x.assign(z['particle_reference_x']);s.ptc_A0.assign(z['particle_A0']);s.sim_time=t0;s.sim_steps=start_step;scene.loading_work=oldrows[start_step-1]['loading_work']
    if cfg['stabilization']=='material_patch':s.enhancements.restart_state=dict(Y=z['marker_after'],X=z['patch_X'],ids=z['patch_ids'],P=z['patch_P'],weights=z['patch_weight'])
    ledger=MaterialLedger();ledger.rows=[dict(mechanical=oldrows[0]['mechanical']-oldrows[0]['cumulative_delta'])]+oldrows[:start_step];ledger.previous_elastic=oldrows[start_step-1]['elastic']
    psi=energy_density(z['center_F_committed'],z['center_A0'],s.aniso_params);ledger.history={tuple(c):float(e) for c,e in zip(z['coords'],psi)};ledger.previous_volumes={tuple(c):float(v) for c,v in zip(z['coords'],z['volume'])};s.energy_ledger=ledger
    audits={round(t/cfg['dt']) for t in p['snapshot_times']};every=round(p['frame_interval']/cfg['dt']);errors={k:0. for k in ('reaction','elastic','kinetic','stabilization','stage_budget','x','F','v','C')};began=time.monotonic();status=dict(run_completed=False,requested_steps=round(p['duration']/cfg['dt']),completed_steps=start_step,error=None,resumed_from=str(path.relative_to(ROOT)))
    try:
        with (dest/'steps.jsonl').open('x',buffering=1) as log:
            for line in oldtext[:start_step]:log.write(line+'\n')
            for step in range(start_step+1,status['requested_steps']+1):
                prepare_warp_cache('/tmp/mpm-lite-warp-cache');ledger.audit_next=step in audits;assert scene.step(),s.last_step_stats
                row=scene.metrics();nodes=ledger.nodes;v=grid_values(s,s.grid_v_new,nodes);grip=(nodes[:,0]*s.dx<=.25)|(nodes[:,0]*s.dx>=.75);expected=np.zeros_like(v);expected[nodes[:,0]*s.dx>=.75,0]=row['loading_velocity'];row.update(phase=phase(s.sim_time),min_particle_det_F=float(np.linalg.det(s.ptc_F.numpy()).min()),grid_grip_velocity_error=float(np.max(abs(v[grip]-expected[grip]))))
                if step<=old_end:
                    old=oldrows[step-1]
                    for a,b in [('reaction','right_force'),('elastic','elastic'),('kinetic','kinetic'),('stabilization','stabilization_energy'),('stage_budget','stage_budget_error')]:errors[a]=max(errors[a],abs(row[b]-old[b]))
                    assert max(errors[k] for k in ('reaction','elastic','kinetic','stabilization','stage_budget'))<1e-11,errors
                log.write(json.dumps(row,allow_nan=False)+'\n');status['completed_steps']=step
                if step in audits:write(dest/f'audit-{step:05d}.json',ledger.audit);np.savez_compressed(dest/f'audit-{step:05d}.npz',**ledger.snapshot)
                if step%every==0:
                    current=dict(time=s.sim_time,x=s.ptc_x.numpy().copy(),F=s.ptc_F.numpy().copy(),v=s.ptc_v.numpy().copy(),C=s.ptc_C.numpy().copy())
                    if step<=old_end:
                        j=int(np.argmin(abs(previous['time']-s.sim_time)));assert abs(previous['time'][j]-s.sim_time)<1e-12
                        for k in ('x','F','v','C'):errors[k]=max(errors[k],float(np.max(abs(current[k]-previous[k][j]))));assert errors[k]<1e-10,errors
                    for k,a in current.items():frames[k].append(a)
                if step==old_end:
                    write(dest/'resume-proof.json',dict(passed=True,checkpoint_step=start_step,interrupted_after_step=old_end,overlap_steps=old_end-start_step,errors=errors,snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),original_prefix_preserved=True));print(name,'overlap proof',errors,flush=True)
        status.update(run_completed=True,sim_time=s.sim_time,last_step_stats=s.last_step_stats)
    except Exception:status['error']=traceback.format_exc();print(status['error'],flush=True)
    finally:status['wall_seconds']=time.monotonic()-began;write(dest/'status.json',status);np.savez_compressed(dest/'frames.npz',**frames)
    return status['run_completed']

def main():
    wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
    if len(sys.argv)>1:raise SystemExit(0 if worker(sys.argv[1]) else 2)
    old=load(OUT/'batch.json');failed=[r['case'] for r in old if r['exit_code']];assert failed and all('StoragePaused' in load(OUT/'cases'/n/'status.json')['error'] for n in failed)
    PAUSED.mkdir(exist_ok=False);(PAUSED/'cases').mkdir();shutil.copyfile(OUT/'protocol.json',PAUSED/'protocol.json');(OUT/'batch.json').rename(PAUSED/'batch.json')
    for n in failed:(OUT/'cases'/n).rename(PAUSED/'cases'/n);(OUT/(n+'.log')).rename(PAUSED/(n+'.log'))
    for src,dst in [('/tmp/mpm-v14-run.log','driver.log'),('/tmp/mpm-v14-analysis.log','dependent-analysis.log')]:shutil.copyfile(src,PAUSED/dst)
    write(PAUSED/'resume-protocol.json',dict(source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),frozen_sources=hashes(),cases=failed,cause='5 GiB storage guard; reclaimed 1.2 GiB reproducible project uv cache; no solver/guard/tolerance modification',policy='preserve failed attempts; require row and saved-frame overlap agreement before accepting continuation'))
    def launch(n):
        with (OUT/(n+'.log')).open('x') as f:r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_material_resume',n],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        print(n,r.returncode,flush=True);return dict(case=n,exit_code=r.returncode)
    with ThreadPoolExecutor(max_workers=3) as pool:resumed=list(pool.map(launch,failed))
    results=[r for r in old if not r['exit_code']]+resumed;write(OUT/'batch.json',results);assert all(r['exit_code']==0 for r in results)
    write(OUT/'resume-summary.json',dict(passed=True,cases={n:load(OUT/'cases'/n/'resume-proof.json') for n in failed},source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),paused_attempt=str(PAUSED.relative_to(ROOT))))

if __name__=='__main__':main()
