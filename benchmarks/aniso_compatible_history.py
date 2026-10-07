"""v15 gated common-input continuation and fresh-cycle history comparison.

All disk originals stay untouched. Regenerable new outputs go to a private
separate-filesystem directory, pass the unchanged storage guard, and are copied
back with SHA256 verification. No volatile storage replaces an original.
"""
import argparse, gc, hashlib, json, os, shutil, subprocess, sys, tempfile, time, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import warp as wp
from benchmarks.aniso_compatible_diagnosis import BASE, ROOT, maps, gradient, write
from benchmarks.aniso_unresolved_history import StageLedger, displacement, phase
from benchmarks.aniso_material_history import TESTS as OLD_TESTS
from engine.aniso_phase1.compatible_patch import pack_reference, carrier_gradient
from engine.aniso_phase1.diagnostics import energy_density
from engine.aniso_phase1.tensile import grid_values
from demos.aniso import Config, Scene
import engine.aniso_phase1.tensile as tensile
from utils.resource_guard import prepare_warp_cache
OUT=BASE/'v15'
LEVELS={'coarse':.001,'fine':.0005,'finest':.00025,'fourth':.000125}


def load(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def sources():
    files=list((ROOT/'engine').rglob('*.py'))+list((ROOT/'demos').rglob('*.py'))
    files+=list((ROOT/'benchmarks').glob('aniso_compatible*.py'))+[ROOT/'tests/test_aniso_compatible_patch.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in files}


def freeze():
    if (OUT/'protocol.json').exists():raise RuntimeError('preserve frozen protocol')
    old=load(BASE/'v14/protocol.json');cases={}
    for start,label in ((.85,'unload'),(1.1,'early_hold'),(1.6,'late_hold')):
        for mode,kind in [('baseline','material_patch'),('compatible','compatible_patch')]:
            for level,dt in LEVELS.items():
                cfg={**old['configs']['material-'+level],'stabilization':kind}
                cases[f'{label}-{mode}-{level}']=dict(config=cfg,start=start,duration=.05,source_case='material-fourth',source_step=round(start/.000125))
    # Full fresh trajectories screen accumulated history; two levels are not
    # reported as a completed four-level global accuracy acceptance.
    for level in ('coarse','fine'):
        cases['cycle-compatible-'+level]=dict(config={**old['configs']['material-'+level],'stabilization':'compatible_patch'},start=0.,duration=1.6)
    write(OUT/'protocol.json',dict(cases=cases,source_sha256=sources(),tests=OLD_TESTS+['tests.test_aniso_compatible_patch'],cuda='not_run',device='cpu',precision='float64',
        diagnostics='24 saved v14 states; 48 original-Hessian massless gates',
        same_input='three fourth-level saved states; every timestep/mode starts with identical x/F/v/C/Y and material/stabilization energy',
        branch_history='R=(G0 Y_start)^-1 F_start retains existing particle history; never reset F to G0Y',
        durations='24 continuations of 0.05 s, plus two fresh full 1.6 s cycles; not a global four-dt acceptance',
        acceptance=dict(history=1e-10,stage_budget_J=1e-12,rebuild_J=1e-14,force_balance_N=1e-7,relative_stress_time=.02),
        default_changed=False,coefficient_changed=False,velocity_dissipation='strict null unchanged',source_v14_protocol_sha256=digest(BASE/'v14/protocol.json')))


class CommonLedger(StageLedger):
    def finish(self,s):
        super().finish(s)
        # Parent's legacy snapshot audit is deliberately disabled: it assumes
        # F_next=(I+dt*G_current*v)F. Audit the actual new map independently here.
        e=s.enhancements;row=self.rows[-1]
        v=grid_values(s,s.grid_v_new,s.mapped_nodes)
        predicted=s.mapped_old_F+s.dt*np.stack([g@v for g in s.local_host_maps],axis=2)
        row['particle_trial_commit_max']=float(np.max(abs(s.ptc_F.numpy()-predicted)))
        row['carrier_commit_max']=float(np.max(abs(e.origin.numpy()-e.last_origin-s.dt*(e.N@v))))
        row['center_trial_commit_max']=float(np.max(abs(s.mapped_trial.numpy()-(s.mapped_W@s.ptc_F.numpy().reshape(s.n_ptc,9)).reshape(-1,3,3))))
        if s.stabilization=='compatible_patch':
            row['common_history_max']=float(np.max(abs(s.ptc_F.numpy()-carrier_gradient(s.compatible_G,e.origin.numpy())@s.compatible_R)))
        else:row['common_history_max']=0.  # baseline's actual defect is audited in analysis, not treated as closure.


def restore(scene,spec):
    src=BASE/'v14/cases'/spec['source_case'];path=src/f'audit-{spec["source_step"]:05d}.npz'
    with np.load(path) as f:z={k:f[k].copy() for k in f.files}
    rows=[json.loads(l) for l in (src/'steps.jsonl').read_text().splitlines()];row=rows[spec['source_step']-1]
    s=scene.solver
    for a,b in [('x','x'),('F','F'),('v','velocity'),('C','C')]:getattr(s,'ptc_'+a).assign(z['particle_'+b+'_after'])
    s.ptc_reference_x.assign(z['particle_reference_x']);s.ptc_A0.assign(z['particle_A0'])
    s.sim_time=spec['start'];s.sim_steps=round(spec['start']/s.dt);scene.loading_work=row['loading_work']
    state=dict(Y=z['marker_after'],X=z['patch_X'],ids=z['patch_ids'],P=z['patch_P'],weights=z['patch_weight'])
    T0,G0=maps(z['particle_reference_x'],np.rint(z['patch_X']/s.dx).astype(int),s.dx)
    R=np.linalg.solve(gradient(G0,z['marker_after']),z['particle_F_after'])
    if s.stabilization=='compatible_patch':state.update(pack_reference(G0,R))
    s.enhancements.restart_state=state
    ledger=CommonLedger();ledger.rows=[{**row,'cumulative_delta':0.}];ledger.previous_elastic=row['elastic']
    psi=energy_density(z['center_F_committed'],z['center_A0'],s.aniso_params)
    ledger.history={tuple(c):float(e) for c,e in zip(z['coords'],psi)}
    ledger.previous_volumes={tuple(c):float(v) for c,v in zip(z['coords'],z['volume'])};s.energy_ledger=ledger
    return z,T0,G0,R,row,digest(path)


def capture_frame(s, carrier_positions=None):
    """Own every saved array; Warp CPU numpy() can return a live memory view."""
    Y=s.enhancements.origin.numpy() if carrier_positions is None else carrier_positions
    return dict(time=float(s.sim_time),Y=np.array(Y,copy=True),
                **{key:getattr(s,'ptc_'+key).numpy().copy() for key in ('x','F','v','C')})


def worker(out,name):
    p=load(OUT/'protocol.json');assert sources()==p['source_sha256'],'frozen source changed'
    wp.config.kernel_cache_dir=str(out/'mpm-lite-warp-cache')
    spec=p['cases'][name];cfg=spec['config'];dest=out/'cases'/name;dest.mkdir(parents=True,exist_ok=False)
    write(dest/'config.json',cfg);tensile.loading_displacement=displacement
    status=dict(completed=False,error=None,steps=0);began=time.monotonic();frames=[];scene=None
    try:
        scene=Scene(Config(**cfg),'cpu');s=scene.solver;s.energy_ledger=CommonLedger()
        if spec['start']:
            z,T0,G0,R,row,sha=restore(scene,spec)
            initial=capture_frame(s,z['marker_after'])
            np.savez_compressed(dest/'initial.npz',**initial)
            write(dest/'initial.json',dict(input_sha256=sha,material_J=row['elastic']-row['stabilization_energy'],stabilization_J=row['stabilization_energy'],kinetic_J=row['kinetic'],history_reconstruction_error=float(np.max(abs(gradient(G0,z['marker_after'])@R-z['particle_F_after'])))))
            frames.append(initial)
        else:T0=G0=R=None
        count=round(spec['duration']/cfg['dt']);every=round(.005/cfg['dt']) if spec['start'] else round(.025/cfg['dt'])
        audit_steps={count} if spec['start'] else {round(t/cfg['dt']) for t in (.5,.85,1.1,1.2,1.4,1.6)}
        with (dest/'steps.jsonl').open('x',buffering=1) as log:
            for step in range(1,count+1):
                wp.config.kernel_cache_dir=prepare_warp_cache(wp.config.kernel_cache_dir)
                assert scene.step(),s.last_step_stats
                r=scene.metrics();e=s.enhancements
                if G0 is None:
                    # Fresh material reference exactly equals carrier initial nodes.
                    T0,G0=maps(scene.reference,np.rint(e.reference_X/s.dx).astype(int),s.dx);R=np.broadcast_to(np.eye(3),(s.n_ptc,3,3)).copy()
                    frames.append(dict(time=0.,x=scene.reference.copy(),F=np.broadcast_to(np.eye(3),(s.n_ptc,3,3)).copy(),v=np.zeros((s.n_ptc,3)),C=np.zeros((s.n_ptc,3,3)),Y=e.reference_X.copy()))
                diff=s.ptc_F.numpy()-gradient(G0,e.origin.numpy())@R
                r.update(branch_history_rms=float(np.sqrt(np.mean(np.sum(diff**2,axis=(1,2))))),
                         branch_position_rms=float(np.sqrt(np.mean(np.sum((s.ptc_x.numpy()-T0@e.origin.numpy())**2,axis=1)))),
                         phase=phase(s.sim_time),min_particle_det_F=float(np.linalg.det(s.ptc_F.numpy()).min()))
                log.write(json.dumps(r,allow_nan=False)+'\n');status['steps']=step
                assert abs(r['stage_budget_error'])<1e-12 and abs(r['stabilization_rebuild_delta'])<1e-14,r
                assert max(r[k] for k in ('particle_trial_commit_max','carrier_commit_max','center_trial_commit_max','common_history_max'))<1e-10
                assert r['free_force_residual_norm']<1e-7
                if step%every==0 or step==count:
                    frames.append(capture_frame(s))
                if step in audit_steps:
                    state=e.state();state.update(particle_F=s.ptc_F.numpy(),particle_x=s.ptc_x.numpy(),particle_F_before=s.mapped_old_F,
                        particle_x_before=s.mapped_old_x,particle_A=s.ptc_A0.numpy(),particle_V=s.ptc_vol0.numpy(),particle_reference_x=s.ptc_reference_x.numpy(),
                        grid_nodes=s.mapped_nodes,center_coords=s.mapped_centers,grid_v=grid_values(s,s.grid_v_new,s.mapped_nodes),grid_raw=grid_values(s,s.grid_v_raw,s.mapped_nodes),Y_before=e.last_origin,
                        mass=s.energy_ledger.node_m,ledger_nodes=s.energy_ledger.nodes,center_trial=s.mapped_trial.numpy(),center_W=s.mapped_W.toarray(),
                        branch_R=R,**{f'branch_G{k}':g.toarray() for k,g in enumerate(G0)})
                    np.savez_compressed(dest/f'audit-{step:05d}.npz',**state)
                if step%max(count//4,1)==0:print(name,step,'/',count,flush=True)
        status.update(completed=True,time=s.sim_time,last_step_stats=s.last_step_stats)
    except Exception:status['error']=traceback.format_exc();print(status['error'],flush=True)
    finally:
        status['wall_seconds']=time.monotonic()-began;write(dest/'status.json',status)
        if frames:np.savez_compressed(dest/'frames.npz',**{k:np.array([f[k] for f in frames]) for k in frames[0]})
    return status['completed']


def run(jobs):
    p=load(OUT/'protocol.json');assert load(OUT/'tests.json')['required_checks_passed']
    gates=load(OUT/'controls/summary.json');assert gates['all_massless_passed'] and gates['all_fd_refinement_passed']
    assert not (OUT/'batch.json').exists()
    temporary=Path(tempfile.mkdtemp(prefix='mpm-lite-v15-new-',dir='/dev/shm'));temporary.chmod(0o700)
    cache_seed=Path(os.environ.get('MPM_LITE_V15_SEED_CACHE','/tmp/mpm-lite-warp-cache'))
    if cache_seed.exists():shutil.copytree(cache_seed,temporary/'mpm-lite-warp-cache')
    write(OUT/'storage-protocol.json',dict(temporary=str(temporary),originals_remain_on_disk=True,guard_unchanged=True,new_regenerable_outputs_only=True))
    def launch(name):
        with (OUT/(name+'.log')).open('x') as log:
            result=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_compatible_history','worker','--case',name,'--output',str(temporary)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'MPM_LITE_DATA_ROOT':str(temporary),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        src=temporary/'cases'/name;dst=OUT/'cases'/name
        if src.exists():
            shutil.copytree(src,dst);files={str(f.relative_to(src)):digest(f) for f in src.rglob('*') if f.is_file()}
            assert all(digest(dst/k)==v for k,v in files.items())
        else:files={}
        record=dict(case=name,exit_code=result.returncode,files_verified=len(files));print(record,flush=True);return record
    names=['cycle-compatible-fine','cycle-compatible-coarse']+[n for n in p['cases'] if not n.startswith('cycle-')]
    with ThreadPoolExecutor(max_workers=jobs) as pool:results=list(pool.map(launch,names))
    write(OUT/'batch.json',dict(completed=all(r['exit_code']==0 for r in results),records=results,temporary_retained=str(temporary)))
    return all(r['exit_code']==0 for r in results)


def main():
    wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache';a=argparse.ArgumentParser(__doc__)
    a.add_argument('action',choices=('freeze','tests','run','worker'));a.add_argument('--case');a.add_argument('--output',type=Path,default=OUT);a.add_argument('--jobs',type=int,default=4);args=a.parse_args()
    if args.action=='freeze':freeze();return
    if args.action=='tests':
        from benchmarks.aniso_mainline import tests
        ok=tests(OUT,load(OUT/'protocol.json'))
    elif args.action=='worker':ok=worker(args.output,args.case)
    else:ok=run(args.jobs)
    raise SystemExit(0 if ok else 2)

if __name__=='__main__':main()
