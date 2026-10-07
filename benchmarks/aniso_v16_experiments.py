"""Frozen v16 protocol, independent workers, and bounded counterfactuals."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile,time,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from benchmarks.aniso_carrier_joint import OUT,ROOT,BASE,load_case,cube,spectrum,write
from engine.aniso_phase1.carrier_joint import CarrierJointSolver,Geometry,gradient,MODES


def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sources():
    files=list((ROOT/'engine').rglob('*.py'))+list((ROOT/'demos').rglob('*.py'))
    files += [ROOT/f for f in ('benchmarks/aniso_carrier_joint.py','benchmarks/aniso_v16_experiments.py','benchmarks/aniso_v16_diagnostics.py','tests/test_aniso_carrier_joint.py')]
    return {str(p.relative_to(ROOT)):sha(p) for p in files}


def freeze():
    assert not (OUT/'protocol.json').exists()
    cases={}
    for t,label in ((1.1,'early_hold'),(1.6,'late_hold')):
        for moving in (False,True):
            for mode in ('legacy_split','common_split','joint'):
                for level,dt in (('coarse',.001),('fine',.0005),('finest',.00025),('fourth',.000125)):
                    name=f'{label}-'+('moving' if moving else 'frozen')+f'-{mode}-{level}'
                    cases[name]=dict(start=t,moving=moving,mode=mode,dt=dt,duration=.05)
    old=load(BASE/'v15/protocol.json')
    write(OUT/'protocol.json',dict(cases=cases,source_sha256=sources(),tests=old['tests']+['tests.test_aniso_carrier_joint'],device='cpu',cuda='not_run',
        scope='same saved unload/hold state; frozen geometry coefficients then moving particles; no complete load cycle or spatial acceptance',
        velocity_controls='All branches use pure incremental transfer, no PIC mixing and no separate null/weak filter. Existing-code baseline is independently checked for one actual Warp step.',
        energy='unchanged v15 carried material/patch potential; no added stiffness or mass shift',
        support='full-row-rank carrier map, quadratic-preserving right inverse, material independent DOFs',
        joint='kinetic pullback J.T M_APIC J; retain old velocity residual orthogonal to admissible JQ',
        source_snapshots={label:load_case(t)[-1] for t,label in ((1.1,'early_hold'),(1.6,'late_hold'))},
        acceptance=dict(history_max=1e-10,work_defect_J=5e-13,newton_force_N=1e-7,stress_relative_time=.02),default_changed=False))
    scratch=Path(tempfile.mkdtemp(prefix='mpm-lite-v16-',dir='/dev/shm'));scratch.chmod(0o700)
    write(OUT/'storage-protocol.json',dict(scratch=str(scratch),originals_remain_on_disk=True,guard_threshold_unchanged=True,regenerable_new_outputs_only=True))


def frame(s,e):
    return dict(time=s.time,x=s.x.copy(),Y=s.Y.copy(),v=s.v.copy(),C=s.C.copy(),F=gradient(e.B,s.Y))


def worker(name,out):
    p=load(OUT/'protocol.json');assert sources()==p['source_sha256'];spec=p['cases'][name]
    dest=out/'cases'/name;dest.mkdir(parents=True,exist_ok=False)
    s,e,m,h,source=load_case(spec['start']);solver=CarrierJointSolver(s,e,m,h,spec['mode'],spec['moving'])
    count=round(spec['duration']/spec['dt']);every=round(.005/spec['dt']);frames=[frame(s,e)];started=time.monotonic()
    write(dest/'config.json',dict(**spec,source=source));status=dict(completed=False,steps=0)
    try:
        with (dest/'steps.jsonl').open('x',buffering=1) as log:
            for k in range(1,count+1):
                old=solver.state.clone();r=solver.step(spec['dt']);log.write(json.dumps(r,allow_nan=False)+'\n');status['steps']=k
                assert r['history_commit_max']<1e-10
                if spec['mode']=='joint':assert abs(r['kinetic_force_work_defect_J'])<5e-13,r
                assert r['newton_residual']/spec['dt']<1e-7
                if k%every==0:frames.append(frame(solver.state,e))
            # Actual accepted last-step maps, not rebuilt maps at its endpoint.
            g=solver.geometry;last=solver.state
            np.savez_compressed(dest/'audit-terminal.npz',**frame(last,e),
                x_before=old.x,Y_before=old.Y,v_before=old.v,C_before=old.C,
                nodes=g.nodes,N=g.N,E=g.E,Q=g.Q,J=g.J,Jold=g.Jold,q=g.metric,
                mass=m,V=e.V,A=e.A,ids=e.ids,P=e.P,weights=e.weights,**{f'B{k}':b for k,b in enumerate(e.B)})
        status.update(completed=True,terminal_time=solver.state.time)
    except Exception:status['error']=traceback.format_exc();print(status['error'],flush=True)
    np.savez_compressed(dest/'frames.npz',**{k:np.array([f[k] for f in frames]) for k in frames[0]})
    status['seconds']=time.monotonic()-started;write(dest/'status.json',status)
    assert sources()==p['source_sha256'];return status['completed']


def run(jobs):
    p=load(OUT/'protocol.json');assert load(OUT/'support-gates/summary.json')['passed'];assert not (OUT/'batch.json').exists()
    scratch=Path(load(OUT/'storage-protocol.json')['scratch'])
    def launch(name):
        with (OUT/(name+'.log')).open('x') as log:
            r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_v16_experiments','worker','--case',name,'--output',str(scratch)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MPM_LITE_DATA_ROOT':str(scratch)})
        src=scratch/'cases'/name;dest=OUT/'cases'/name
        if src.exists():
            shutil.copytree(src,dest);files={str(f.relative_to(src)):sha(f) for f in src.rglob('*') if f.is_file()}
            assert all(sha(dest/n)==v for n,v in files.items())
        else:files={}
        record=dict(case=name,exit_code=r.returncode,files_verified=len(files));print(record,flush=True);return record
    with ThreadPoolExecutor(max_workers=jobs) as pool:records=list(pool.map(launch,p['cases']))
    result=dict(completed=all(r['exit_code']==0 for r in records),records=records,scratch_retained=str(scratch))
    write(OUT/'batch.json',result);return result['completed']


def small_steps():
    assert not (OUT/'small-step-controls.json').exists();records=[]
    for t in (1.1,1.6):
        s,e,m,h,source=load_case(t)
        for mode in MODES:
            for dt in (.001,.0005,.00025,.000125,1e-5,1e-6,1e-7):
                so=CarrierJointSolver(s,e,m,h,mode);r=so.step(dt)
                r.update(start=t,dt=dt,max_Y_increment=float(np.max(abs(so.state.Y-s.Y))),source=source)
                records.append(r)
    write(OUT/'small-step-controls.json',dict(completed=True,records=records,
        scope='One-step ablations, including intentionally inconsistent mass/adjoint mix and initial boundary-projection control. Neither rejected control is a recommended solver.'))
    print('one-step controls',len(records),flush=True)


def crossing():
    assert not (OUT/'moving-support.json').exists();v=np.array([.1,.025,-.02]);s,e,m,h=cube(velocity=v)
    so=CarrierJointSolver(s,e,m,h,'joint',True,False);records=[];ranks=[]
    for k in range(180):
        r=so.step(.005);t=so.state.time
        r.update(x_error=float(np.max(abs(so.state.x-s.x-t*v))),Y_error=float(np.max(abs(so.state.Y-s.Y-t*v))),
                 F_error=float(np.max(abs(gradient(e.B,so.state.Y)-np.eye(3)))),v_error=float(np.max(abs(so.state.v-v))),C_error=float(np.max(abs(so.state.C))))
        records.append(r)
        if k+1 in (1,2,60,120,180):ranks.append(dict(step=k+1,gate=spectrum(e.tangent(so.state.Y),6)))
    maxima={k:max(r[k] for r in records) for k in ('x_error','Y_error','F_error','v_error','C_error','momentum_change_norm')}
    assert max(maxima.values())<1e-10 and all(r['gate']['passed'] for r in ranks)
    write(OUT/'moving-support.json',dict(passed=True,steps=180,translation_cells=(v*.005*180/h).tolist(),
        grid_node_counts=sorted(set(r['grid_nodes'] for r in records)),independent_carriers=len(s.Y),max_errors=maxima,
        actual_massless_gates=ranks,records=records,scope='Actual moving x/Y/v/C with support rebuild, material independent DOFs; no position reset.'))
    print('moving support',maxima,flush=True)


def regression():
    from benchmarks.aniso_mainline import tests
    return tests(OUT,load(OUT/'protocol.json'))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('freeze','run','worker','small','crossing','tests'))
    p.add_argument('--case');p.add_argument('--output',type=Path);p.add_argument('--jobs',type=int,default=6);a=p.parse_args()
    if a.action=='freeze':freeze();return
    if a.action=='worker':ok=worker(a.case,a.output)
    elif a.action=='run':ok=run(a.jobs)
    elif a.action=='tests':ok=regression()
    elif a.action=='small':
        from benchmarks.aniso_v16_diagnostics import controls
        controls();ok=True
    else:crossing();ok=True
    raise SystemExit(0 if ok else 2)

if __name__=='__main__':main()
