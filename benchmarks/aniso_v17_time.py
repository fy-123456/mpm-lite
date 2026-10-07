"""v17 immutable same-state dense-output AVF time refinement."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback,tempfile,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from benchmarks.aniso_carrier_joint import ROOT,BASE,load_case,spectrum
from benchmarks.aniso_compatible_diagnosis import write
from engine.aniso_phase1.carrier_avf import CarrierAVFSolver
from engine.aniso_phase1.unresolved_velocity import pack
from benchmarks.aniso_v17_modes import snapshot_model
OUT=BASE/'v17'
DTS=(.00025,.000125,.0000625,.00003125,.000015625,.0000078125)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def sources():
    names=list((ROOT/'engine').rglob('*.py'))+[ROOT/'benchmarks/aniso_carrier_joint.py',ROOT/'benchmarks/aniso_v17_modes.py',ROOT/'benchmarks/aniso_v17_time.py',ROOT/'tests/test_aniso_v17_modes.py',ROOT/'benchmarks/aniso_v17_diagnosis.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in names}

def freeze():
    assert not (OUT/'protocol.json').exists()
    prior=load(BASE/'v16/artifact-sha256.json')
    for name,digest in prior.items():assert sha(ROOT/name)==digest,name
    cases={f'{label}-L{i}':dict(start=t,dt=dt,duration=.05,label=label) for label,t in [('early',1.1),('late',1.6)] for i,dt in enumerate(DTS)}
    scratch=Path(tempfile.mkdtemp(prefix='mpm-lite-v17-',dir='/dev/shm'));scratch.chmod(0o700)
    write(OUT/'protocol.json',dict(cases=cases,source_sha256=sources(),scratch=str(scratch),v16_files_verified=len(prior),
        scope='unchanged v16 Gauss2 AVF, fixed geometry, same saved input, every-step stress and modal output; no physical damping, no full-cycle/spatial accuracy claim',
        inputs={k:load_case(t)[-1] for k,t in [('early',1.1),('late',1.6)]},duration=.05,stress_threshold=.02,
        storage='new regenerable data in private tmpfs, copied with SHA verification; originals retained; existing 5 GiB guard unchanged',default_changed=False))

def worker(name):
    p=load(OUT/'protocol.json');assert sources()==p['source_sha256'];spec=p['cases'][name]
    dest=Path(p['scratch'])/name;dest.mkdir(exist_ok=False);s,e,m,h,src=load_case(spec['start']);so=CarrierAVFSolver(s,e,m,h)
    model=snapshot_model(spec['start']);count=round(spec['duration']/spec['dt']);nt=count+1
    P=np.empty((nt,len(s.x),3,3));coords=np.empty((nt,len(model.omega)));vel=np.empty_like(coords)
    Ysample=[];state=so.state;P[0]=e.evaluate(state.Y)['P'];coords[0],vel[0]=model.project(state.Y,pack(state.v,state.C))
    started=time.monotonic();status=dict(completed=False,steps=0);total0=e.evaluate(s.Y)['U']+.5*np.sum(so.geometry.metric[:,None]*pack(s.v,s.C)**2)
    try:
        with (dest/'steps.jsonl').open('x',buffering=1) as f:
            for k in range(1,nt):
                old=so.state.clone();row=so.step(spec['dt']);state=so.state;status['steps']=k
                P[k]=e.evaluate(state.Y)['P'];coords[k],vel[k]=model.project(state.Y,pack(state.v,state.C));f.write(json.dumps(row,allow_nan=False)+'\n')
                assert abs(row['kinetic_force_work_defect_J'])<5e-13,row
                assert abs(row['potential_quadrature_error_J'])<5e-13,row
                assert row['history_commit_max']<1e-10 and row['newton_residual']/spec['dt']<1e-7,row
                if k%max(1,count//10)==0:print(name,k,count,flush=True)
        gate=spectrum(e.tangent(state.Y,so.geometry.Q));assert gate['passed']
        np.savez_compressed(dest/'terminal.npz',x=state.x,Y=state.Y,v=state.v,C=state.C,F=e.evaluate(state.Y)['F'],
            x_before=old.x,Y_before=old.Y,v_before=old.v,C_before=old.C)
        status.update(completed=True,gate=gate,terminal_total_change_J=row['total_J']-total0,initial_total_J=float(total0))
    except Exception:status['error']=traceback.format_exc();print(status['error'],flush=True)
    end=status['steps']+1
    np.savez_compressed(dest/'series.npz',time=np.arange(end)*spec['dt'],P=P[:end],modal_q=coords[:end],modal_v=vel[:end])
    status['seconds']=time.monotonic()-started;write(dest/'status.json',status)
    assert sources()==p['source_sha256'];return status['completed']

def run(jobs):
    p=load(OUT/'protocol.json');assert load(OUT/'tests.json')['passed'];assert not (OUT/'batch.json').exists()
    (OUT/'logs').mkdir(exist_ok=True)
    def launch(name):
        with (OUT/'logs'/f'{name}.log').open('x') as f:
            r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_v17_time','worker','--case',name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        src=Path(p['scratch'])/name;dest=OUT/'cases'/name;shutil.copytree(src,dest)
        manifest={str(f.relative_to(src)):sha(f) for f in src.iterdir() if f.is_file()}
        for path,digest in manifest.items():assert sha(dest/path)==digest
        row=dict(case=name,exit_code=r.returncode,files=manifest);print(row['case'],row['exit_code'],flush=True);return row
    # Longest cases first to reduce the tail without oversubscribing BLAS.
    names=sorted(p['cases'],key=lambda n:p['cases'][n]['dt'])
    with ThreadPoolExecutor(max_workers=jobs) as pool:rows=list(pool.map(launch,names))
    write(OUT/'batch.json',dict(completed=all(r['exit_code']==0 for r in rows),records=rows))
    assert all(r['exit_code']==0 for r in rows)

def main():
    a=argparse.ArgumentParser(__doc__);a.add_argument('action',choices=['freeze','run','worker']);a.add_argument('--case');a.add_argument('--jobs',type=int,default=4);v=a.parse_args()
    if v.action=='freeze':freeze()
    elif v.action=='worker':raise SystemExit(0 if worker(v.case) else 2)
    else:run(v.jobs)
if __name__=='__main__':main()
