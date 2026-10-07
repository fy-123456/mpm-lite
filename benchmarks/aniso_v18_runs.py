"""v18 moving snapshot and full prescribed-displacement cycle acceptance."""
import argparse,hashlib,json,os,subprocess,sys,tempfile,time,traceback,shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from benchmarks.aniso_carrier_joint import ROOT,BASE,load_case,spectrum
from benchmarks.aniso_v17_modes import controlled_case
from benchmarks.aniso_compatible_diagnosis import write
from engine.aniso_phase1.carrier_driven import DrivenAVF
from engine.aniso_phase1.carrier_avf import CarrierAVFSolver
from engine.aniso_phase1.carrier_joint import gradient,Geometry
from utils.resource_guard import inspect_storage
OUT=BASE/'v18'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def sources():
    files=list((ROOT/'engine').rglob('*.py'))+[ROOT/f for f in ('benchmarks/aniso_v18_runs.py','benchmarks/aniso_carrier_joint.py','benchmarks/aniso_v17_modes.py','tests/test_aniso_carrier_driven.py')]
    return {str(f.relative_to(ROOT)):sha(f) for f in files}

def initial(spec):
    if spec['kind']=='snapshot':return load_case(spec['start'])
    s,e,m,h,meta=controlled_case();s.v[:]=0;s.C[:]=0
    return s,e,m,h,dict(kind='stress_free',physical_volume=float(e.V.sum()),h=h,particles=len(m))

def freeze():
    assert not (OUT/'protocol.json').exists();cases={}
    for label,t in [('early',1.1),('late',1.6)]:
        for i,dt in enumerate((.000125,.0000625,.00003125,.000015625)):
            cases[f'{label}-moving-L{i}']=dict(kind='snapshot',start=t,label=label,dt=dt,duration=.05,mode='hold')
    for i,dt in enumerate((.00025,.000125,.0000625,.00003125)):
        cases[f'cycle-L{i}']=dict(kind='cycle',start=0.,label='cycle',dt=dt,duration=1.6,mode='driven')
    prior=load(BASE/'v17/artifact-sha256.json')
    for n,h in prior.items():assert sha(ROOT/n)==h,n
    scratch=Path(tempfile.mkdtemp(prefix='mpm-lite-v18-',dir='/dev/shm'));scratch.chmod(0o700);assert not inspect_storage(scratch).paused
    write(OUT/'protocol.json',dict(cases=cases,source_sha256=sources(),scratch=str(scratch),prior_v17_artifacts_verified=len(prior),device='cpu',precision='float64',
        cycle=dict(ramp=[0,.5],hold=[.5,.6],unload=[.6,1.1],final_hold=[1.1,1.6],peak_displacement=.005),
        snapshots=[.05,.25,.5,.55,.6,.85,1.1,1.2,1.4,1.6],stress_output='every accepted step, complete Piola tensor',
        boundary='same fixed Eulerian grip zones; driven midpoint velocity equals exact displacement difference/dt; full-J orthogonal residual reflection, not the stationary-snapshot admissible residual rule',
        acceptance=dict(stress_time=.02,work_defect_J=5e-13,path_error_J=5e-13,history=1e-10,grip_velocity=1e-8),
        input_snapshots={k:load_case(t)[-1] for k,t in [('early',1.1),('late',1.6)]},
        stationary_snapshot_backend='unchanged original CarrierAVFSolver, no fast path',algorithm='same original elastic energy and kinetic metric; equivalent LU/QR/BLAS paths and residual-converged chord iterations; no physical damping or regularization',default_changed=False))

def worker(name):
    p=load(OUT/'protocol.json');assert sources()==p['source_sha256'];spec=p['cases'][name]
    dest=Path(p['scratch'])/name;dest.mkdir(exist_ok=False);s,e,m,h,src=initial(spec);so=CarrierAVFSolver(s,e,m,h,moving=True) if spec['kind']=='snapshot' else DrivenAVF(s,e,m,h,mode=spec['mode'])
    count=round(spec['duration']/spec['dt']);P=np.empty((count+1,len(m),3,3));P[0]=e.evaluate(s.Y)['P'];status=dict(completed=False,steps=0,requested_steps=count);start=time.monotonic()
    write(dest/'config.json',dict(**spec,input=src));initial_gate=spectrum(e.tangent(s.Y,so.geometry.Q));assert initial_gate['passed'];gates=[]
    audits={round(t/spec['dt']) for t in p['snapshots']} if spec['kind']=='cycle' else {count//2,count}
    def audit(k):
        st=so.state;old=old_state;g=so.geometry
        W=(st.Y-old.Y)/spec['dt']
        gate=spectrum(e.tangent(st.Y,g.Q));gates.append(dict(step=k,gate=gate));assert gate['passed']
        np.savez_compressed(dest/f'audit-{k:06d}.npz',x=st.x,Y=st.Y,v=st.v,C=st.C,F=gradient(e.B,st.Y),time=st.time,
            x_before=old.x,Y_before=old.Y,v_before=old.v,C_before=old.C,nodes=g.nodes,E=g.E,N=g.N,Q=g.Q,J=g.J,q=g.metric,W=W)
    try:
        with (dest/'steps.jsonl').open('x',buffering=1) as log:
            for k in range(1,count+1):
                old_state=so.state.clone();r=so.step(spec['dt']);status['steps']=k;P[k]=so.energy.evaluate(so.state.Y)['P']
                if spec['kind']=='snapshot':r.update(dt=spec['dt'],boundary_work_J=0.,loading_speed=0.,reaction_N=0.,grip_velocity_error=0.,kinetic_projection_rank=so.geometry.Q.shape[1])
                log.write(json.dumps(r,allow_nan=False)+'\n')
                assert abs(r['kinetic_force_work_defect_J'])<5e-13,r
                assert abs(r['potential_quadrature_error_J'])<5e-13 and r['history_commit_max']<1e-10,r
                assert r['grip_velocity_error']<1e-8 and r['newton_residual']/spec['dt']<1e-7,r
                if k in audits:audit(k)
                if k%max(1,count//32)==0:print(name,k,count,'t',so.state.time,flush=True)
        status['completed']=True
    except Exception:status['error']=traceback.format_exc();print(status['error'],flush=True)
    np.savez_compressed(dest/'stress.npz',time=spec['start']+np.arange(status['steps']+1)*spec['dt'],P=P[:status['steps']+1])
    status.update(seconds=time.monotonic()-start,initial_massless=initial_gate,snapshot_gates=gates,terminal_time=so.state.time)
    write(dest/'status.json',status);assert sources()==p['source_sha256'];return status['completed']

def run(group,jobs):
    p=load(OUT/'protocol.json');assert load(OUT/'tests.json')['passed'];assert not (OUT/f'batch-{group}.json').exists();(OUT/'logs').mkdir(exist_ok=True)
    names=[n for n,s in p['cases'].items() if s['kind']==('cycle' if group=='cycle' else 'snapshot')];names.sort(key=lambda n:p['cases'][n]['dt'])
    if group=='cycle':assert load(OUT/'moving-acceptance.json')['passed'],'moving time check precedes full loading'
    def launch(name):
        with (OUT/'logs'/f'{name}.log').open('x') as log:
            r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_v18_runs','worker','--case',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MPM_LITE_DATA_ROOT':p['scratch']})
        src=Path(p['scratch'])/name;dest=OUT/'cases'/name;shutil.copytree(src,dest);hashes={f.name:sha(f) for f in src.iterdir() if f.is_file()}
        for path,digest in hashes.items():assert sha(dest/path)==digest
        row=dict(case=name,exit_code=r.returncode,files=hashes);print(name,r.returncode,flush=True);return row
    with ThreadPoolExecutor(max_workers=jobs) as pool:rows=list(pool.map(launch,names))
    write(OUT/f'batch-{group}.json',dict(completed=all(r['exit_code']==0 for r in rows),records=rows))

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['freeze','worker','run']);p.add_argument('--case');p.add_argument('--group',choices=['moving','cycle'],default='moving');p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='worker':raise SystemExit(0 if worker(a.case) else 2)
    else:run(a.group,a.jobs)
if __name__=='__main__':main()
