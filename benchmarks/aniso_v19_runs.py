"""v19 endpoint impulse cycles; private scratch, immutable per-run source IDs."""
import argparse,json,os,sys,time,tempfile,subprocess,shutil,traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from benchmarks.aniso_v18_runs import ROOT,BASE,load,sha,write
from benchmarks.aniso_v17_modes import controlled_case
from benchmarks.aniso_carrier_joint import spectrum
from engine.aniso_phase1.endpoint_boundary import EndpointAVF
from engine.aniso_phase1.carrier_joint import gradient
from utils.resource_guard import inspect_storage
OUT=BASE/'v19'

def sources():
    names=['engine/aniso_phase1/endpoint_boundary.py','engine/aniso_phase1/carrier_driven.py','engine/aniso_phase1/carrier_joint.py','benchmarks/aniso_v19_runs.py','benchmarks/aniso_v17_modes.py']
    return {n:sha(ROOT/n) for n in names}

def freeze():
    assert not (OUT/'cycle-protocol.json').exists()
    old=load(BASE/'v18/artifact-sha256.json')
    for n,d in old.items():assert sha(ROOT/n)==d,n
    scratch=Path(tempfile.mkdtemp(prefix='mpm-lite-v19-',dir='/dev/shm'));scratch.chmod(0o700);assert not inspect_storage(scratch).paused
    write(OUT/'cycle-protocol.json',dict(scratch=str(scratch),source_sha256=sources(),prior_v18_verified=len(old),
        dt=[.0005,.00025,.000125,.0000625],duration=1.6,snapshots=[.05,.5,.6,.85,1.1,1.4,1.6],
        initial='F45 stress free with zero v,C; physical volume .046875',
        changes='Only endpoint boundary velocity impulse; all original material, quadrature and patch energy unchanged.',
        gates=dict(stress_relative=.02,reaction_relative=.02,energy_budget_J=1e-13,history=1e-10,endpoint_constraint=1e-9),
        reaction='sum midpoint and endpoint physical impulses divided by dt; no smoothing',
        constraint_loss='explicitly recorded; convergence checked independently of energy budget',production_default_changed=False))

def worker(level):
    p=load(OUT/'cycle-protocol.json');assert sources()==p['source_sha256'];dt=p['dt'][level];n=round(p['duration']/dt)
    dest=Path(p['scratch'])/f'cycle-L{level}';dest.mkdir();s,e,m,h,meta=controlled_case();s.v[:]=0;s.C[:]=0;so=EndpointAVF(s,e,m,h,mode='driven')
    P=np.empty((n+1,len(m),3,3));P[0]=e.evaluate(s.Y)['P'];status=dict(completed=False,steps=0,requested_steps=n);gates=[];start=time.monotonic();audit={round(t/dt) for t in p['snapshots']}
    try:
        assert spectrum(e.tangent(s.Y,so.geometry.Q))['passed']
        with (dest/'steps.jsonl').open('w',buffering=1) as log:
            for k in range(1,n+1):
                r=so.step(dt);status['steps']=k;P[k]=so.energy.evaluate(so.state.Y)['P'];log.write(json.dumps(r,allow_nan=False)+'\n')
                assert abs(r['budget_defect_J'])<1e-13 and r['history_commit_max']<1e-10
                assert r['endpoint_velocity_constraint']<1e-9 and r['grip_velocity_error']<1e-8
                if k in audit:
                    st=so.state;old=so.last['old'];g=so.geometry;end=so.endpoint_geometry;d=so.endpoint_data;gate=spectrum(e.tangent(st.Y,end.Q));assert gate['passed'];gates.append(dict(time=st.time,static=gate))
                    np.savez_compressed(dest/f'audit-{k:06d}.npz',x=st.x,Y=st.Y,v=st.v,C=st.C,time=st.time,F=gradient(e.B,st.Y),x_before=old.x,Y_before=old.Y,v_before=old.v,C_before=old.C,
                        J=g.J,Q=g.Q,q=g.metric,W=so.last['W'],force=so.last['force'],impulse=so.last['impulse'],endJ=end.J,endQ=end.Q,endq=end.metric,endpoint_impulse=d['impulse'],endpoint_delta=d['delta'],endpoint_lift=d['lift'],midpoint_trial_z=so.midpoint_trial_z)
                if k%max(1,n//32)==0:print(level,k,n,so.state.time,flush=True)
        status['completed']=True
    except Exception:status['error']=traceback.format_exc();print(status['error'],flush=True)
    np.savez_compressed(dest/'stress.npz',time=np.arange(status['steps']+1)*dt,P=P[:status['steps']+1]);status.update(seconds=time.monotonic()-start,dt=dt,static_gates=gates,source_sha256=sources());write(dest/'status.json',status)
    assert sources()==p['source_sha256'];return status['completed']

def run(jobs):
    p=load(OUT/'cycle-protocol.json');(OUT/'logs').mkdir(exist_ok=True)
    def one(level):
        name=f'cycle-L{level}'
        with (OUT/'logs'/f'{name}.log').open('x') as log:r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_v19_runs','worker','--level',str(level)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MPM_LITE_DATA_ROOT':p['scratch']})
        src=Path(p['scratch'])/name;dst=OUT/'cases'/name;shutil.copytree(src,dst);hashes={f.name:sha(f) for f in src.iterdir() if f.is_file()}
        for n,d in hashes.items():assert sha(dst/n)==d
        print(name,r.returncode,flush=True);return dict(case=name,exit_code=r.returncode,sha256=hashes)
    with ThreadPoolExecutor(max_workers=jobs) as pool:r=list(pool.map(one,range(3,-1,-1)))
    write(OUT/'cycle-batch.json',dict(completed=all(v['exit_code']==0 for v in r),records=r))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','worker','run']);p.add_argument('--level',type=int);p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='worker':sys.exit(0 if worker(a.level) else 2)
    else:run(a.jobs)
