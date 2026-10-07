"""Separated inertia / variational condensation full-cycle ablations (v20)."""
import argparse,json,os,sys,time,subprocess,shutil,traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from benchmarks.aniso_v20_common import *
from benchmarks.aniso_carrier_joint import spectrum
from engine.aniso_phase1.integrated_avf import IntegratedAVF
from engine.aniso_phase1.carrier_joint import gradient

def sources():
    names=['engine/aniso_phase1/integrated_avf.py','engine/aniso_phase1/separate_kinetic.py','benchmarks/aniso_v20_common.py','benchmarks/aniso_v20_runs.py']
    return {n:sha(ROOT/n) for n in names}

def setup(kind):
    s,e,m,h,meta=controlled_case();s.v[:]=0;s.C[:]=0
    if kind.startswith('gauss3'):s,e,m,h=lift_state(s,e,m,h,meta,*gauss_sites(h,3))
    e.carrier_reference=meta['carrier_reference'];return s,e,m,h,meta

def freeze():
    assert not (OUT/'cycle-protocol.json').exists()
    cases=[dict(name=f'gauss3-condensed-L{k}',kind='gauss3-condensed',dt=dt) for k,dt in enumerate([.0005,.00025,.000125,.0000625])]
    # Two full-cycle ablations separate inertia quadrature from eliminating
    # material-invisible coordinates. The four condensed levels remain primary.
    cases += [dict(name=kind+'-ablation',kind=kind,dt=.000125) for kind in ('sampled-condensed','gauss3-original')]
    write(OUT/'cycle-protocol.json',dict(cases=cases,duration=1.6,snapshots=[.05,.5,.6,.85,1.1,1.4,1.6],source_sha256=sources(),reaction_gate=.02,stress_gate=.02,raw_reaction_only=True,initial='zero velocity and stress-free; no old-state projection',condensation='exact stationary elimination of material-null coordinates of original potential; dynamics is a new candidate, not asserted equivalent',quadrature='fixed positive Gauss3 kinetic particles; moving cut-cell oracle verified independently, no per-step quadrature remeshing'))

def worker(name):
    protocol=load(OUT/'cycle-protocol.json');assert sources()==protocol['source_sha256'];p=next(p for p in protocol['cases'] if p['name']==name);dt=p['dt'];n=round(protocol['duration']/dt);kind=p['kind']
    dest=Path(load(OUT/'protocol.json')['scratch'])/name;dest.mkdir();s,e,m,h,meta=setup(kind);so=IntegratedAVF(s,e,m,h,condense=kind.endswith('condensed'))
    P=np.empty((n+1,len(e.V),3,3));P[0]=e.evaluate(s.Y)['P'];status=dict(completed=False,steps=0,requested_steps=n);gates=[];start=time.monotonic();audit={round(t/dt) for t in protocol['snapshots']}
    try:
        assert spectrum(e.tangent(s.Y,so.geometry.Q))['passed']
        with (dest/'steps.jsonl').open('x',buffering=1) as log:
            for k in range(1,n+1):
                r=so.step(dt);status['steps']=k;P[k]=so.energy.evaluate(so.state.Y)['P'];log.write(json.dumps(r,allow_nan=False)+'\n')
                assert r['history_commit_max']<1e-10 and r['endpoint_velocity_constraint']<1e-9
                if k in audit:
                    st=so.state;g=so.endpoint_geometry;gate=spectrum(e.tangent(st.Y,g.Q));assert gate['passed'];gates.append(dict(time=st.time,static=gate))
                    np.savez_compressed(dest/f'audit-{k:06d}.npz',x=st.x,Y=st.Y,v=st.v,C=st.C,time=st.time,F=gradient(e.B,st.Y),Q=g.Q)
                if k%max(1,n//32)==0:print(name,k,n,so.state.time,flush=True)
        status['completed']=True
    except Exception:status['error']=traceback.format_exc();print(status['error'],flush=True)
    np.savez_compressed(dest/'stress.npz',time=np.arange(status['steps']+1)*dt,P=P[:status['steps']+1]);status.update(seconds=time.monotonic()-start,dt=dt,static_gates=gates,source_sha256=sources(),kind=kind,material_points=len(e.V),kinetic_points=len(m));write(dest/'status.json',status)
    assert sources()==protocol['source_sha256'];return status['completed']

def run(jobs):
    p=load(OUT/'protocol.json');cases=load(OUT/'cycle-protocol.json')['cases'];(OUT/'logs').mkdir(exist_ok=True)
    def one(a):
        name=a['name']
        with (OUT/'logs'/f'{name}.log').open('x') as log:r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_v20_runs','worker','--name',name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MPM_LITE_DATA_ROOT':p['scratch']})
        src=Path(p['scratch'])/name;dst=OUT/'cases'/name;shutil.copytree(src,dst);hashes={f.name:sha(f) for f in src.iterdir() if f.is_file()}
        for n,d in hashes.items():assert sha(dst/n)==d
        print(name,r.returncode,flush=True);return dict(case=name,exit_code=r.returncode,sha256=hashes)
    with ThreadPoolExecutor(max_workers=jobs) as pool:records=list(pool.map(one,sorted(cases,key=lambda a:a['dt'])))
    write(OUT/'cycle-batch.json',dict(completed=all(r['exit_code']==0 for r in records),records=records))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','worker','run']);p.add_argument('--name');p.add_argument('--jobs',type=int,default=6);a=p.parse_args()
    if a.action=='freeze':freeze()
    elif a.action=='worker':sys.exit(0 if worker(a.name) else 2)
    else:run(a.jobs)
