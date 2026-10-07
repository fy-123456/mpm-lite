"""Same v19 unloading/hold states: fixed vs moving, original vs Gauss inertia."""
import argparse,os,sys,json,time,subprocess,traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from benchmarks.aniso_v20_common import *
from engine.aniso_phase1.integrated_avf import IntegratedAVF
from engine.aniso_phase1.unresolved_velocity import pack
from engine.aniso_phase1.carrier_driven import kinetic_metric
from benchmarks.aniso_carrier_joint import spectrum

def worker(name):
    t,kind,motion,lev=name.split('-');t=float(t);level=int(lev);moving=motion=='moving';dt=[.0000625,.00003125,.000015625][level];n=round(.00625/dt);s,e,m,h,meta=snapshot(t)
    if kind=='gauss3':s,e,m,h=lift_state(s,e,m,h,meta,*gauss_sites(h,3))
    e.carrier_reference=meta['carrier_reference'];so=IntegratedAVF(s,e,m,h,moving=moving,condense=False);dest=OUT/'snapshot-runs'/name;dest.mkdir();P=[e.evaluate(s.Y)['P']];z=pack(s.v,s.C);initial=dict(kinetic_J=.5*float(np.sum(kinetic_metric(s.x,m,h)[:,None]*z*z)),momentum=(m[:,None]*s.v).sum(0).tolist(),potential_J=e.evaluate(s.Y)['U'])
    rows=[];start=time.monotonic();status=dict(completed=False,name=name,time=t,kind=kind,moving=moving,dt=dt,requested_steps=n,initial=initial,snapshot_sha256=meta['sha256'])
    try:
        for k in range(n):
            r=so.step(dt);rows.append(r);P.append(so.energy.evaluate(so.state.Y)['P'])
        gate=spectrum(e.tangent(so.state.Y,so.endpoint_geometry.Q));assert gate['passed'];status.update(completed=True,static_gate=gate)
    except Exception:status['error']=traceback.format_exc()
    (dest/'steps.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));np.savez_compressed(dest/'stress.npz',time=t+np.arange(len(P))*dt,P=np.array(P));st=so.state;np.savez_compressed(dest/'terminal.npz',x=st.x,Y=st.Y,v=st.v,C=st.C,time=st.time)
    status.update(steps=len(rows),seconds=time.monotonic()-start);write(dest/'status.json',status);print(status,flush=True);return status

def main():
    dest=OUT/'snapshot-runs';dest.mkdir(exist_ok=False);names=[f'{t:.2f}-{k}-{g}-{l}' for t in (.85,1.4) for k in ('sampled','gauss3') for g in ('fixed','moving') for l in range(3)]
    write(dest/'protocol.json',dict(duration=.00625,dt=[.0000625,.00003125,.000015625],names=names,material_and_patch_energy_unchanged=True,all_from_same_original_snapshots=True,no_condensation=True,initial_field='exact old values extended by common continuous piecewise trilinear interpolation; integral kinetic quantities separately reported, no renormalization'))
    def one(name):
        with (dest/(name+'.log')).open('x') as f:r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_v20_snapshot_runs','--name',name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        status=load(dest/name/'status.json');print(name,status['completed'],flush=True);return status
    with ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(one,names))
    write(OUT/'snapshot-batch.json',dict(completed=all(r['completed'] for r in rows),records=rows))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name');a=p.parse_args()
    if a.name:sys.exit(0 if worker(a.name)['completed'] else 2)
    else:main()
