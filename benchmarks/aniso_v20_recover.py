"""Resume failed runs from exact archived states; preserve all failed artifacts.

Only the Newton line-search merit fallback changes. Final residual tolerance,
energy ledger, dt, all potentials, and constraint impulse are unchanged.
"""
import time,json,traceback,shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from benchmarks.aniso_v20_common import *
from benchmarks.aniso_v20_runs import setup
from benchmarks.aniso_carrier_joint import spectrum
from engine.aniso_phase1.robust_integrated_avf import RobustIntegratedAVF
from engine.aniso_phase1.carrier_joint import State,gradient

def recover(p):
    base=OUT/'fast-cycle';src=base/'cases'/p['name'];dest=OUT/'recovered'/p['name'];dest.mkdir(parents=True,exist_ok=False);old=load(src/'status.json');assert not old['completed'];assert 'AVF residual failure' in old['error'] or 'AVF line search failed' in old['error'];checkpoint=max(src.glob('audit-*.npz'));k0=int(checkpoint.stem.split('-')[-1]);dt=p['dt'];n=round(1.6/dt);s,e,m,h,meta=setup(p['kind'])
    with np.load(checkpoint) as z:s=State(z['x'],z['Y'],z['v'],z['C'],float(z['time']))
    so=RobustIntegratedAVF(s,e,m,h,condense=p['kind'].endswith('condensed'));so.steps=k0;P=np.empty((n+1,len(e.V),3,3))
    with np.load(src/'stress.npz') as z:P[:k0+1]=z['P'][:k0+1]
    np.testing.assert_allclose(P[k0],e.evaluate(s.Y)['P'],atol=1e-12);prefix=(src/'steps.jsonl').read_text().splitlines()[:k0];gates=[v for v in old['static_gates'] if v['time']<=s.time+1e-10];audit={round(t/dt) for t in [.05,.5,.6,.85,1.1,1.4,1.6]}
    for f in src.glob('audit-*.npz'):
        if int(f.stem.split('-')[-1])<=k0:shutil.copy2(f,dest/f.name)
    status=dict(completed=False,steps=k0,requested_steps=n,dt=dt,kind=p['kind'],resumed_from=str(checkpoint.relative_to(ROOT)),resumed_from_sha256=sha(checkpoint),prefix_steps=k0,previous_failure_steps=old['steps'],previous_status_sha256=sha(src/'status.json'),change='Near-root residual-decrease line search fallback only; final residual remains 1e-17; all dt, energies, histories unchanged.',newton_predictor='reinitialized, physical x/Y/v/C retained exactly',source_sha256={q:sha(ROOT/q) for q in ['engine/aniso_phase1/robust_integrated_avf.py','benchmarks/aniso_v20_recover.py']});start=time.monotonic()
    try:
        with (dest/'steps.jsonl').open('x',buffering=1) as log:
            log.write('\n'.join(prefix)+'\n')
            for k in range(k0+1,n+1):
                r=so.step(dt);status['steps']=k;P[k]=so.energy.evaluate(so.state.Y)['P'];log.write(json.dumps(r,allow_nan=False)+'\n');assert r['history_commit_max']<1e-10 and r['endpoint_velocity_constraint']<1e-9
                if k in audit:
                    st=so.state;gate=spectrum(e.tangent(st.Y,so.endpoint_geometry.Q));assert gate['passed'];gates.append(dict(time=st.time,static=gate));np.savez_compressed(dest/f'audit-{k:06d}.npz',x=st.x,Y=st.Y,v=st.v,C=st.C,time=st.time,F=gradient(e.B,st.Y),Q=so.endpoint_geometry.Q)
                if k%max(1,n//32)==0:print('recovered',p['name'],k,n,flush=True)
        status['completed']=True
    except Exception:status['error']=traceback.format_exc();print(status['error'],flush=True)
    np.savez_compressed(dest/'stress.npz',time=np.arange(status['steps']+1)*dt,P=P[:status['steps']+1]);status.update(seconds=time.monotonic()-start,static_gates=gates);write(dest/'status.json',status);return dict(path=str(dest.relative_to(ROOT)),completed=status['completed'],recovered=True)

def main():
    cases=load(OUT/'fast-cycle/cycle-protocol.json')['cases'];pending={p['name']:p for p in cases};results={};futures={}
    with ThreadPoolExecutor(max_workers=6) as pool:
        while pending or futures:
            for name,p in list(pending.items()):
                folder=OUT/'fast-cycle/cases'/name
                if not (folder/'status.json').exists():continue
                status=load(folder/'status.json')
                if status['completed']:results[name]=dict(path=str(folder.relative_to(ROOT)),completed=True,recovered=False)
                else:futures[name]=pool.submit(recover,p)
                pending.pop(name)
            for name,f in list(futures.items()):
                if f.done():results[name]=f.result();futures.pop(name)
            write(OUT/'completed-case-paths.json',dict(completed=len(results)==len(cases) and all(r['completed'] for r in results.values()),cases=results,pending=list(pending)+list(futures)))
            if pending or futures:time.sleep(10)
if __name__=='__main__':main()
