"""Short moving release: separate time errors for each spatial discretization."""
import json,time,shutil,traceback
import numpy as np
from benchmarks.aniso_v19_runs import OUT,load,write,sha
from benchmarks.aniso_v17_modes import controlled_case,analytic_field
from benchmarks.aniso_apic_frequency import Oracle
from engine.aniso_phase1.compatible_carrier import CompatibleReconstruction,make_case
from engine.aniso_phase1.compatible_avf import CompatibleAVF
from engine.aniso_phase1.endpoint_boundary import EndpointAVF
from benchmarks.aniso_carrier_joint import spectrum
from pathlib import Path

def main():
    dest=OUT/'short-moving';dest.mkdir(exist_ok=False);protocol=load(OUT/'cycle-protocol.json');scratch=Path(protocol['scratch'])/'short-moving';scratch.mkdir()
    s,e,m,h,meta=controlled_case();rec=CompatibleReconstruction(s.Y,h,16);records=[]
    write(dest/'protocol.json',dict(duration=.00625,dt=[.0000625,.00003125,.000015625],amplitude=1e-5,
        initial='same analytic carrier displacement, zero velocity/C, homogeneous hold; native material coordinates and integration weights per case',
        scope='Time convergence within each discretization; changing the position/gradient basis also changes its kinetic functional and grip representation, so not a one-factor transient attribution.',stress='every step on all native weighted quadrature points',threshold=.02))
    for name in ('sampled','gauss3','compatible16'):
        for level,dt in enumerate((.0000625,.00003125,.000015625)):
            case=controlled_case() if name=='sampled' else make_case(None if name=='gauss3' else rec,3)
            s,e,m,h,meta=case;s.v[:]=0;s.C[:]=0;s.time=1.1;u,_=analytic_field(s.Y,1e-5)
            if name=='compatible16':T=e.position_basis
            else:o=Oracle(s.x,m,h);T=o.S@o.H
            s.Y+=u;s.x+=T@u;cls=CompatibleAVF if name=='compatible16' else EndpointAVF;so=cls(s,e,m,h,mode='hold');n=round(.00625/dt)
            sub=scratch/f'{name}-L{level}';sub.mkdir();P=np.empty((n+1,len(m),3,3));P[0]=e.evaluate(s.Y)['P'];rows=[];t=time.monotonic();status=dict(completed=False,steps=0,requested_steps=n)
            try:
                for k in range(1,n+1):
                    r=so.step(dt);P[k]=so.energy.evaluate(so.state.Y)['P'];rows.append(r);status['steps']=k
                    assert abs(r['budget_defect_J'])<1e-13 and r['history_commit_max']<1e-10
                gate=spectrum(e.tangent(so.state.Y,so.endpoint_geometry.Q));assert gate['passed'];status['static_gate']=gate;status['completed']=True
            except Exception:status['error']=traceback.format_exc()
            (sub/'steps.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));np.savez_compressed(sub/'stress.npz',time=np.arange(status['steps']+1)*dt,P=P[:status['steps']+1],V=m)
            st=so.state;np.savez_compressed(sub/'terminal.npz',x=st.x,Y=st.Y,v=st.v,C=st.C,time=st.time)
            status.update(name=name,level=level,dt=dt,particles=len(m),seconds=time.monotonic()-t);write(sub/'status.json',status);copy=dest/sub.name;shutil.copytree(sub,copy)
            for f in sub.iterdir():assert sha(f)==sha(copy/f.name)
            records.append(status);print(status,flush=True)
    write(OUT/'short-moving-summary.json',dict(completed=all(r['completed'] for r in records),records=records))
if __name__=='__main__':main()
