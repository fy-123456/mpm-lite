"""Preserve all outcomes, including failed ablations, of the frozen prototype."""
import argparse,traceback
import numpy as np
from benchmarks.aniso_v16_experiments import OUT,write,load,sources
from benchmarks.aniso_carrier_joint import load_case,cube,spectrum
from engine.aniso_phase1.carrier_joint import CarrierJointSolver,gradient,MODES


def controls():
    assert not (OUT/'small-step-controls.json').exists();records=[]
    for t in (1.1,1.6):
        s,e,m,h,source=load_case(t)
        for mode in MODES:
            for dt in (.001,.0005,.00025,.000125,1e-5,1e-6,1e-7):
                so=CarrierJointSolver(s,e,m,h,mode)
                r=dict(start=t,mode=mode,dt=dt,source=source,completed=False)
                try:
                    r.update(so.step(dt));r.update(completed=True,max_Y_increment=float(np.max(abs(so.state.Y-s.Y))))
                except Exception:r['error']=traceback.format_exc()
                records.append(r);print(t,mode,dt,r['completed'],r.get('delta_total_J'),flush=True)
    write(OUT/'small-step-controls.json',dict(completed=True,all_steps_succeeded=all(r['completed'] for r in records),records=records,
        scope='One-step controls. Adjoint-only and unconstrained-history projection controls are intentionally not promoted. Errors are retained, not replaced by successful steps.'))


def crossing():
    assert not (OUT/'moving-support-diagnostic.json').exists()
    velocity=np.array([.1,.025,-.02]);s,e,m,h=cube(velocity=velocity)
    so=CarrierJointSolver(s,e,m,h,'joint',True,False);records=[];ranks=[]
    for k in range(180):
        r=so.step(.005);t=so.state.time
        r.update(x_error=float(np.max(abs(so.state.x-s.x-t*velocity))),Y_error=float(np.max(abs(so.state.Y-s.Y-t*velocity))),
                 F_error=float(np.max(abs(gradient(e.B,so.state.Y)-np.eye(3)))),v_error=float(np.max(abs(so.state.v-velocity))),C_error=float(np.max(abs(so.state.C))))
        records.append(r)
        if k+1 in (1,2,10,30,60,90,120,150,180):
            ranks.append(dict(step=k+1,gate=spectrum(e.tangent(so.state.Y),6)))
            print(k+1,{n:v for n,v in r.items() if n.endswith('_error') or n in ('grid_nodes','total_J','kinetic_rank')},ranks[-1],flush=True)
    maxima={k:max(r[k] for r in records) for k in ('x_error','Y_error','F_error','v_error','C_error','momentum_change_norm')}
    write(OUT/'moving-support-diagnostic.json',dict(completed=True,passed=max(maxima.values())<1e-10 and all(r['gate']['passed'] for r in ranks),steps=180,
        max_errors=maxima,records=records,actual_massless_gates=ranks,source_sha256=sources(),
        scope='Repeat of failed long crossing with every observation retained; frozen algorithm unchanged.'))

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('controls','crossing'));a=p.parse_args()
    controls() if a.action=='controls' else crossing()
