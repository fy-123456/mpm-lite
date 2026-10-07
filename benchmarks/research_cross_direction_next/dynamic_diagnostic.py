"""Physical mass overlaps explain changed modal coordinates; no extra trajectory."""
from pathlib import Path
import argparse,numpy as np
from .provenance import *
from .spaces import load_selected
from .run import load_model
from benchmarks.research_phase_stress_next.time_study import history
from engine.aniso_phase1.research_post_release.fields import CachedProbes

def review(run):
    run=Path(run);old,_=load_selected(read(run/'baseline-space.json')['package']);new,_=load_selected(read(run/'selected-space.json')['package'])
    with np.load(REFERENCE/'Q1/R3/data.npz') as z:M=z['M'].copy()
    with np.load(LOCAL/'S1/candidates/global-snapshot6/space.npz') as z:Ta=z['T'].copy()
    with np.load(run/'S1/candidates/cross-direction-snapshot6/space.npz') as z:Tb=z['T'].copy()
    A=Ta@old.P[:,old.free];B=Tb@new.P[:,new.free];cross=np.kron(A.T@M@B,np.eye(3))
    with np.load(APP/'S1/modal-basis.npz') as z:va=z['values'].copy();Va=z['vectors'].copy()
    with np.load(run/'S1/modal-basis.npz') as z:vb=z['values'].copy();Vb=z['vectors'].copy()
    overlap=(Va.T@cross@Vb)**2;rows=[]
    for j in (4,72,523):
        k=int(np.argmax(overlap[j]));wa,wb=np.sqrt(va[j]),np.sqrt(vb[k]);rows.append(dict(old_mode=j,new_mode=k,mass_overlap_squared=float(overlap[j,k]),old_omega_rad_s=float(wa),new_omega_rad_s=float(wb),relative_frequency_difference=float(wb/wa-1),linear_phase_difference_at_1_6_s=float(1.6*(wb-wa))))
    cfg=read(run/'cases/final-full/execution-protocol.json');m,_=load_model(run,cfg);cache=CachedProbes(m);a=history(run/'cases/final-full');b=history(run/'cases/daily-q5-retry');common_J=[]
    for i in (0,40,88,150,214,252):
        x,y=cache.frame(a[i]['state']),cache.frame(b[i]['state']);jx,jy=np.linalg.det(x['F']),np.linalg.det(y['F']);common_J.append(dict(time_s=a[i]['state'].time,q7_min_J=float(jx.min()),q5_min_J=float(jy.min()),max_common_point_J_difference=float(np.max(abs(jx-jy)))))
    write(run/'S6/dynamic-diagnostic.json',dict(status='scoped_diagnostic',modes=rows,common_probe_J=common_J,interpretation='changed physical trial space shifts frequencies and modal overlaps; old velocity history is not a dynamic truth reference; local phase evidence remains BASELINE only',not_a_dynamic_accuracy_proof=True,quadrature_minimum_note='raw q5/q7 min_detF samples different material points and path probes; use common physical points for comparison',new_steps=0))
    print('MODE_CHANGES',rows,'COMMON_J',max(x['max_common_point_J_difference'] for x in common_J),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):review(a.run)
