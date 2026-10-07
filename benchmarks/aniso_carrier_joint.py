"""v16 support-lift and same-input joint-velocity experiments (CPU float64)."""
import argparse,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.carrier_joint import CarrierEnergy,State,Geometry,CarrierJointSolver,gradient,current_gradient,MODES
from engine.aniso_phase1.material_patch import carrier_map
from engine.aniso_phase1.selective_patch import scalar_matrix
from engine.aniso_phase1.types import AnisotropicMaterialParams
from engine.aniso_phase1.unresolved_velocity import maps as apic_maps
from benchmarks.aniso_compatible_diagnosis import BASE,maps,write
from benchmarks.aniso_apic_frequency import Oracle
ROOT=BASE.parents[2]
OUT=BASE/'v16'


def load_case(time=1.6,level='fourth'):
    cfg=json.loads((BASE/'v14/protocol.json').read_text())['configs']['material-'+level]
    path=BASE/'v14/cases'/('material-'+level)/f'audit-{round(time/cfg["dt"]):05d}.npz'
    with np.load(path) as f:z={k:f[k].copy() for k in f.files}
    h=1/(cfg['grid']-1);Y=z['marker_after'];F=z['particle_F_after']
    _,G0=maps(z['particle_reference_x'],np.rint(z['patch_X']/h).astype(int),h)
    R=np.linalg.solve(gradient(G0,Y),F)
    energy=CarrierEnergy(G0,R,z['particle_volume'],z['particle_A0'],z['patch_ids'],z['patch_P'],z['patch_weight'])
    state=State(z['particle_x_after'],Y,z['particle_velocity_after'],z['particle_C_after'],time)
    assert np.max(abs(energy.evaluate(Y)['F']-F))<1e-12
    return state,energy,z['particle_mass'],h,dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_time=time)


def cube(kf=200.,angle=45.,velocity=None):
    h=1/16;x=np.array(list(itertools.product(.34375+np.arange(4)*h/2,repeat=3)))
    V=np.full(len(x),h**3/8);m=V.copy();o=Oracle(x,m,h);X=o.nodes*h
    _,ids,P=scalar_matrix(X,o.c,o.S.T@V,h)
    weights=10*(o.S.T@V)/(h*h*ids.shape[1]);G=current_gradient(x,o.nodes,h)
    a=np.array([np.cos(np.radians(angle)),np.sin(np.radians(angle)),0.]);A=np.broadcast_to(np.outer(a,a),(len(x),3,3)).copy()
    energy=CarrierEnergy(G,np.broadcast_to(np.eye(3),(len(x),3,3)),V,A,ids,P,weights,AnisotropicMaterialParams(10.,20.,kf))
    v=np.zeros_like(x) if velocity is None else np.broadcast_to(velocity,x.shape).copy()
    return State(x.copy(),X.copy(),v,np.zeros((len(x),3,3))),energy,m,h


def spectrum(K,rigid=0):
    ev=la.eigvalsh(K);tol=1e-9*max(float(np.max(np.sum(abs(K),axis=1))),1.)
    zeros=int(np.sum(ev<=tol))
    return dict(min_eigenvalue=float(ev[0]),zero_or_negative_modes=zeros,physical_rigid_modes=rigid,
                extra_modes=zeros-rigid,passed=zeros==rigid,mass_included=False,diagonal_shift_added=False,threshold=tol)


def support_gates(out):
    out.mkdir(parents=True,exist_ok=False);records=[]
    for label,kf,angle in [('ISO',0,0),('F0',200,0),('F45',200,45),('F90',200,90)]:
        state,e,m,h=cube(kf,angle);initial=e.evaluate(state.Y);K=e.tangent(state.Y)
        for cells in (0.,.008,.49,.99,1.01,1.44):
            shift=h*cells*np.array([1.,.25,-.2]);s=state.clone();s.x+=shift;s.Y+=shift
            g=Geometry(s,e,m,h,False);new=e.evaluate(s.Y)
            NN=la.block_diag(g.N,g.N,g.N);QQ=la.block_diag(g.E,g.E,g.E)
            raw=NN.T@K@NN;lift=QQ.T@raw@QQ
            record=dict(material=label,cells=cells,**g.info,raw_grid=spectrum(raw,6),carrier=spectrum(lift,6),
                tangent_preservation_relative=float(la.norm(lift-K)/la.norm(K)),
                energy_continuity_J=abs(new['U']-initial['U']),F_continuity_max=float(np.max(abs(new['F']-initial['F']))))
            records.append(record)
    assert all(r['carrier']['passed'] and r['tangent_preservation_relative']<1e-10 and r['F_continuity_max']<1e-12 for r in records)
    controls=[]
    for t in (.85,1.1,1.6):
        s,e,m,h,source=load_case(t);g=Geometry(s,e,m,h,True)
        gate=spectrum(e.tangent(s.Y,g.Q));controls.append(dict(time=t,source=source,gate=gate,**g.info))
    assert all(r['gate']['passed'] for r in controls)
    write(out/'summary.json',dict(passed=True,translation_support_gates=records,same_snapshot_gates=controls,
        scope='Material-DOF restriction preserving all quadratic grid fields; no new elastic energy; not arbitrary-grid-space certification.'))
    print('support gates',len(records),'same-state gates',len(controls),flush=True)


def smoke():
    for moving in (False,True):
        for mode in MODES:
            s,e,m,h,_=load_case();solver=CarrierJointSolver(s,e,m,h,mode,moving);start=time.monotonic()
            r=solver.step(.001);print(moving,mode,'seconds',time.monotonic()-start,r,flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('smoke','gates'));p.add_argument('--output',type=Path,default=OUT/'support-gates');a=p.parse_args()
    if a.action=='smoke':smoke()
    else:support_gates(a.output)

if __name__=='__main__':main()
