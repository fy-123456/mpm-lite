"""Identify material-invisible coordinates and verify retained static stiffness."""
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v20_common import *
from engine.aniso_phase1.integrated_avf import material_null_condensation,IntegratedAVF
from engine.aniso_phase1.carrier_joint import gradient,State
from benchmarks.aniso_v20_runs import setup

def main():
    s,e,m,h,meta=controlled_case();H,R,Z,C=material_null_condensation(e,s.Y,h);records=[]
    for t in (.85,1.1,1.4,1.6):
        s,e,m,h,meta=snapshot(t);out=e.evaluate(s.Y);Yr=R@s.Y;after=e.evaluate(Yr);r=load(OUT/'modes'/f'{t:.2f}-sampled.json')
        with np.load(OUT/'modes'/f'{t:.2f}-sampled.npz') as z:
            rows=[]
            for mode in r['reaction_modes'][:10]:
                u=z['full'][:,:,mode['mode']];fraction=float(la.norm(Z.T@u)**2/la.norm(u)**2);rows.append(dict(mode=mode['mode'],omega_rad_s=mode['omega_rad_s'],null_displacement_fraction=fraction,stabilization_fraction=mode['stabilization_fraction']))
        records.append(dict(time=t,old_constraint_norm=float(la.norm(C@s.Y)),old_potential_J=out['U'],hypothetical_instant_projection_energy_change_J=after['U']-out['U'],material_F_change=float(np.max(abs(out['F']-after['F']))),top_modes=rows))
    candidates=[];protocol=load(OUT/'fast-cycle/cycle-protocol.json');scratch=__import__('pathlib').Path(load(OUT/'fast-cycle/protocol.json')['scratch'])
    for t in (0.,.05,.5,.6,.85,1.1,1.4,1.6):
        s,e,m,h,meta=setup('gauss3-condensed')
        if t:
            path=scratch/'gauss3-condensed-L3'/f'audit-{round(t/.0000625):06d}.npz'
            completed=OUT/'completed-case-paths.json'
            if completed.exists():
                entry=load(completed)['cases'].get('gauss3-condensed-L3')
                if entry and entry['completed']:path=ROOT/entry['path']/path.name
            if not path.exists():continue
            with np.load(path) as z:s=State(z['x'],z['Y'],z['v'],z['C'],float(z['time']))
        so=IntegratedAVF(s,e,m,h,condense=True);g=so.geometry;K=e.tangent(s.Y,g.Q);L=la.cholesky(K,lower=True);W=la.solve_triangular(L.T,np.eye(len(K)),lower=False);n=g.Q.shape[1];J=np.sqrt(g.metric)[:,None]*(g.J@g.Q);A=np.einsum('pi,aij->apj',J,W.reshape(3,n,-1)).reshape(-1,len(K));sv=la.svdvals(A);finite=sv[sv>1e-12*sv[0]];w=1/finite
        candidates.append(dict(time=t,finite_modes=len(w),zero_inertia_modes=len(sv)-len(w),max_omega_rad_s=float(w.max()),minimum_period_s=float(2*np.pi/w.max()),static_min=float(la.eigvalsh(K)[0]),constraint_norm=float(la.norm(C@s.Y))))
    write(OUT/'condensation-diagnosis.json',dict(completed=True,null_scalar_dofs=Z.shape[1],material_null_error=float(la.norm(np.vstack(e.B)@Z)),old_snapshots=records,candidate_modes=candidates,old_history_reset_performed=False,exact_original_static_elimination=True,exact_dynamic_equivalence_claimed=False))
if __name__=='__main__':main()
