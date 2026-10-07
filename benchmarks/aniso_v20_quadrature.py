"""Moving-state cut integration; global matrices and reaction-important modes."""
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v20_common import OUT,snapshot,SnapshotField,gauss_sites,write,load
from engine.aniso_phase1.swept_quadrature import sweep
from engine.aniso_phase1.separate_kinetic import KineticGeometry,grid_position_map
from engine.aniso_phase1.compatible_carrier import lite_gradient
from engine.aniso_phase1.carrier_joint import gradient
from engine.aniso_phase1.carrier_driven import kinetic_metric
from engine.aniso_phase1.local_reference import LO,HI

def integrate(s,e,m,h,meta,X,V,modes):
    g=KineticGeometry(s,e,m,h);field=SnapshotField(s,meta['particle_reference']);M=np.zeros((e.n,e.n));energy=np.zeros(len(modes));volume=0.
    for a in range(0,len(X),512):
        x=X[a:a+512];v=V[a:a+512];phys=field.at(x)['x'];B=lite_gradient(x,meta['carrier_reference'],h);inv=np.linalg.inv(gradient(B,s.Y));L=[sum(b*inv[:,k,j,None] for k,b in enumerate(B)) for j in range(3)];T=grid_position_map(phys,g.nodes,h)@g.E;J=np.vstack([T]+L);q=kinetic_metric(phys,v,h)
        M+=J.T@(q[:,None]*J);vel=np.einsum('pn,tna->tpa',J,modes,optimize=True);energy+=np.einsum('tpa,tpa,p->t',vel,vel,q);volume+=v.sum()
    return M,energy

def main():
    dest=OUT/'quadrature';dest.mkdir(exist_ok=False);records=[]
    for t in (.85,1.1,1.4,1.6):
        s,e,m,h,meta=snapshot(t);field=SnapshotField(s,meta['particle_reference']);saved=load(OUT/'modes'/f'{t:.2f}-sampled.json');ids=[r['mode'] for r in saved['reaction_modes'][:6]]
        with np.load(OUT/'modes'/f'{t:.2f}-sampled.npz') as z:modes=z['full'][:,:,ids].transpose(2,0,1)
        modes/=np.linalg.norm(modes,axis=(1,2))[:,None,None]
        edges=[]
        for j,(lo,hi) in enumerate(zip(LO,HI)):
            centers=(np.arange(-1,int(np.ceil(hi/h))+2)+.5)*h;edges.append(np.unique(np.r_[lo,hi,field.axes[j],centers[(centers>lo)&(centers<hi)]]))
        matrices=[];checks=[];naive=integrate(s,e,m,h,meta,*gauss_sites(h,3),modes)
        for order,refine in [(2,0),(3,0),(4,0)]:
            X,V,info=sweep(edges,field.maps['x'],h,order,refine);M,k=integrate(s,e,m,h,meta,X,V,modes);matrices.append((M,k));np.savez_compressed(dest/f'{t:.2f}-cut{order}.npz',M=M,mode_inertia=k)
            if len(matrices)>1:
                p,q=matrices[-2];info.update(matrix_relative=float(la.norm(M-p)/la.norm(M)),mode_inertia_max_relative=float(np.max(abs(k-q)/k)))
            checks.append(info);print(t,info,flush=True)
        if checks[-1]['matrix_relative']>1e-5 or checks[-1]['mode_inertia_max_relative']>.02:
            X,V,info=sweep(edges,field.maps['x'],h,4,1);M,k=integrate(s,e,m,h,meta,X,V,modes);p,q=matrices[-1];info.update(matrix_relative=float(la.norm(M-p)/la.norm(M)),mode_inertia_max_relative=float(np.max(abs(k-q)/k)));matrices.append((M,k));checks.append(info);np.savez_compressed(dest/f'{t:.2f}-cut4-refined.npz',M=M,mode_inertia=k);print(t,info,flush=True)
        M,k=matrices[-1];r=dict(time=t,mode_ids=ids,checks=checks,naive_gauss3_matrix_relative=float(la.norm(naive[0]-M)/la.norm(M)),naive_gauss3_mode_inertia_relative=(abs(naive[1]-k)/k).tolist(),reference_passed=checks[-1]['matrix_relative']<1e-5 and checks[-1]['mode_inertia_max_relative']<.02)
        records.append(r);write(dest/f'{t:.2f}.json',r)
    write(OUT/'moving-quadrature.json',dict(completed=True,records=records,all_reference_checks_passed=all(r['reference_passed'] for r in records),scope='Same piecewise trilinear x/v/C reconstruction of each original snapshot; derivative history uses unchanged material carrier gradient. No time integration or history reset.'))
if __name__=='__main__':main()
