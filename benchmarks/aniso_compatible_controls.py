"""Same-input v14 quadratic release attribution and deformed massless gates."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from benchmarks.aniso_compatible_diagnosis import BASE, ROOT, maps, gradient, scalar_K, write
from benchmarks.aniso_dynamic_check import pk1
from benchmarks.aniso_residual_gate import rank_gate
from engine.aniso_phase1.material_patch import carrier_map


def stiffness(F,A,V,B,kf,eps=1e-6):
    H=np.empty((len(F),9,9))
    for k in range(9):
        d=np.zeros((3,3));d.flat[k]=eps
        H[:,:,k]=((pk1(F+d,A,kf)-pk1(F-d,A,kf))/(2*eps)).reshape(-1,9)
    asym=float(np.max(abs(H-H.swapaxes(1,2))));H=(H+H.swapaxes(1,2))/2
    blocks=[[sum(B[b].T@B[d].multiply((V*H[:,3*a+b,3*c+d])[:,None]) for b in range(3) for d in range(3)) for c in range(3)] for a in range(3)]
    return sp.bmat(blocks,format='csr'),asym


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    p=json.loads((BASE/'v14/protocol.json').read_text());records=[]
    for level in ('coarse','fine','finest','fourth'):
        cfg=p['configs']['material-'+level];h=1/(cfg['grid']-1)
        for time in (.5,.85,1.1,1.2,1.4,1.6):
            path=BASE/'v14/cases'/('material-'+level)/f'audit-{round(time/cfg["dt"]):05d}.npz'
            with np.load(path) as f:z={k:f[k].copy() for k in f.files}
            Y=z['marker_after'];F=z['particle_F_after'];V=z['particle_volume'];A=z['particle_A0']
            _,G0=maps(z['particle_reference_x'],np.rint(z['patch_X']/h).astype(int),h)
            _,G=maps(z['particle_x_after'],z['native_nodes'],h)
            _,_,N=carrier_map(Y,z['native_nodes'],h)
            R=np.linalg.solve(gradient(G0,Y),F)
            old=[sum(G[j].multiply(F[:,j,k,None]) for j in range(3)).tocsr() for k in range(3)]
            new=[sum((G0[j]@N).multiply(R[:,j,k,None]) for j in range(3)).tocsr() for k in range(3)]
            Ky=scalar_K(z);Ks=(N.T@Ky@N).tocsr()
            rr=np.einsum('cij,cja->cia',z['patch_P'],Y[z['patch_ids']])
            cf=np.zeros_like(Y);pf=z['patch_weight'][:,None,None]*np.einsum('cji,cja->cia',z['patch_P'],rr)
            np.add.at(cf,z['patch_ids'].ravel(),pf.reshape(-1,3));fs=N.T@cf
            nodes=z['native_nodes']*h;free=(nodes[:,0]>.25)&(nodes[:,0]<.75)
            H=Ks[free][:,free].toarray();lam,U=np.linalg.eigh(H);keep=lam>max(lam.max(),1)*1e-12
            f=U[:,keep].T@fs[free];release=.5*float(np.sum(f*f/lam[keep,None]))
            Us=float(.5*np.sum(z['patch_weight'][:,None,None]*rr*rr))
            du=np.zeros_like(fs);du[free]=-U[:,keep]@(f/lam[keep,None]);Ymin=Y+N@du
            rmin=np.einsum('cij,cja->cia',z['patch_P'],Ymin[z['patch_ids']])
            Emin=float(.5*np.sum(z['patch_weight'][:,None,None]*rmin*rmin))
            modes={}
            for name,B in [('baseline',old),('compatible',new)]:
                Km,asym=stiffness(F,A,V,B,cfg['kf']);Km2,_=stiffness(F,A,V,B,cfg['kf'],5e-7)
                K=Km+sp.block_diag([Ks]*3,format='csr')
                gate=rank_gate({'free':np.flatnonzero(np.tile(free,3))},K,True)
                fm=sum(b.T@(V[:,None]*pk1(F,A,cfg['kf'])[:,:,k]) for k,b in enumerate(B))
                modes[name]=dict(rank_gate=gate,fd_refinement_relative=float(abs(Km-Km2).max()/max(abs(Km).max(),1e-30)),material_hessian_asymmetry=asym,free_material_force_N=float(np.linalg.norm(fm[free])),free_total_force_N=float(np.linalg.norm((fm+fs)[free])))
            record=dict(level=level,time=time,modes=modes,initial_F_preservation_max=float(np.max(abs(gradient(G0,Y)@R-F))),
                stabilization_J=Us,quadratic_free_release_J=Us-Emin,quadratic_constrained_min_J=Emin,release_formula_error_J=abs(Us-Emin-release),
                input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            records.append(record)
            print(level,time,'free release',release/Us,'ranks',[v['rank_gate']['passed'] for v in modes.values()],flush=True)
    result=dict(completed=True,records=records,all_massless_passed=all(v['rank_gate']['passed'] for r in records for v in r['modes'].values()),
                all_fd_refinement_passed=all(v['fd_refinement_relative']<1e-6 for r in records for v in r['modes'].values()),
                scope='fresh tangent at identical committed x/F/Y; original material Hessian, no inertia/positive clamp/diagonal shift',
                release_scope='minimum stabilization energy along frozen free-grid displacement only; NOT a physical energy deletion or full-material equilibrium',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(out/'summary.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,default=BASE/'v15/controls');a=p.parse_args();r=run(a.output)
    raise SystemExit(0 if r['all_massless_passed'] and r['all_fd_refinement_passed'] else 2)
