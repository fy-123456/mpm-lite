"""Independent massless Hessians at saved deformed v14 states."""
import hashlib
import numpy as np
import scipy.sparse as sp
from benchmarks.aniso_material_history import OUT,ROOT,load,write
from benchmarks.aniso_material_check import material_N
from benchmarks.aniso_dynamic_check import pk1,CORNERS,interpolation
from benchmarks.aniso_residual_gate import rank_gate

def check(path,cfg):
    with np.load(path) as f:z={k:f[k].copy() for k in f.files}
    h=1/(cfg['grid']-1);nodes=z['grid_nodes'];centers=z['coords'];n=len(nodes);lookup={tuple(p):i for i,p in enumerate(nodes)}
    ids=np.array([[lookup[tuple(c+o)] for o in CORNERS] for c in centers]);rows=np.repeat(np.arange(len(centers)),8)
    D=[sp.csr_matrix((np.tile((2*CORNERS[:,k]-1)/(4*h),len(centers)),(rows,ids.ravel())),shape=(len(centers),n)) for k in range(3)]
    S=interpolation(z['particle_x_before'],centers,h);G=[S@d for d in D];F=z['particle_F_after'];F0=z['particle_F_before'];A=z['particle_A0'];V=z['particle_volume'];B=[sum(G[j].multiply(F0[:,j,k,None]) for j in range(3)).tocsr() for k in range(3)]
    def assemble(eps):
        H=np.empty((len(F),9,9))
        for k in range(9):
            d=np.zeros((3,3));d.flat[k]=eps;H[:,:,k]=((pk1(F+d,A,cfg['kf'])-pk1(F-d,A,cfg['kf']))/(2*eps)).reshape(-1,9)
        symmetry=float(np.max(abs(H-H.swapaxes(1,2))));H=(H+H.swapaxes(1,2))/2
        blocks=[[sum(B[b].T@B[d].multiply((V*H[:,3*a+b,3*c+d])[:,None]) for b in range(3) for d in range(3)) for c in range(3)] for a in range(3)]
        return sp.bmat(blocks,format='csr'),symmetry
    Km,asym=assemble(1e-6);Km2,_=assemble(5e-7);fd_difference=float(abs(Km-Km2).max())/max(float(abs(Km).max()),1e-30)
    pi=z['patch_ids'];P=z['patch_P'];w=z['patch_weight'];m=pi.shape[1];rr=np.repeat(pi,m,axis=1).ravel();cc=np.tile(pi,(1,m)).ravel();vals=(w[:,None,None]*(P.swapaxes(1,2)@P)).ravel();Sm=sp.csr_matrix((vals,(rr,cc)),shape=(len(z['patch_origin']),)*2)
    if 'marker_after' in z:
        material_N.dx=h;N=sp.csr_matrix(material_N(z));Sm=N.T@Sm@N
    native={tuple(p):i for i,p in enumerate(z['native_nodes'])};perm=[native[tuple(p)] for p in nodes];Sm=Sm[perm][:,perm];K=Km+sp.block_diag([Sm]*3,format='csr')
    fixed=(nodes[:,0]*h<=.25)|(nodes[:,0]*h>=.75);g={'free':np.flatnonzero(np.tile(~fixed,3))};gate=rank_gate(g,K,True)
    return dict(case=path.parent.name,snapshot=path.name,time=int(path.stem.split('-')[-1])*cfg['dt'],gate=gate,fd_relative_change=fd_difference,material_tangent_asymmetry=asym,scope='original nonlinear material Hessian at accepted trial, frozen old-F pullback and carrier map; no inertia or tangent clamp')

def main():
    p=load(OUT/'protocol.json');records=[]
    for mode in ('baseline','material'):
        for level in ('coarse','fourth'):
            name=mode+'-'+level;cfg=p['configs'][name]
            for t in (.5,1.1,1.6):
                r=check(OUT/'cases'/name/f'audit-{round(t/cfg["dt"]):05d}.npz',cfg);records.append(r);print(name,t,r['gate']['min_eigenvalue'],r['gate']['passed'],flush=True)
    write(OUT/'deformed-massless.json',dict(completed=True,records=records,all_rank_gates_passed=all(r['gate']['passed'] for r in records),finite_difference_refinement_passed=all(r['fd_relative_change']<1e-6 for r in records),source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest()))

if __name__=='__main__':main()
