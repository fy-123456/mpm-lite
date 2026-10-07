"""Rank correctable stress error, with original variational equilibrium intact."""
import time
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import splu
from benchmarks.aniso_v21_common import *
from engine.aniso_phase1 import local_reference as ref
from engine.aniso_phase1.tensor_reference import interpolate
from engine.aniso_phase1.stress_local_space import patches,patch_mask,scalar_grams,reduced_solve
from engine.aniso_phase1.stress_gain import reference_stress_load,predicted_reduction

def main():
    folder=OUT/'adaptive/gain';folder.mkdir(parents=True,exist_ok=False);names=['benchmarks/aniso_v21_gain.py','engine/aniso_phase1/stress_gain.py','engine/aniso_phase1/stress_local_space.py'];write(folder/'protocol.json',dict(source_sha256={n:sha(ROOT/n) for n in names},rounds=6,patches_per_round=8,reason='Raw stress-error ranking stagnated; select directions by predicted correctable stress-error reduction.',score='-2 r_sigma^T w - w^T K_sigma w, where w solves the original local material residual. H.T H is ONLY an error norm, never equilibrium stiffness.',selection_reference_sha256=sha(BASE/'v11-reference-q3/cases/local2.npz'),held_out_reference_used=False))
    s,e,_,_,_=controlled_case();reference=read_field(BASE/'v11-reference-q3/cases/local2.npz',3);edges=reference[0]
    with np.load(BASE/'v19/space/reconstruction32.npz') as z:oldedges=[z[f'axis{k}'] for k in range(3)];A=interpolate(oldedges,2,z['A'],edges,2)
    with np.load(BASE/'v20/space/F45-r32-e0.npz') as z:u=interpolate(oldedges,2,z['u'],edges,2)
    H=hessian('F45');nodes,K=ref.assemble(edges,2,H);_,Ks=ref.assemble(edges,2,H.T@H);gref,refnorm=reference_stress_load(edges,reference,H);free=(s.Y[:,0]>.25)&(s.Y[:,0]<.75);Q=np.eye(len(s.Y))[:,free];nf=Q.shape[1];lift=np.zeros_like(s.Y);lift[s.Y[:,0]>=.75,0]=.005;patch=Q.T@e.Ks@Q;patchlift=Q.T@e.Ks@lift;patchconstant=.5*np.sum(lift*(e.Ks@lift));W=np.empty((len(nodes),0));ps=patches();locals_=[];records={};last=np.inf;start=time.monotonic()
    for number,p in enumerate(ps):
        idx=np.flatnonzero(patch_mask(nodes,p));ids=np.concatenate([idx+j*len(nodes) for j in range(3)]);Apatch=K[ids][:,ids].tocsc();locals_.append((idx,ids,splu(Apatch),Ks[ids][:,ids].tocsr()));print('factor',number,len(ids),flush=True)
    for step in range(6):
        flat=u.T.ravel();force=K@flat;rs=Ks@flat-gref;scores=[];corrections=[]
        for idx,ids,fac,S in locals_:
            w=fac.solve(-force[ids]);scores.append(predicted_reduction(ids,w,rs,S));corrections.append(w)
        chosen=np.argsort(-np.array(scores),kind='stable')[:8];new=[]
        for k in chosen:
            idx,ids,fac,S=locals_[k];w=corrections[k].reshape(3,-1).T;UU,ss,_=la.svd(w,full_matrices=False);rank=int(np.sum(ss>max(ss[0]*1e-10,1e-18)))
            for j in range(rank):
                col=np.zeros(len(nodes));col[idx]=UU[:,j];new.append(col)
        local=np.column_stack(new);raw=local.copy()
        for _ in range(2):local-=W@(W.T@local)
        ZZ,ss,_=la.svd(local,full_matrices=False);keep=ss>1e-8;W=np.column_stack((W,ZZ[:,keep]));orth=float(la.norm(W.T@W-np.eye(W.shape[1])));assert orth<1e-8;G=scalar_grams(edges,np.column_stack((A@Q,W,A@lift[:,0])));gram=W.T@W;materials={}
        for label in ('ISO','F0','F45','F90'):
            coef,T,r,KK=reduced_solve(G,hessian(label),patch,patchlift,patchconstant,nf,list(range(W.shape[1])),gram);assert r['static_passed'] and r['work_identity_relative']<1e-8 and r['free_residual']<1e-8;materials[label]=r
            if label=='F45':coef45=coef;assert r['energy_J']<=last*(1+1e-10);last=r['energy_J']
        y=lift+Q@coef45[:nf];u=A@y+W@coef45[nf:];f=(K@u.T.ravel()).reshape(3,-1).T;R=float((A.T@f+e.Ks@y)[s.Y[:,0]>=.75,0].sum());U=.5*np.sum(u*f)+.5*np.sum(y*(e.Ks@y));r=materials['F45'];r.update(independent_reaction_error_N=abs(R-r['reaction_N']),independent_energy_error_J=abs(U-r['energy_J']));assert r['independent_reaction_error_N']<1e-9 and r['independent_energy_error_J']<1e-11
        name=f'round{step+1}';record=dict(round=step+1,ordering='gain',chosen_patches=chosen.tolist(),predicted_stress_reductions=scores,scalar_local_dofs=W.shape[1],new_scalar_dofs=int(keep.sum()),orthogonality_error=orth,materials=materials);records[name]=record;write(folder/(name+'.json'),record);np.savez_compressed(folder/(name+'.npz'),u=u,y=y,local_coefficients=coef45[nf:],degree=2,**{f'axis{k}':v for k,v in enumerate(edges)});np.savez_compressed(folder/(name+'-basis.npz'),W=W,raw_local_corrections=raw);print('gain',name,W.shape[1],r['reaction_N'],'max predicted stress decrease',max(scores),flush=True)
    write(folder/'summary.json',dict(completed=True,records=records,static_checks=24,all_static_passed=True,seconds=time.monotonic()-start,scalar_component_closure=True,reference_vectors_copied=False,held_out_reference_used_in_selection=False))
if __name__=='__main__':main()
