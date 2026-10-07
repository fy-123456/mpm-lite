"""Matched Q2/Q3/Q4 local corrections ranked by correctable stress error."""
import argparse,copy,time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from benchmarks.aniso_v22_common import *
from engine.aniso_phase1.tensor_reference import interpolate,coordinates
from engine.aniso_phase1.high_order_space import BoxElastic,local_box,scalar_grams,reference_stress_load
from engine.aniso_phase1.stress_local_space import patches,reduced_solve


def freeze():
    names=['benchmarks/aniso_v22_space.py','engine/aniso_phase1/high_order_space.py','engine/aniso_phase1/stress_local_space.py','tests/test_aniso_v22_space.py']
    path=OUT/'space-protocol.json';assert not path.exists()
    write(path,dict(source_sha256={n:sha(ROOT/n) for n in names},degrees=[2,3,4],rounds=6,patches_per_round=8,patches=patches(),selection_reference='docs/results/lite-aniso-mainline/v21/reference-extension/level1-q4.npz',selection_reference_sha256=sha(BASE/'v21/reference-extension/level1-q4.npz'),held_out_new_reference_used=False,physical_mesh='Same v11 local2 intervals, with exact original A prolongation; p alone changes in the matched comparison.',basis_archive='Exact sparse original local functions and accumulated change-of-basis matrix, with reconstructed final basis checked.',directions=['ISO','F0','F45','F90']))


def run(p):
    protocol=load(OUT/'space-protocol.json')
    for n,d in protocol['source_sha256'].items():assert sha(ROOT/n)==d,n
    folder=OUT/f'space/q{p}';folder.mkdir(parents=True,exist_ok=False);s,e,_,_,_=controlled_case();start=time.monotonic()
    edges=read_field(BASE/'v11-reference-q3/cases/local2.npz',3)[0]
    with np.load(BASE/'v19/space/reconstruction32.npz') as z:oldedges=[z[f'axis{k}'] for k in range(3)];A=interpolate(oldedges,2,z['A'],edges,p)
    with np.load(BASE/'v20/space/F45-r32-e0.npz') as z:u=interpolate(oldedges,2,z['u'],edges,p)
    H=hessian('F45');op=BoxElastic(edges,p,H);opstress=copy.copy(op);opstress.H=(H.T@H).reshape(3,3,3,3);n=op.n
    reference=read_field(ROOT/protocol['selection_reference']);gref,refnorm=reference_stress_load(edges,p,reference,H);del reference
    print('PREPARED stress load',p,n,flush=True)
    free=(s.Y[:,0]>.25)&(s.Y[:,0]<.75);Q=np.eye(len(s.Y))[:,free];nf=Q.shape[1];lift=np.zeros_like(s.Y);lift[s.Y[:,0]>=.75,0]=.005;patch=Q.T@e.Ks@Q;patchlift=Q.T@e.Ks@lift;patchconstant=.5*np.sum(lift*(e.Ks@lift));W=np.empty((n,0));raw_all=sp.csr_matrix((n,0));transform=np.empty((0,0));ps=patches();locals_=[];records={};last=np.inf
    X=np.array(np.meshgrid(*coordinates(edges,p),indexing='ij')).reshape(3,-1).T;poly=lambda x:np.column_stack((np.ones(len(x)),x,x*x,x[:,0]*x[:,1],x[:,0]*x[:,2],x[:,1]*x[:,2]));polyerr=float(np.max(abs(A@poly(s.Y)-poly(X))));assert polyerr<1e-10;fixed=(X[:,0]<=.25)|(X[:,0]>=.75);del X
    for number,definition in enumerate(ps):
        idx,L,meta=local_box(edges,p,definition,H);S=copy.copy(L);S.H=(H.T@H).reshape(3,3,3,3);ids=np.concatenate([idx+j*n for j in range(3)]);locals_.append((idx,ids,L,S,meta))
    for step in range(protocol['rounds']):
        flat=u.T.ravel();force=op.apply(flat);rs=opstress.apply(flat)-gref;scores=[];corrections=[];checks=[]
        for number,(idx,ids,L,S,meta) in enumerate(locals_):
            w,check=L.correction(-force[ids]);scores.append(float(-2*rs[ids]@w-w@S.free_apply(w)));corrections.append(w);checks.append(check)
        chosen=np.argsort(-np.array(scores),kind='stable')[:8];new=[]
        for k in chosen:
            idx,ids,L,S,meta=locals_[k];w=corrections[k].reshape(3,-1).T;UU,ss,_=la.svd(w,full_matrices=False);rank=int(np.sum(ss>max(ss[0]*1e-10,1e-18)))
            for j in range(rank):
                col=np.zeros(n);col[idx]=UU[:,j];new.append(col)
        raw=np.column_stack(new);local=raw.copy();proj=np.zeros((W.shape[1],local.shape[1]))
        for _ in range(2):
            now=W.T@local;local-=W@now;proj+=now
        _,sing,Vt=la.svd(local,full_matrices=False);keep=sing>1e-8;rotation=Vt.T[:,keep]/sing[keep];Z=local@rotation
        oldrank=W.shape[1];newrank=Z.shape[1];nraw=raw.shape[1];next_transform=np.zeros((transform.shape[0]+nraw,oldrank+newrank));next_transform[:transform.shape[0],:oldrank]=transform;next_transform[:transform.shape[0],oldrank:]=-transform@proj@rotation;next_transform[transform.shape[0]:,oldrank:]=rotation;transform=next_transform;raw_all=sp.hstack((raw_all,sp.csr_matrix(raw)),format='csr');W=np.column_stack((W,Z));orth=float(la.norm(W.T@W-np.eye(W.shape[1])));assert orth<1e-8
        G=scalar_grams(edges,p,np.column_stack((A@Q,W,A@lift[:,0])));materials={}
        for label in protocol['directions']:
            coef,T,r,KK=reduced_solve(G,hessian(label),patch,patchlift,patchconstant,nf,list(range(W.shape[1])),W.T@W);assert r['static_passed'] and r['work_identity_relative']<1e-8 and r['free_residual']<1e-8;materials[label]=r
            if label=='F45':coef45=coef;assert r['energy_J']<=last*(1+1e-10);last=r['energy_J']
        y=lift+Q@coef45[:nf];u=A@y+W@coef45[nf:];flat=u.T.ravel();f=op.apply(flat).reshape(3,-1).T;R=float((A.T@f+e.Ks@y)[s.Y[:,0]>=.75,0].sum());U=.5*np.sum(u*f)+.5*np.sum(y*(e.Ks@y));r=materials['F45'];r.update(independent_reaction_error_N=abs(R-r['reaction_N']),independent_energy_error_J=abs(U-r['energy_J']));assert r['independent_reaction_error_N']<1e-9 and r['independent_energy_error_J']<1e-11
        err2=float(flat@opstress.apply(flat)-2*flat@gref+refnorm);assert err2>=-1e-12;grip=float(np.max(abs(W[fixed])));trace=float(np.max(abs(W[fixed]@coef45[nf:])));assert grip<1e-12 and trace<1e-12
        name=f'round{step+1}';record=dict(degree=p,round=step+1,chosen_patches=chosen.tolist(),predicted_stress_reductions=scores,local_checks=checks,scalar_local_dofs=W.shape[1],new_scalar_dofs=newrank,orthogonality_error=orth,quadratic_polynomial_error=polyerr,local_fixed_grip_value=grip,local_fixed_grip_displacement=trace,training_global_stress_relative=float(np.sqrt(max(0,err2)/refnorm)),materials=materials);records[name]=record;write(folder/(name+'.json'),record);np.savez_compressed(folder/(name+'.npz'),u=u,y=y,local_coefficients=coef45[nf:],degree=p,**{f'axis{k}':v for k,v in enumerate(edges)})
        print('ROUND',p,name,W.shape[1],'training stress',record['training_global_stress_relative'],'R',r['reaction_N'],flush=True)
    reconstructed=raw_all@transform;err=float(la.norm(reconstructed-W)/la.norm(W));assert err<1e-9
    np.savez_compressed(folder/'basis-transform.npz',transform=transform);sp.save_npz(folder/'basis-raw.npz',raw_all)
    write(folder/'summary.json',dict(completed=True,degree=p,records=records,static_checks=24,all_static_passed=True,seconds=time.monotonic()-start,scalar_component_closure=True,basis_reconstruction_relative_error=err,reference_vectors_copied=False,held_out_reference_used_in_selection=False))
if __name__=='__main__':
    arg=argparse.ArgumentParser();arg.add_argument('action',choices=['freeze','run']);arg.add_argument('--degree',type=int,choices=[2,3,4]);a=arg.parse_args();freeze() if a.action=='freeze' else run(a.degree)
