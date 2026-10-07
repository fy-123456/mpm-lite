"""Residual-updated local corrections; stress selection vs matched geometric control."""
import argparse,time
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v21_common import *
from benchmarks.aniso_v21_space import ranking
from engine.aniso_phase1 import local_reference as ref
from engine.aniso_phase1.tensor_reference import interpolate,solve
from engine.aniso_phase1.stress_local_space import patches,geometric_order,patch_corrections,scalar_grams,reduced_solve

def freeze():
    path=OUT/'adaptive-protocol.json';assert not path.exists();names=['benchmarks/aniso_v21_adaptive.py','benchmarks/aniso_v21_space.py','engine/aniso_phase1/stress_local_space.py'];write(path,dict(source_sha256={n:sha(ROOT/n) for n in names},reason='Initial residual correction loses effectiveness at larger budgets; update the local residual after every global variational solve.',rounds=6,patches_per_round=8,selection=['stress','geometric'],material_directions=['ISO','F0','F45','F90'],stress_selection_reference='archived local2 Q3 only',geometric_control='same correction rule and budgets; cycle through a fixed farthest-point ordering',orthogonalization='Scalar span enlarged with local correction component functions; reorthogonalization changes basis only. All spatial components share this span.',history_or_dynamic_change=False))

def run(ordering):
    protocol=load(OUT/'adaptive-protocol.json')
    for n,d in protocol['source_sha256'].items():assert sha(ROOT/n)==d,n
    folder=OUT/'adaptive'/ordering;folder.mkdir(parents=True,exist_ok=False);s,e,m,h,_=controlled_case()
    with np.load(BASE/'v19/space/reconstruction32.npz') as z:oldedges=[z[f'axis{k}'] for k in range(3)];A=z['A']
    edges=read_field(BASE/'v11-reference-q3/cases/local2.npz',3)[0];A=interpolate(oldedges,2,A,edges,2)
    with np.load(BASE/'v20/space/F45-r32-e0.npz') as z:u=interpolate(oldedges,2,z['u'],edges,2)
    nodes,K=ref.assemble(edges,2,hessian('F45'));free=(s.Y[:,0]>.25)&(s.Y[:,0]<.75);Q=np.eye(len(s.Y))[:,free];nf=Q.shape[1];lift=np.zeros_like(s.Y);lift[s.Y[:,0]>=.75,0]=.005;patch=Q.T@e.Ks@Q;patchlift=Q.T@e.Ks@lift;patchconstant=.5*np.sum(lift*(e.Ks@lift));W=np.empty((len(nodes),0));ps=patches();geo=geometric_order(ps);records={};last_energy=np.inf;start=time.monotonic()
    for step in range(protocol['rounds']):
        if ordering=='stress':selection=ranking(edges,u);chosen=selection['stress_order'][:protocol['patches_per_round']]
        else:chosen=[geo[(step*protocol['patches_per_round']+j)%len(geo)] for j in range(protocol['patches_per_round'])];selection=dict(geometric_order=geo)
        local,local_records=patch_corrections(nodes,K,u,[ps[i] for i in chosen]);raw=local.copy()
        for _ in range(2):local-=W@(W.T@local)
        Z,singular,_=la.svd(local,full_matrices=False);keep=singular>1e-8;W=np.column_stack((W,Z[:,keep]));orth=float(la.norm(W.T@W-np.eye(W.shape[1])));assert orth<1e-8
        G=scalar_grams(edges,np.column_stack((A@Q,W,A@lift[:,0])));Wgram=W.T@W;round_records={};coef45=None
        for label in protocol['material_directions']:
            coef,T,result,KK=reduced_solve(G,hessian(label),patch,patchlift,patchconstant,nf,list(range(W.shape[1])),Wgram);assert result['static_passed'] and result['work_identity_relative']<1e-8 and result['free_residual']<1e-8;round_records[label]=result
            if label=='F45':coef45=coef;assert result['energy_J']<=last_energy*(1+1e-10);last_energy=result['energy_J']
        y=lift+Q@coef45[:nf];u=A@y+W@coef45[nf:];f=(K@u.T.ravel()).reshape(3,-1).T;R=float((A.T@f+e.Ks@y)[s.Y[:,0]>=.75,0].sum());U=.5*np.sum(u*f)+.5*np.sum(y*(e.Ks@y));r=round_records['F45'];r.update(independent_reaction_error_N=abs(R-r['reaction_N']),independent_energy_error_J=abs(U-r['energy_J']));assert r['independent_reaction_error_N']<1e-9 and r['independent_energy_error_J']<1e-11
        name=f'round{step+1}';record=dict(round=step+1,ordering=ordering,chosen_patches=chosen,selection=selection,local_records=local_records,scalar_local_dofs=W.shape[1],new_scalar_dofs=int(keep.sum()),orthogonality_error=orth,materials=round_records);records[name]=record;write(folder/(name+'.json'),record);np.savez_compressed(folder/(name+'.npz'),u=u,y=y,local_coefficients=coef45[nf:],degree=2,**{f'axis{k}':v for k,v in enumerate(edges)});np.savez_compressed(folder/(name+'-basis.npz'),W=W,raw_local_corrections=raw);print(ordering,name,W.shape[1],r['reaction_N'],flush=True)
    write(folder/'summary.json',dict(completed=True,records=records,static_checks=len(records)*4,all_static_passed=True,seconds=time.monotonic()-start,scalar_component_closure=True,reference_vectors_copied=False,held_out_reference_used_in_selection=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--ordering',choices=['stress','geometric']);a=p.parse_args();freeze() if a.action=='freeze' else run(a.ordering)
