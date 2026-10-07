"""Matched-budget geometric/error-selected local spaces on two physical meshes."""
import argparse,time
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v21_common import *
from engine.aniso_phase1 import local_reference as ref
from engine.aniso_phase1.tensor_reference import interpolate
from engine.aniso_phase1.stress_local_space import patches,geometric_order,patch_corrections,scalar_grams,reduced_solve

def ranking(edges,u):
    ps=patches();scores=np.zeros(len(ps));old=read_field(BASE/'v11-reference-q3/cases/local2.npz',3);common=[np.union1d(a,b) for a,b in zip(edges,old[0])];H=hessian('F45');fiber=FIBER;axial=np.zeros_like(scores)
    for X,V in chunks(common,order=4,size=16):
        L=ref.gradient(X,edges,2,u)-gradient(X,*old);dP=L.reshape(-1,9)@H.T;error=V*np.sum(dP*dP,axis=1);ae=V*np.einsum('i,pij,j->p',fiber,L,fiber)**2
        for k,p in enumerate(ps):
            mask=np.all((X>=p['lo'])&(X<=p['hi']),axis=1);scores[k]+=error[mask].sum();axial[k]+=ae[mask].sum()
    return dict(stress_order=np.argsort(-scores,kind='stable').tolist(),geometric_order=geometric_order(ps),patch_stress_error_squared=scores.tolist(),patch_fiber_error_squared=axial.tolist(),selection_reference_sha256=sha(BASE/'v11-reference-q3/cases/local2.npz'),overlap_note='Patches overlap; indicator fractions are not additive.',reference_vectors_used_as_basis=False)

def freeze():
    p=OUT/'space-protocol.json';assert not p.exists();names=['engine/aniso_phase1/stress_local_space.py','benchmarks/aniso_v21_space.py','tests/test_aniso_v21_local_space.py'];write(p,dict(source_sha256={n:sha(ROOT/n) for n in names},meshes=['coarse','graded'],patch_counts=[2,6,12,18,24,40],directions=['ISO','F0','F45','F90'],ranking='Initial full tensor stress difference to archived local Q3, fixed before candidate solves; geometric farthest-point control.',basis='Local physical residual solve, scalar component closure; no reference solution vector copied.',coarse='v19 reconstruction32 mesh',graded='archived local Q3 physical intervals, represented with Q2',patches=patches()))

def run(mesh):
    p=load(OUT/'space-protocol.json')
    for n,d in p['source_sha256'].items():assert sha(ROOT/n)==d,n
    folder=OUT/'space'/mesh;folder.mkdir(parents=True,exist_ok=True);s,e,m,h,_=controlled_case()
    with np.load(BASE/'v19/space/reconstruction32.npz') as z:oldedges=[z[f'axis{k}'] for k in range(3)];A=z['A'];nodes=z['nodes']
    with np.load(BASE/'v20/space/F45-r32-e0.npz') as z:ub=z['u']
    edges=oldedges if mesh=='coarse' else read_field(BASE/'v11-reference-q3/cases/local2.npz',3)[0]
    if mesh!='coarse':A=interpolate(oldedges,2,A,edges,2);ub=interpolate(oldedges,2,ub,edges,2)
    start=time.monotonic();nodes,K=ref.assemble(edges,2,hessian('F45'));print(mesh,'assembled',len(nodes),K.nnz,flush=True)
    if not (folder/'selection.json').exists():write(folder/'selection.json',ranking(edges,ub))
    selection=load(folder/'selection.json');free=(s.Y[:,0]>.25)&(s.Y[:,0]<.75);Q=np.eye(len(s.Y))[:,free];nf=Q.shape[1];lift=np.zeros_like(s.Y);lift[s.Y[:,0]>=.75,0]=.005
    if not (folder/'corrections.npz').exists():
        W,patch_records=patch_corrections(nodes,K,ub,p['patches'],callback=lambda r:print(mesh,'patch',r['patch'],r['scalar_rank'],r['local_vector_dofs'],flush=True));np.savez_compressed(folder/'corrections.npz',W=W);write(folder/'patches.json',patch_records)
    else:W=np.load(folder/'corrections.npz')['W'];patch_records=load(folder/'patches.json')
    if not (folder/'grams.npz').exists():
        G=scalar_grams(edges,np.column_stack((A@Q,W,A@lift[:,0])));np.savez_compressed(folder/'grams.npz',**{f'g{i}{j}':v for (i,j),v in G.items()})
    else:
        with np.load(folder/'grams.npz') as z:G={(i,j):z[f'g{i}{j}'] for i in range(3) for j in range(3)}
    Wgram=W.T@W;patch=Q.T@e.Ks@Q;patchlift=Q.T@e.Ks@lift;patchconstant=.5*np.sum(lift*(e.Ks@lift));records={};baseline=None
    for label in p['directions']:
        for order in ('stress','geometric'):
            previous=np.inf
            for count in [0]+p['patch_counts']:
                if count==0 and order=='geometric':continue
                chosen=selection[order+'_order'][:count];ids=[i for k in chosen for i in patch_records[k]['columns']];coef,T,result,KK=reduced_solve(G,hessian(label),patch,patchlift,patchconstant,nf,ids,Wgram);assert result['static_passed'] and result['work_identity_relative']<1e-8 and result['free_residual']<1e-8;assert result['energy_J']<=previous*(1+1e-9);previous=result['energy_J'];name=f'{label}-{order}-{count:02d}';result.update(mesh=mesh,direction=label,patch_count=count,selected_patches=chosen,selected_columns=ids);records[name]=result
                if label=='F45':
                    y=lift+Q@coef[:nf];u=A@y+(W[:,ids]@coef[nf:] if ids else 0);force=(K@u.T.ravel()).reshape(3,-1).T;R=float((A.T@force+e.Ks@y)[s.Y[:,0]>=.75,0].sum());energy=.5*np.sum(u*force)+.5*np.sum(y*(e.Ks@y));result.update(independent_reaction_error_N=abs(R-result['reaction_N']),independent_energy_error_J=abs(energy-result['energy_J']));assert abs(R-result['reaction_N'])<1e-9 and abs(energy-result['energy_J'])<1e-11
                    np.savez_compressed(folder/(name+'.npz'),u=u,y=y,local_coefficients=coef[nf:],degree=2,**{f'axis{k}':v for k,v in enumerate(edges)})
                write(folder/(name+'.json'),result);print(mesh,name,result['reaction_N'],'dofs',result['scalar_local_dofs'],flush=True)
    x=s.Y;poly=np.column_stack((np.ones(len(x)),x,x*x,x[:,0]*x[:,1],x[:,0]*x[:,2],x[:,1]*x[:,2]));target=np.column_stack((np.ones(len(nodes)),nodes,nodes*nodes,nodes[:,0]*nodes[:,1],nodes[:,0]*nodes[:,2],nodes[:,1]*nodes[:,2]));error=float(np.max(abs(A@poly-target)));assert error<1e-10;fixed=(nodes[:,0]<=.25)|(nodes[:,0]>=.75);assert np.max(abs(W[fixed]))==0
    write(folder/'summary.json',dict(completed=True,records=records,static_checks=len(records),all_static_passed=True,polynomial_max_error=error,local_grip_value_max=0.,seconds=time.monotonic()-start,scalar_corrections=W.shape[1],new_dynamic_claim=False));print(mesh,'DONE',time.monotonic()-start,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run']);p.add_argument('--mesh',choices=['coarse','graded']);a=p.parse_args();freeze() if a.action=='freeze' else run(a.mesh)
