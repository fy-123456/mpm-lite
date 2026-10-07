"""Full FE relaxation inside grip strips, plus an interior-space limiting check.

Eliminate local coordinates through the exact Schur complement of the original
quadratic physical material energy. Carrier patch matrix is never reduced.
"""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from benchmarks.aniso_v20_common import *
from benchmarks.aniso_boundary_reference import hessian
from benchmarks.aniso_local_q3 import gradient
from engine.aniso_phase1 import local_reference as ref

def main():
    dest=OUT/'local-relaxation';dest.mkdir(exist_ok=False);s,e,m,h,_=controlled_case();Y=s.Y;n=e.n;free=(Y[:,0]>.25)&(Y[:,0]<.75);Q=np.eye(n)[:,free];nf=Q.shape[1];lift=np.zeros_like(Y);lift[Y[:,0]>=.75,0]=.005;records={};solutions={};H=hessian('F45')
    for resolution in (16,32):
        with np.load(BASE/'v19/space'/f'reconstruction{resolution}.npz') as z:A=z['A'];edges=[z[f'axis{k}'] for k in range(3)]
        X,K=ref.assemble(edges,2,H);N=len(X);B=sp.block_diag([sp.csr_matrix(A@Q)]*3,format='csr');KB=K@B;Kyy=(B.T@KB).toarray()+la.block_diag(*([Q.T@e.Ks@Q]*3));u0=(A@lift).T.ravel();f0=K@u0;gy=B.T@f0+(Q.T@e.Ks@lift).T.ravel()
        for width in (.125,.1875,.25):
            # width=.25 gives all free material coordinates (limiting diagnosis).
            local=(X[:,0]>.25+1e-12)&(X[:,0]<.75-1e-12)&(((X[:,0]<.25+width-1e-12)|(X[:,0]>.75-width+1e-12)) if width<.25 else True)
            ids=np.flatnonzero(np.tile(local,3));Kaa=K[ids][:,ids].tocsc();fac=splu(Kaa);Kay=KB[ids].toarray();fa=f0[ids];cross=fac.solve(Kay);part=fac.solve(fa);S=Kyy-Kay.T@cross;S=(S+S.T)/2;v=la.solve(S,-gy+Kay.T@part,assume_a='pos');a=-part-cross@v;u=u0+B@v;u[ids]+=a;y=lift+Q@v.reshape(3,nf).T;uf=u.reshape(3,N).T;F=(K@u).reshape(3,N).T;fc=A.T@F+e.Ks@y;R=float(fc[Y[:,0]>=.75,0].sum());Um=.5*float(u@(K@u));Us=.5*float(np.sum(y*(e.Ks@y)));eig=la.eigvalsh(S);err=la.norm(Kaa@a+Kay@v+fa);tol=1e-9*max(abs(eig));row=dict(resolution=resolution,strip_width=width,local_vector_dofs=len(ids),reaction_N=R,material_J=Um,stabilization_J=Us,energy_J=Um+Us,static_min=float(eig[0]),static_passed=bool(eig[0]>tol),local_residual=float(err),work_identity_relative=abs(Um+Us-.5*.005*R)/(Um+Us),mass_included=False,stiffness_shift=0.)
            assert row['static_passed'] and row['work_identity_relative']<1e-8 and err<1e-9;name=f'r{resolution}-w{width}';records[name]=row;solutions[name]=(edges,uf);np.savez_compressed(dest/(name+'.npz'),u=uf,y=y);write(dest/(name+'.json'),row);print(name,row,flush=True)
    source=BASE/'v11-reference-q3/cases/local2.npz'
    with np.load(source) as z:ed=[z[f'axis{k}'] for k in range(3)];ur=z['u']
    common=[np.union1d(ed[k],solutions['r32-w0.25'][0][k]) for k in range(3)];sums={n:{r:np.zeros(2) for r in ('global','grip','interior')} for n in solutions}
    for X,V in ref.chunks(common,order=4,size=32):
        P=(gradient(X,ed,3,ur).reshape(-1,9)@H.T).reshape(-1,3,3);grip=(X[:,0]<=.3125)|(X[:,0]>=.6875)
        for name,(ee,u) in solutions.items():
            p=(ref.gradient(X,ee,2,u).reshape(-1,9)@H.T).reshape(-1,3,3);v=V[:,None]*np.column_stack((np.sum((p-P)**2,axis=(1,2)),np.sum(P*P,axis=(1,2))))
            for region,mask in [('global',np.ones(len(X),bool)),('grip',grip),('interior',~grip)]:sums[name][region]+=v[mask].sum(0)
    Rref=load(source.with_suffix('.json'))['reaction_N']
    for name,regions in sums.items():records[name].update(stress_relative={r:float(np.sqrt(a[0]/a[1])) for r,a in regions.items()},reaction_relative=abs(records[name]['reaction_N']-Rref)/abs(Rref))
    # Patch-only minimum is the irreducible extra reaction in the full interior
    # relaxed limit, independently computed without any material stiffness.
    y=lift.copy();y+=Q@la.solve(Q.T@e.Ks@Q,-Q.T@e.Ks@lift,assume_a='pos');floor=float((e.Ks@y)[Y[:,0]>=.75,0].sum())
    write(OUT/'local-relaxation.json',dict(completed=True,records=records,reference_reaction_N=Rref,reference_certified=False,patch_only_minimum_reaction_N=floor,nonlinear_dynamic_candidate=False))
if __name__=='__main__':main()
