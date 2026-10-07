"""Cross-check new static candidates against archived local Q3 F45 reference."""
import numpy as np
from benchmarks.aniso_v19_space import field_gradient
from benchmarks.aniso_v19_runs import ROOT,OUT,BASE,load,write,sha
from benchmarks.aniso_v17_modes import controlled_case
from engine.aniso_phase1.compatible_carrier import CompatibleReconstruction,make_case
from engine.aniso_phase1 import local_reference as ref
from benchmarks.aniso_local_q3 import gradient
from benchmarks.aniso_boundary_reference import hessian

def main():
    source=BASE/'v11-reference-q3/cases/local2.npz';q2src=BASE/'v11-reference/cases/F45-local2-q2.npz'
    with np.load(source) as z:edges=[z[f'axis{k}'] for k in range(3)];u=z['u']
    with np.load(q2src) as z:edges2=[z[f'axis{k}'] for k in range(3)];u2=z['u']
    s,e,m,h,meta=controlled_case();r16=CompatibleReconstruction(s.Y,h,16);r32=CompatibleReconstruction(s.Y,h,32)
    cases=dict(sampled=controlled_case(),gauss3=make_case(None,3),compatible16=make_case(r16,3),compatible32=make_case(r32,3));sol={n:np.load(OUT/'space'/f'F45-{n}.npz')['u'] for n in cases}
    common=[np.unique(np.concatenate([edges[k],edges2[k],r32.edges[k]])) for k in range(3)]
    sums={n:{r:np.zeros(2) for r in ('global','grip','interior')} for n in [*cases,'Q2-local2-reference']};H=hessian('F45')
    for X,V in ref.chunks(common,order=4,size=16):
        E=gradient(X,edges,3,u);P=(E.reshape(-1,9)@H.T).reshape(-1,3,3)
        masks=dict(global_=np.ones(len(X),bool),grip=(X[:,0]<=.3125)|(X[:,0]>=.6875),interior=(X[:,0]>.3125)&(X[:,0]<.6875));masks['global']=masks.pop('global_')
        for n in sums:
            e=gradient(X,edges2,2,u2) if n=='Q2-local2-reference' else field_gradient(X,cases[n],sol[n]);p=(e.reshape(-1,9)@H.T).reshape(-1,3,3)
            a=np.column_stack((np.sum((p-P)**2,axis=(1,2)),np.sum(P*P,axis=(1,2))))*V[:,None]
            for r,mask in masks.items():sums[n][r]+=a[mask].sum(0)
    R=load(source.with_suffix('.json'))['reaction_N'];out={}
    for n,regions in sums.items():
        row=dict(stress_relative={r:float(np.sqrt(a[0]/a[1])) for r,a in regions.items()})
        if n in cases:row['reaction_relative']=abs(load(OUT/'space/F45.json')['cases'][n]['reaction_N']-R)/abs(R)
        out[n]=row
    write(OUT/'local-q3-reference.json',dict(completed=True,reference_reaction_N=R,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [source,source.with_suffix('.json'),q2src]},comparisons=out,reference_certified=False,integration_order=4,common_interfaces_split=True))
    print(out,flush=True)
if __name__=='__main__':main()
