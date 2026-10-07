"""Nested local grip spaces; unchanged patch coefficient; independent local Q3."""
import numpy as np
from benchmarks.aniso_v20_common import *
from benchmarks.aniso_boundary_reference import hessian
from benchmarks.aniso_local_q3 import gradient
from engine.aniso_phase1.compatible_carrier import CompatibleReconstruction
from engine.aniso_phase1.grip_enrichment import static_solve
from engine.aniso_phase1 import local_reference as ref

def main():
    dest=OUT/'space';dest.mkdir(exist_ok=False);s,e,m,h,meta=controlled_case();records={};solutions={};recs={}
    for resolution in (16,32):
        path=BASE/'v19/space'/f'reconstruction{resolution}.npz'
        # The archived operator is read-only; exactly the previous base space.
        r=CompatibleReconstruction.__new__(CompatibleReconstruction);r.nodes=s.Y.copy();r.h=h;r.resolution=resolution
        with np.load(path) as z:r.A=z['A'];r.fe_nodes=z['nodes'];r.edges=[z[f'axis{k}'] for k in range(3)]
        recs[resolution]=r
        for label in ('ISO','F0','F45','F90'):
            previous=None
            for level in range(4):
                u,y,a,result=static_solve(r,e.Ks,hessian(label),level);name=f'{label}-r{resolution}-e{level}';assert result['static_passed'];assert result['work_identity_relative']<1e-8
                if previous is not None:assert result['reaction_N']<=previous*(1+1e-10)
                previous=result['reaction_N'];result.update(label=label,resolution=resolution);records[name]=result
                np.savez_compressed(dest/f'{name}.npz',u=u,y=y,a=a);write(dest/f'{name}.json',result)
                if label=='F45':solutions[name]=(r,u)
                print(name,result,flush=True)
    source=BASE/'v11-reference-q3/cases/local2.npz'
    with np.load(source) as z:edges=[z[f'axis{k}'] for k in range(3)];ur=z['u']
    common=[np.union1d(edges[k],recs[32].edges[k]) for k in range(3)];H=hessian('F45');sums={n:{r:np.zeros(2) for r in ('global','grip','interior')} for n in solutions}
    for X,V in ref.chunks(common,order=4,size=32):
        P=(gradient(X,edges,3,ur).reshape(-1,9)@H.T).reshape(-1,3,3);grip=(X[:,0]<=.3125)|(X[:,0]>=.6875);masks=dict(global_=np.ones(len(X),bool),grip=grip,interior=~grip);masks['global']=masks.pop('global_')
        for name,(r,u) in solutions.items():
            p=(ref.gradient(X,r.edges,2,u).reshape(-1,9)@H.T).reshape(-1,3,3);values=V[:,None]*np.column_stack((np.sum((p-P)**2,axis=(1,2)),np.sum(P*P,axis=(1,2))))
            for region,mask in masks.items():sums[name][region]+=values[mask].sum(0)
    reaction=load(source.with_suffix('.json'))['reaction_N']
    for name,regions in sums.items():records[name].update(stress_relative={r:float(np.sqrt(a[0]/a[1])) for r,a in regions.items()},reaction_relative=abs(records[name]['reaction_N']-reaction)/abs(reaction))
    write(OUT/'local-space.json',dict(completed=True,records=records,reference_reaction_N=reaction,reference_certified=False,reference_sha256=sha(source),same_original_stabilization=True,nonlinear_time_trajectory_implemented=False))
if __name__=='__main__':main()
