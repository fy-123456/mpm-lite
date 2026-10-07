"""Ablations: physical quadrature, conforming gradient, independent reference."""
import itertools,time
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v19_runs import OUT,ROOT,load,write,sha
from benchmarks.aniso_v17_modes import controlled_case
from benchmarks.aniso_boundary_reference import hessian
from benchmarks.aniso_carrier_joint import spectrum
from benchmarks.aniso_v18_space import gauss_sites
from benchmarks.aniso_apic_frequency import Oracle
from engine.aniso_phase1.compatible_carrier import CompatibleReconstruction,make_case,lite_gradient
from engine.aniso_phase1.carrier_driven import kinetic_metric
from engine.aniso_phase1 import local_reference as ref


def matrices(s,e,m,h,label='F45'):
    H=hessian(label);B=e.B
    K=np.block([[sum(H[3*a+i,3*b+j]*(B[i].T@(e.V[:,None]*B[j])) for i in range(3) for j in range(3) if H[3*a+i,3*b+j]!=0)+(e.Ks if a==b else 0) for b in range(3)] for a in range(3)])
    if getattr(e,'position_basis',None) is None:o=Oracle(s.x,m,h);T=o.S@o.H
    else:T=e.position_basis
    q=kinetic_metric(s.x,m,h);J=np.vstack([T]+list(B));M=J.T@(q[:,None]*J);return (K+K.T)/2,M,J,q


def rest_solution(s,e,m,h,label='F45'):
    K,M,J,q=matrices(s,e,m,h,label);n=e.n;free=(s.Y[:,0]>.25+1e-12)&(s.Y[:,0]<.75-1e-12);f=np.flatnonzero(np.tile(free,3));u=np.zeros(3*n);right=s.Y[:,0]>=.75-1e-12;u[:n][right]=.005
    A=K[f][:,f];u[f]=la.solve(A,-(K@u)[f],assume_a='pos');force=K@u;R=float(force[:n][right].sum());gate=spectrum(A)
    sv=la.svdvals(np.sqrt(q)[:,None]*J[:,free]);null=3*int(np.sum(sv<1e-12*sv[0]));return u.reshape(3,n).T,dict(reaction_N=R,static_gate=gate,zero_inertia_modes=null,energy_J=.5*float(u@force),free_residual=float(la.norm(force[f])),inertia_trace=float(np.trace(M))), (K,M,J,q)


def field_gradient(X,case,u):
    s,e,m,h,_=case;rec=getattr(e,'reconstruction',None)
    B=lite_gradient(X,s.Y,h) if rec is None else rec.maps(X)[1:]
    return np.stack([b@u for b in B],axis=2)


def compare_reference(cases,solutions,reference,H):
    edges,ur=reference;sums={name:{r:np.zeros(2) for r in ('global','grip','interior')} for name in cases}
    for X,V in ref.chunks(edges,order=3,size=16):
        Er=ref.gradient(X,edges,2,ur);Pr=(Er.reshape(-1,9)@H.T).reshape(-1,3,3)
        masks=dict(global_=np.ones(len(X),bool),grip=(X[:,0]<=.3125)|(X[:,0]>=.6875),interior=(X[:,0]>.3125)&(X[:,0]<.6875));masks['global']=masks.pop('global_')
        for name,case in cases.items():
            E=field_gradient(X,case,solutions[name]);P=(E.reshape(-1,9)@H.T).reshape(-1,3,3)
            a=np.column_stack((np.sum((P-Pr)**2,axis=(1,2)),np.sum(Pr*Pr,axis=(1,2))))*V[:,None]
            for r,mask in masks.items():sums[name][r]+=a[mask].sum(0)
    return {n:{r:float(np.sqrt(a[0]/a[1])) for r,a in rs.items()} for n,rs in sums.items()}


def main():
    dest=OUT/'space';dest.mkdir(exist_ok=False);start=time.monotonic();s,e,m,h,meta=controlled_case();rec16=CompatibleReconstruction(s.Y,h,16);rec32=CompatibleReconstruction(s.Y,h,32)
    write(dest/'protocol.json',dict(grids=[16,32],quadrature=[3,4],reference='independent physical conforming Q2 elastic solve, no stabilization or mass',labels=['ISO','F0','F45','F90'],space_target=.02,gradient='scalar H1 projection, quadratic one-sided grip trace',inertia='same discrete APIC kinetic functional with real positive material Gauss weights',production_default_changed=False))
    np.savez_compressed(dest/'reconstruction16.npz',A=rec16.A,nodes=rec16.fe_nodes,**{f'axis{k}':v for k,v in enumerate(rec16.edges)})
    np.savez_compressed(dest/'reconstruction32.npz',A=rec32.A,nodes=rec32.fe_nodes,**{f'axis{k}':v for k,v in enumerate(rec32.edges)})
    records={};quadrature_checks=[];modes=[]
    for label in ('ISO','F0','F45','F90'):
        # Identical physical parameters, right displacement and original patch coefficient.
        base=controlled_case();bs,be,bm,bh,meta=base
        if label!='F45':
            a={'ISO':[1.,0,0],'F0':[1.,0,0],'F90':[0.,1.,0]}[label];be.A[:]=np.outer(a,a)
            if label=='ISO':
                from engine.aniso_phase1.types import AnisotropicMaterialParams
                be.params=AnisotropicMaterialParams(10.,20.,0.)
        cases=dict(sampled=(bs,be,bm,bh,meta),gauss3=make_case(None,3,label),compatible16=make_case(rec16,3,label),compatible32=make_case(rec32,3,label))
        solutions={};recs={};allmat={}
        for name,case in cases.items():
            u,r,mat=rest_solution(*case[:4],label);assert r['static_gate']['passed'];solutions[name]=u;recs[name]=r;allmat[name]=mat;np.savez_compressed(dest/f'{label}-{name}.npz',u=u);print(label,name,r,flush=True)
        references={}
        for n in (16,32):
            edges=[np.linspace(a,b,round((b-a)*n)+1) for a,b in zip(ref.LO,ref.HI)];nodes,u,r=ref.solve(edges,2,hessian(label));assert r['passed'];references[n]=(edges,2,u,r);np.savez_compressed(dest/f'{label}-reference{n}.npz',u=u,nodes=nodes)
        reference_check=ref.compare(references[16],references[32],hessian(label));errs=compare_reference(cases,solutions,(references[32][0],references[32][2]),hessian(label))
        for name,r in recs.items():r.update(stress_relative=errs[name],reaction_relative=abs(r['reaction_N']-references[32][3]['reaction_N'])/abs(references[32][3]['reaction_N']))
        records[label]=dict(cases=recs,reference=references[32][3],reference_refinement=reference_check)
        if label=='F45':
            for name,rec in [('gauss',None),('compatible',rec16)]:
                a=make_case(rec,3,label);b=make_case(rec,4,label);ka,ma,_,_=matrices(*a[:4],label);kb,mb,_,_=matrices(*b[:4],label)
                q=dict(name=name,K_relative=float(la.norm(ka-kb)/la.norm(kb)),M_relative=float(la.norm(ma-mb)/la.norm(mb)),points=[len(a[2]),len(b[2])]);quadrature_checks.append(q);assert max(q['K_relative'],q['M_relative'])<1e-10
            with np.load(ROOT/'docs/results/lite-aniso-mainline/v18/space-mode-targets.npz') as z:fields=z['fields'];labels=z['labels'].tolist()
            for name,(K,M,J,q) in allmat.items():
                e=cases[name][1]
                for tag,v in zip(labels,fields):
                    flat=v.T.ravel();mk=float(np.sum(v*(M@v)));km=float(flat@K@flat);ks=float(np.sum(v*(e.Ks@v)));modes.append(dict(case=name,target=tag,modal_mass=mk,stiffness=km,material_stiffness=km-ks,stabilization_stiffness=ks,rayleigh_rad_s=float(np.sqrt(km/mk))))
        write(dest/f'{label}.json',records[label])
    write(OUT/'space-summary.json',dict(completed=True,seconds=time.monotonic()-start,records=records,quadrature=quadrature_checks,fixed_field_modes=modes,reconstruction=[rec16.info,rec32.info],
        spatial_accuracy_passed=all(r['cases']['compatible32']['reaction_relative']<.02 and max(r['cases']['compatible32']['stress_relative'].values())<.02 for r in records.values()),
        scope='Linear physical boundary-value comparison and fixed carrier-field Rayleigh quotients; no nonlinear continuum or eigenbranch certification.'))

if __name__=='__main__':main()
