"""Exact common-cell spatial error integration; batched independent fields."""
import argparse,time
import numpy as np
from benchmarks.aniso_v21_common import *
from engine.aniso_phase1.compatible_carrier import shape_matrices


def compare_many(fields,reference,label='F45',order=None):
    er,pr,ur=reference;order=order or pr+1;common=[e.copy() for e in er];groups={}
    for name,(e,p,u) in fields.items():
        assert p==2;key=tuple(tuple(x) for x in e);groups.setdefault(key,[]).append((name,u))
        common=[np.union1d(a,b) for a,b in zip(common,e)]
    common[0]=np.union1d(common[0],[.3125,.375,.625,.6875]);prepared=[]
    for key,rows in groups.items():prepared.append(([np.array(e) for e in key],[n for n,u in rows],np.column_stack([u for n,u in rows])))
    totals={n:{r:np.zeros(7) for r in ('global_','grip','interior','deep_interior')} for n in fields};H=hessian(label);fiber=FIBER
    for X,V in chunks(common,order=order,size=12):
        Lr=gradient(X,er,pr,ur);Pr=Lr.reshape(-1,9)@H.T;ar=np.einsum('i,pij,j->p',fiber,Lr,fiber);normP=np.sum(Pr*Pr,axis=1);normL=np.sum(Lr*Lr,axis=(1,2));masks=regions(X)
        for edges,names,U in prepared:
            _,*B=shape_matrices(X,edges);L=np.stack([b@U for b in B],axis=-1).reshape(len(X),len(names),3,3);P=L.reshape(-1,9)@H.T;P=P.reshape(len(X),len(names),9);aa=np.einsum('i,pcij,j->pc',fiber,L,fiber)
            errs=(np.sum((P-Pr[:,None])**2,axis=2),(aa-ar[:,None])**2,np.sum((L-Lr[:,None])**2,axis=(2,3)))
            for k,name in enumerate(names):
                val=V[:,None]*np.column_stack((errs[0][:,k],normP,errs[1][:,k],ar*ar,errs[2][:,k],normL,np.ones(len(X))))
                for region,m in masks.items():totals[name][region]+=val[m].sum(0)
    out={}
    for name,rows in totals.items():
        out[name]={r.rstrip('_'):dict(stress_relative=float(np.sqrt(v[0]/v[1])),stress_absolute_rms_Pa=float(np.sqrt(v[0]/v[6])),fiber_strain_relative=float(np.sqrt(v[2]/v[3])),fiber_strain_absolute_rms=float(np.sqrt(v[2]/v[6])),gradient_relative=float(np.sqrt(v[4]/v[5])),volume=float(v[6]),stress_error_squared=float(v[0])) for r,v in rows.items()}
    return out


def evaluate(stage):
    while not all((OUT/'space'/m/'summary.json').exists() for m in ('coarse','graded')):time.sleep(5)
    paths={}
    for mesh in ('coarse','graded'):
        for p in sorted((OUT/'space'/mesh).glob('F45-*.npz')):paths[mesh+'/'+p.stem]=p
    if stage=='training':reference=BASE/'v11-reference-q3/cases/local2.npz';degree=3
    elif stage=='validation':reference=OUT/'reference/local3-q4.npz';degree=None
    else:
        while not (OUT/'reference-extension-summary.json').exists() or not load(OUT/'reference-extension-summary.json')['completed']:time.sleep(10)
        reference=OUT/load(OUT/'reference-extension-summary.json')['latest_reference'];degree=None
        training=load(OUT/'space-training.json')['cases'];keys=[]
        for mesh in ('coarse','graded'):
            for ordering in ('stress','geometric'):
                names=[n for n in training if n.startswith(mesh+'/F45-'+ordering) and not n.endswith('-00')];best=min(names,key=lambda n:training[n]['regions']['global']['stress_relative']);keys.append(best)
            keys.append(mesh+'/F45-stress-00')
        paths={k:paths[k] for k in keys}
    start=time.monotonic();fields={n:read_field(p) for n,p in paths.items()};result=compare_many(fields,read_field(reference,degree));rr=load(reference.with_suffix('.json'))['reaction_N'];cases={}
    for name,regions_ in result.items():
        record=load(paths[name].with_suffix('.json'));cases[name]=dict(scalar_local_dofs=record['scalar_local_dofs'],reaction_relative=abs(record['reaction_N']-rr)/abs(rr),regions=regions_,stress_passed=all(regions_[r]['stress_relative']<.02 for r in ('global','grip','interior')),fiber_passed=all(regions_[r]['fiber_strain_relative']<.02 for r in ('global','grip','interior')),solution_sha256=sha(paths[name]));print(stage,name,cases[name],flush=True)
    write(OUT/f'space-{stage}.json',dict(completed=True,stage=stage,cases=cases,reference=str(reference.relative_to(ROOT)),reference_sha256=sha(reference),reference_continuum_certified=False,seconds=time.monotonic()-start,quadrature_order=degree+1 if degree else read_field(reference)[1]+1,grip_partition='x <= .3125 or x >= .6875; exact common-cell integration; includes rigid grip volumes'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['training','validation','final']);evaluate(p.parse_args().stage)
