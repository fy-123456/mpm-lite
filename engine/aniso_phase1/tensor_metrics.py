"""Exact common-cell regional metrics using tensor sum factorization.

This evaluates the same quadrature as pointwise gradient sampling, without
expanding each point into (p+1)^3 shape-function entries. No stress smoothing.
"""
import numpy as np
import scipy.sparse as sp
from .tensor_reference import apply_axis
from benchmarks.aniso_local_q3 import basis


def sampling(edges,p,x,derivative=False):
    e=np.asarray(edges);x=np.asarray(x);cell=np.clip(np.searchsorted(e,x,side='right')-1,0,len(e)-2)
    h=e[cell+1]-e[cell];N,D=basis((x-e[cell])/h,p);v=D/h[:,None] if derivative else N
    ids=p*cell[:,None]+np.arange(p+1)
    return sp.csr_matrix((v.ravel(),(np.repeat(np.arange(len(x)),p+1),ids.ravel())),shape=(len(x),p*(len(e)-1)+1))


def quadrature_axis(edges,order):
    q,w=np.polynomial.legendre.leggauss(order);e=np.asarray(edges);h=np.diff(e)
    return (e[:-1,None]+h[:,None]*(q+1)/2).ravel(),(h[:,None]*w/2).ravel()


def evaluate_gradient(field,points):
    edges,p,u=field;U=np.asarray(u).reshape(*(p*(len(e)-1)+1 for e in edges),3)
    B=[(sampling(e,p,x),sampling(e,p,x,True)) for e,x in zip(edges,points)]
    out=[]
    for j in range(3):
        a=U
        for k in range(3):a=apply_axis(B[k][k==j],a,k)
        out.append(a)
    return np.stack(out,axis=-1)


def compare_fields(a,b,H,fiber,order=None,indicators=False,slab_cells=2):
    ea,pa,_=a;eb,pb,_=b;order=order or max(pa,pb)+1
    common=[np.union1d(np.union1d(x,y),[.3125,.375,.625,.6875] if k==0 else []) for k,(x,y) in enumerate(zip(ea,eb))]
    qs=[quadrature_axis(e,order) for e in common];totals={k:np.zeros(7) for k in ('global','grip','interior','deep_interior')};hist=[np.zeros(len(e)-1) for e in eb]
    for start in range(0,len(qs[0][0]),slab_cells*order):
        sl=slice(start,start+slab_cells*order);points=[qs[0][0][sl],qs[1][0],qs[2][0]];weights=[qs[0][1][sl],qs[1][1],qs[2][1]]
        La=evaluate_gradient(a,points);Lb=evaluate_gradient(b,points);Pa=(La.reshape(-1,9)@H.T).reshape(La.shape);Pb=(Lb.reshape(-1,9)@H.T).reshape(Lb.shape)
        aa=np.einsum('i,...ij,j->...',fiber,La,fiber);ab=np.einsum('i,...ij,j->...',fiber,Lb,fiber)
        val=np.stack((np.sum((Pa-Pb)**2,axis=(-2,-1)),np.sum(Pb*Pb,axis=(-2,-1)),(aa-ab)**2,ab*ab,np.sum((La-Lb)**2,axis=(-2,-1)),np.sum(Lb*Lb,axis=(-2,-1)),np.ones(aa.shape)),axis=-1)
        V=weights[0][:,None,None]*weights[1][None,:,None]*weights[2][None,None,:];val*=V[...,None];x=points[0];grip=(x<=.3125)|(x>=.6875)
        masks={'global':np.ones(len(x),bool),'grip':grip,'interior':~grip,'deep_interior':(x>.375)&(x<.625)}
        reduced=val.sum(axis=(1,2))
        for key,mask in masks.items():totals[key]+=reduced[mask].sum(0)
        if indicators:
            for k in range(3):
                vals=val[...,0].sum(axis=tuple(j for j in range(3) if j!=k));idx=np.clip(np.searchsorted(eb[k],points[k],side='right')-1,0,len(eb[k])-2);hist[k]+=np.bincount(idx,weights=vals,minlength=len(hist[k]))
    out={k:dict(stress_relative=float(np.sqrt(v[0]/max(v[1],1e-300))),stress_absolute_rms_Pa=float(np.sqrt(v[0]/v[6])),fiber_strain_relative=float(np.sqrt(v[2]/max(v[3],1e-300))),fiber_strain_absolute_rms=float(np.sqrt(v[2]/v[6])),gradient_relative=float(np.sqrt(v[4]/max(v[5],1e-300))),volume=float(v[6]),stress_error_squared=float(v[0])) for k,v in totals.items()}
    return dict(regions=out,quadrature_order=order,stress_passed=all(out[k]['stress_relative']<.02 for k in ('global','grip','interior')),fiber_strain_passed=all(out[k]['fiber_strain_relative']<.02 for k in ('global','grip','interior')),axis_indicators=[v.tolist() for v in hist] if indicators else None,no_smoothing=True)
