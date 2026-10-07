"""Common-cell physical stress audit that cannot miss narrow h supports."""
import numpy as np
from engine.aniso_phase1.tensor_metrics import quadrature_axis,evaluate_gradient
from engine.aniso_phase1.consistent_transfer import material_response


def invariants(s):
    axes=[np.linspace(e[0],e[-1],n) for e,n in zip(s.edges,[9,3,3])]
    X=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1);A=np.array([[.01,.002,0],[0,-.003,.001],[.001,0,.002]]);b=np.array([.003,-.002,.001])
    q=np.zeros((s.ndof,3));q[:s.n]=s.carrier_X@A.T+b;u,g=s._sample(s.nodes(q),axes)
    affine=max(float(np.max(abs(u-X@A.T-b))),float(np.max(abs(g-A))))
    theta=.13;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    q[:s.n]=s.carrier_X@(R-np.eye(3)).T;_,g=s._sample(s.nodes(q),axes);F=np.eye(3)+g
    _,P=material_response(F.reshape(-1,3,3),np.broadcast_to(s.A,F.reshape(-1,3,3).shape),s.params)
    rigid=float(np.max(abs(P)))
    if affine>1e-8 or rigid>1e-8:raise ValueError('reference affine/rigid invariant failed')
    return dict(affine_max=affine,rigid_PK1_max_Pa=rigid)


def compare_nodal(a,b,A,params,*,order=7):
    edges=[np.union1d(x,y) for x,y in zip(a[0],b[0])];qs=[quadrature_axis(e,order) for e in edges]
    totals={n:np.zeros(5) for n in ('global_domain','clamps','transition','interior')};direction=np.asarray(params.fiber_direction)
    for start in range(0,len(qs[0][0]),2*order):
        sl=slice(start,start+2*order);points=[qs[0][0][sl],qs[1][0],qs[2][0]]
        w=qs[0][1][sl,None,None]*qs[1][1][None,:,None]*qs[2][1][None,None,:]
        Fa=np.eye(3)+evaluate_gradient(a,points);Fb=np.eye(3)+evaluate_gradient(b,points)
        def response(F):
            _,P=material_response(F.reshape(-1,3,3),np.broadcast_to(A,F.reshape(-1,3,3).shape),params);return P.reshape(F.shape)
        Pa,Pb=response(Fa),response(Fb);da=np.einsum('i,...ij,j->...',direction,Pa,direction);db=np.einsum('i,...ij,j->...',direction,Pb,direction)
        values=np.stack([np.sum((Pa-Pb)**2,axis=(-2,-1)),np.sum(Pb*Pb,axis=(-2,-1)),(da-db)**2,db**2,np.ones_like(db)],axis=-1)*w[...,None]
        frac=(points[0]-edges[0][0])/(edges[0][-1]-edges[0][0]);masks=dict(global_domain=np.ones(len(frac),bool),clamps=(frac<=.125)|(frac>=.875),transition=((frac>.125)&(frac<=.25))|((frac>=.75)&(frac<.875)),interior=(frac>.25)&(frac<.75))
        reduced=values.sum(axis=(1,2))
        for name,mask in masks.items():totals[name]+=reduced[mask].sum(axis=0)
    result={}
    for name,v in totals.items():
        result[name]={}
        for key,i in [('PK1',0),('fiber_PK1',2)]:
            err=float(np.sqrt(v[i]/v[4]));scale=float(np.sqrt(v[i+1]/v[4]));result[name][key]=dict(absolute=err,reference_norm=scale,relative=err/scale if scale>1e-14 else None,passed=err<=.02+.05*scale,budget=.02+.05*scale)
    return result
