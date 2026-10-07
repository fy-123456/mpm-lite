"""Independent CPU RT0 injection and streamed current-geometry integration."""
import time
import numpy as np
from engine.aniso_phase1.tensor_metrics import sampling,quadrature_axis
from engine.aniso_phase1.tensor_reference import apply_axis

def prolongation(coarse,fine):
    E=np.zeros((fine.nflux,coarse.nflux))
    for i,(x,axis,area) in enumerate(zip(fine.centres,fine.axes,fine.areas)):
        cell=int(coarse.locate(x[None,:])[0]);b=coarse.cell_bounds[cell];lo,hi=b[axis];faces=coarse.faces[cell]
        E[i,faces[2*axis]]=area*(hi-x[axis])/coarse.V0[cell]
        E[i,faces[2*axis+1]]=area*(x[axis]-lo)/coarse.V0[cell]
    return E

def integrate(reduction,q,top,mobility,order,deadline):
    """No point-by-coefficient table: one tensor slab plus one nodal dual."""
    s=reduction.parent;nodal=s.nodes(reduction.expand(q)).reshape(*s.shape,3);H=np.zeros((top.nflux,top.nflux));V=np.zeros(top.cells);G=np.zeros((top.cells,reduction.P.shape[1],3));minJ=float('inf');count=0;axes=np.repeat(np.arange(3),2);invk=np.linalg.inv(mobility);started=time.perf_counter()
    for c,b in enumerate(top.cell_bounds):
        quad=[quadrature_axis(np.unique(np.r_[b[k],s.edges[k][(s.edges[k]>b[k,0])&(s.edges[k]<b[k,1])]]),order) for k in range(3)]
        dual=np.zeros_like(nodal);local=np.zeros((6,6));ends=b[axes,np.tile([1,0],3)]
        for start in range(0,len(quad[0][0]),order):
            if time.perf_counter()>deadline:raise TimeoutError('S3 static wall-clock budget')
            xs=(quad[0][0][start:start+order],quad[1][0],quad[2][0]);ws=(quad[0][1][start:start+order],quad[1][1],quad[2][1]);B=[(sampling(e,s.p,x),sampling(e,s.p,x,True)) for e,x in zip(s.edges,xs)];gr=[]
            for j in range(3):
                v=nodal
                for k in range(3):v=apply_axis(B[k][k==j],v,k)
                gr.append(v)
            F=np.eye(3)+np.stack(gr,axis=-1);shape=F.shape[:-2];f=F.reshape(-1,3,3);J=np.linalg.det(f);minJ=min(minJ,float(J.min()))
            if not np.isfinite(f).all() or minJ<=.1:raise ValueError('invalid true frozen deformation')
            w=(ws[0][:,None,None]*ws[1][None,:,None]*ws[2][None,None,:]).ravel();cof=(J[:,None,None]*np.swapaxes(np.linalg.inv(f),1,2)).reshape(*shape,3,3);V[c]+=float(w@J)
            for j in range(3):
                v=cof[...,j]*w.reshape(*shape,1)
                for k in (2,1,0):v=apply_axis(B[k][k==j].T,v,k)
                dual+=v
            X=np.stack(np.meshgrid(*xs,indexing='ij'),axis=-1).reshape(-1,3);phi=(X[:,axes]-ends)/top.V0[c]*top.signs;invK=(np.swapaxes(f,1,2)@(invk@f))/J[:,None,None]
            for a in range(6):
                for bb in range(a,6):
                    value=float(np.dot(w*phi[:,a]*phi[:,bb],invK[:,axes[a],axes[bb]]));local[a,bb]+=value
                    if a!=bb:local[bb,a]+=value
            count+=len(w)
        G[c]=reduction.P.T@s.adjoint(dual.reshape(-1,3));faces=top.faces[c];H[np.ix_(faces,faces)]+=local
        if c==0:print('FROZEN_FIRST_CELL',order,top.shape,'seconds',time.perf_counter()-started,flush=True)
    return dict(H=H,volume=V,gradient=G,min_detF=np.array(minJ),points=np.array(count),seconds=np.array(time.perf_counter()-started))
