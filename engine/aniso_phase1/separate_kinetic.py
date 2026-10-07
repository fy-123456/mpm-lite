"""v20 geometry with independent kinetic quadrature and unchanged potential."""
import numpy as np
import scipy.linalg as la
from .carrier_joint import gradient
from .carrier_driven import kinetic_metric,CORNERS


def grid_position_map(x,nodes,h):
    lo=nodes.min(0);shape=nodes.max(0)-lo+1;q=x/h-.5;base=np.floor(q).astype(int);f=q-base;T=np.zeros((len(x),len(nodes)));rows=np.arange(len(x))
    for a in CORNERS:
        w=np.prod(np.where(a,f,1-f),axis=1)/8
        for b in CORNERS:
            ijk=base+a+b-lo;keep=w!=0
            if np.any(ijk[keep]<0) or np.any(ijk[keep]>=shape):raise ValueError('quadrature outside frozen complete grid support')
            ids=np.ravel_multi_index(ijk[keep].T,tuple(shape));T[rows[keep],ids]+=w[keep]
    return T

class KineticGeometry:
    def __init__(self,s,e,m,h,clamped=True):
        # Complete tensor carrier support; do not silently expand/regularize it.
        n=e.n;ref=getattr(e,'carrier_reference',None)
        if ref is None:
            from .carrier_driven import position_map
            from .material_patch import carrier_map
            # Quadrature support found without the old runtime particle cap.
            q=s.x/h-.5;centers=np.floor(q).astype(int)[:,None,:]+CORNERS
            nodes=np.unique((centers[:,:,None,:]+CORNERS).reshape(-1,3),axis=0)
        else:nodes=np.rint(ref/h).astype(int)
        if len(nodes)!=n or n>512 or len(s.x)>200000:raise ValueError('bounded square carrier support required')
        lo=nodes.min(0);hi=nodes.max(0);shape=hi-lo+1;loc=s.Y/h;idx=np.maximum(lo,np.minimum(np.floor(loc).astype(int),hi-1));f=loc-idx
        ids=np.ravel_multi_index((idx[:,None,:]+CORNERS-lo).reshape(-1,3).T,tuple(shape)).reshape(n,8);N=np.zeros((n,n));N[np.arange(n)[:,None],ids]=np.prod(np.where(CORNERS,f[:,None,:],1-f[:,None,:]),axis=2)
        lu,piv=la.lu_factor(N);E=la.lu_solve((lu,piv),np.eye(n));T=grid_position_map(s.x,nodes,h)@E
        B=getattr(e,'kinetic_B',e.B);inv=np.linalg.inv(gradient(B,s.Y));L=[sum(b*inv[:,k,j,None] for k,b in enumerate(B)) for j in range(3)]
        fixed=((nodes[:,0]*h<=.25)|(nodes[:,0]*h>=.75)) if clamped else np.zeros(n,bool)
        self.nodes=nodes;self.N=N;self.E=E;self.T=T;self.Q=N[:,~fixed];self.J=np.vstack([T]+L);self.metric=kinetic_metric(s.x,m,h);self.fast_square=True
        err=max(float(np.max(abs(T@s.Y-s.x))),float(np.max(abs(gradient(L,s.Y)-np.eye(3)))))
        if err>1e-9:raise ValueError('kinetic affine consistency')
        self.info=dict(carriers=n,grid_nodes=n,free_scalar_dofs=self.Q.shape[1],affine_error=err,right_inverse_error=float(np.max(abs(N@E-np.eye(n)))),constraint_error=float(np.max(abs(E[fixed]@self.Q))) if fixed.any() else 0.)
