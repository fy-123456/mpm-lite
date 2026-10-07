"""Device-resident sparse and axis-factored maps; no dense nodal basis."""
import time
import numpy as np
import scipy.sparse as sp
import warp as wp
from ...tensor_metrics import sampling
from .contracts import digest, array_digest

@wp.kernel
def csr_axis_kernel(ptr:wp.array(dtype=wp.int32), col:wp.array(dtype=wp.int32), val:wp.array(dtype=wp.float64),
                    x:wp.array(dtype=wp.float64), y:wp.array(dtype=wp.float64), rows:int, cols:int, inner:int):
    i=wp.tid()
    row=(i//inner)%rows
    offset=(i//(inner*rows))*cols*inner+i%inner
    a=wp.float64(0)
    for k in range(ptr[row],ptr[row+1]):
        a+=val[k]*x[offset+col[k]*inner]
    y[i]=a

@wp.kernel
def add_kernel(a:wp.array(dtype=wp.float64),b:wp.array(dtype=wp.float64),out:wp.array(dtype=wp.float64)):
    i=wp.tid();out[i]=a[i]+b[i]

@wp.kernel
def expand_kernel(q:wp.array(dtype=wp.float64), lift:wp.array(dtype=wp.float64), ids:wp.array(dtype=wp.int32),
                  full:wp.array(dtype=wp.float64), direction:int):
    i=wp.tid();r=i//3;c=i%3
    if ids[r]>=0:full[i]=q[3*ids[r]+c]
    elif direction==0:full[i]=lift[i]
    else:full[i]=wp.float64(0)

@wp.kernel
def join_kernel(a:wp.array(dtype=wp.float64), b:wp.array(dtype=wp.float64), out:wp.array(dtype=wp.float64), n:int):
    i=wp.tid()
    if i<n:out[i]=a[i]
    else:out[i]=b[i-n]

@wp.kernel
def gradient_kernel(g0:wp.array(dtype=wp.float64),g1:wp.array(dtype=wp.float64),g2:wp.array(dtype=wp.float64),
                    F:wp.array(dtype=wp.mat33d), identity:int):
    p=wp.tid();m=wp.mat33d()
    for i in range(3):
        m[i,0]=g0[3*p+i];m[i,1]=g1[3*p+i];m[i,2]=g2[3*p+i]
    if identity!=0:m+=wp.identity(3,dtype=wp.float64)
    F[p]=m

@wp.kernel
def column_kernel(P:wp.array(dtype=wp.mat33d),w:wp.array(dtype=wp.float64),g:wp.array(dtype=wp.float64),j:int):
    i=wp.tid();g[i]=w[i//3]*P[i//3][i%3,j]

class CSR:
    def __init__(self,matrix,device):
        m=sp.csr_matrix(matrix);m.sort_indices()
        self.shape=m.shape;self.device=device
        self.ptr=wp.array(m.indptr,dtype=wp.int32,device=device)
        self.col=wp.array(m.indices,dtype=wp.int32,device=device)
        self.val=wp.array(m.data,dtype=wp.float64,device=device)
        self.bytes=sum(a.capacity for a in (self.ptr,self.col,self.val))
    def apply(self,x,shape,axis=0):
        if shape[axis]!=self.shape[1]:raise ValueError('axis dimension mismatch')
        outshape=list(shape);outshape[axis]=self.shape[0]
        y=wp.empty(int(np.prod(outshape)),dtype=wp.float64,device=self.device)
        wp.launch(csr_axis_kernel,dim=y.size,inputs=[self.ptr,self.col,self.val,x,y,*self.shape,int(np.prod(shape[axis+1:]))],device=self.device)
        return y,tuple(outshape)

class GPUSpace:
    def __init__(self,space,device='cuda:0'):
        t=time.perf_counter();self.space=space;self.device=wp.get_device(device)
        self.pairs=[]
        def pair(m):
            a=(CSR(m,self.device),CSR(sp.csr_matrix(m).T,self.device));self.pairs.extend(a);return a
        self.old=pair(space.oldA);self.raw=pair(space.raw);self.transform=pair(space.transform)
        self.prolong=[pair(p) for p in space.prolong]
        ids=np.full(space.ndof,-1,dtype=np.int32);ids[space.free_scalar_ids]=np.arange(len(space.free_scalar_ids))
        self.ids=wp.array(ids,dtype=wp.int32,device=self.device)
        self.lift=self.upload(space.lift);self.layouts={}
        wp.synchronize_device(self.device)
        self.build_seconds=time.perf_counter()-t
        self.static_bytes=sum(p.bytes for p in self.pairs)+self.ids.capacity+self.lift.capacity
        self.cache_key=digest(dict(space=space.signature,lift=array_digest(space.lift),device=str(self.device),context=int(self.device.context or 0)))
    def upload(self,a):return wp.array(np.ascontiguousarray(a,dtype=np.float64).ravel(),dtype=wp.float64,device=self.device)
    def expand(self,q,*,direction=False):
        self.space._check(q,self.space.q_shape,'free direction' if direction else 'free displacement')
        out=wp.empty(self.space.ndof*3,dtype=wp.float64,device=self.device)
        wp.launch(expand_kernel,dim=out.size,inputs=[self.upload(q),self.lift,self.ids,out,int(direction)],device=self.device)
        return out
    def nodes(self,full):
        s=self.space
        a,_=self.old[0].apply(full[:s.n*3],(s.n,3))
        shape=(*s.oldshape,3)
        for k,p in enumerate(self.prolong):a,shape=p[0].apply(a,shape,k)
        b,_=self.transform[0].apply(full[s.n*3:],(s.ndof-s.n,3))
        b,_=self.raw[0].apply(b,(s.transform.shape[0],3))
        out=wp.empty_like(a);wp.launch(add_kernel,dim=a.size,inputs=[a,b,out],device=self.device)
        return out
    def adjoint(self,nodal):
        s=self.space;a=nodal;shape=(*s.shape,3)
        for k in (2,1,0):a,shape=self.prolong[k][1].apply(a,shape,k)
        a,_=self.old[1].apply(a,(int(np.prod(s.oldshape)),3))
        b,_=self.raw[1].apply(nodal,(int(np.prod(s.shape)),3))
        b,_=self.transform[1].apply(b,(s.transform.shape[0],3))
        out=wp.empty(s.ndof*3,dtype=wp.float64,device=self.device)
        wp.launch(join_kernel,dim=out.size,inputs=[a,b,out,s.n*3],device=self.device)
        return out
    def layout(self,points):
        points=self.space._points(points);key=tuple(array_digest(x) for x in points)
        if key not in self.layouts:
            pairs=[]
            for e,x in zip(self.space.edges,points):
                axes=[]
                for deriv in (False,True):
                    m=sampling(e,self.space.p,x,deriv)
                    axes.append((CSR(m,self.device),CSR(m.T,self.device)))
                pairs.append(axes)
            self.layouts[key]=(pairs,tuple(map(len,points)))
        return self.layouts[key]
    def sample(self,nodal,layout,derivative=None,transpose=False):
        pairs,pshape=layout;out=nodal;shape=(*(pshape if transpose else self.space.shape),3)
        for k in ((2,1,0) if transpose else (0,1,2)):
            out,shape=pairs[k][int(k==derivative)][int(transpose)].apply(out,shape,k)
        return out
    def gradient(self,nodal,layout,*,identity=False):
        g=[self.sample(nodal,layout,j) for j in range(3)]
        F=wp.empty(int(np.prod(layout[1])),dtype=wp.mat33d,device=self.device)
        wp.launch(gradient_kernel,dim=F.size,inputs=[*g,F,int(identity)],device=self.device)
        return F
    def gradient_adjoint(self,P,layout,weights):
        nodal=wp.zeros(int(np.prod(self.space.shape))*3,dtype=wp.float64,device=self.device)
        for j in range(3):
            g=wp.empty(P.size*3,dtype=wp.float64,device=self.device)
            wp.launch(column_kernel,dim=g.size,inputs=[P,weights,g,j],device=self.device)
            term=self.sample(g,layout,j,transpose=True)
            wp.launch(add_kernel,dim=nodal.size,inputs=[nodal,term,nodal],device=self.device)
        return self.adjoint(nodal)
    def evaluate(self,q,points,direction=False):
        layout=self.layout(points);nodal=self.nodes(self.expand(q,direction=direction))
        u=self.sample(nodal,layout).numpy().reshape(*layout[1],3)
        F=self.gradient(nodal,layout,identity=not direction).numpy().reshape(*layout[1],3,3)
        if not direction:u+=np.stack(np.meshgrid(*points,indexing='ij'),axis=-1)
        return u,F
