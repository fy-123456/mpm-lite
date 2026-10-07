"""Quadratic-preserving nodal patch energy (experimental).

For frozen reconstructed material coordinates X, fit all degree <= 2 fields.
E = sum_c mu Vc/(2 h^2 m_c) ||P_c y_c||_F^2,
P = I - Q Q.T. Coefficient is fixed by matrix shear modulus, not reaction data.
"""
import itertools
import numpy as np
import scipy.sparse as sp


def polynomial(x):
    a, b, c = np.asarray(x).T
    return np.column_stack((np.ones(len(x)), a, b, c, a*a, b*b, c*c, a*b, a*c, b*c))


def patches(nodes, centers):
    nodes=np.asarray(nodes, int); centers=np.asarray(centers, int)
    lo, hi=nodes.min(axis=0), nodes.max(axis=0)
    if np.any(hi-lo < 3):
        raise ValueError('quadratic patch requires four active node layers in each direction')
    lookup={tuple(x):i for i,x in enumerate(nodes)}; offsets=np.array(list(itertools.product(range(4),repeat=3)))
    rows=[]
    for c in centers:
        start=np.maximum(lo,np.minimum(c-1,hi-3))
        ids=[lookup.get(tuple(x),-1) for x in start+offsets]
        if min(ids)<0: raise ValueError('quadratic patch currently requires complete rectangular local support')
        rows.append(ids)
    return np.asarray(rows)


def projectors(X, ids, h):
    x=(X[ids]-X[ids].mean(axis=1)[:,None,:])/h
    a,b,c=x.transpose(2,0,1)
    B=np.stack((np.ones_like(a),a,b,c,a*a,b*b,c*c,a*b,a*c,b*c),axis=2)
    Q,R=np.linalg.qr(B,mode='reduced')
    if np.min(abs(np.diagonal(R,axis1=1,axis2=2))) < 1e-8:
        raise ValueError('degenerate material patch')
    return np.eye(ids.shape[1])[None,:,:]-Q@Q.transpose(0,2,1)


def scalar_matrix(nodes,centers,V,h,X=None,mu=10.):
    ids=patches(np.rint(nodes/h).astype(int),centers)
    if X is None:X=nodes
    P=projectors(X,ids,h)
    # P.T P is used, rather than relying on finite precision idempotence.
    local=P.transpose(0,2,1)@P * (mu*V/(h*h*ids.shape[1]))[:,None,None]
    m=ids.shape[1]
    return sp.csr_matrix((local.ravel(),(np.repeat(ids,m,axis=1).ravel(),np.tile(ids,(1,m)).ravel())),shape=(len(nodes),)*2),ids,P

# The frozen-step energy is quadratic in nodal trial positions. Its force and
# tangent are exact; no polar derivative or separately chosen force is needed.
import warp as wp
from engine.types import real,vec3
from engine.sp_grid import B
from .enhancements import scatter_reference,normalize_reference,gather_values
from .stabilization_probe import node_coordinates
from .residual_history import ResidualHistoryLiteSolver


@wp.kernel
def patch_apply(ids:wp.array(dtype=int,ndim=2),P:wp.array(dtype=real,ndim=3),weights:wp.array(dtype=real),
                origin:wp.array(dtype=vec3),v:wp.array(dtype=vec3),out:wp.array(dtype=vec3),energy:wp.array(dtype=real),dt:real,tangent:int):
    c,j=wp.tid();r=vec3(real(0))
    for k in range(ids.shape[1]):
        value=dt*v[ids[c,k]]
        if tangent==0:value+=origin[ids[c,k]]
        r+=P[c,j,k]*value
    w=weights[c]
    for k in range(ids.shape[1]):
        wp.atomic_add(out,ids[c,k],dt*w*P[c,j,k]*r)
    if tangent==0:
        e=real(.5)*w*wp.dot(r,r)
        wp.atomic_add(energy,0,e);wp.atomic_add(energy,1,e)


class PatchEnhancements:
    def __init__(self,s):
        self.s=s;self.ready=False;self.reference_valid=True
        self.energy=wp.zeros(2,dtype=real,device=s.device);self.hg_energy=wp.zeros_like(self.energy)

    def resample(self):
        self.ready=False;self.reference_valid=True

    def prepare(self):
        if self.ready:return
        s=self.s;nn=int(s.n_active_nodes.numpy()[0]);nc=int(s.n_active_centers.numpy()[0])
        self.values=wp.zeros_like(s.node_residual);self.extra=wp.zeros_like(s.node_residual)
        u=wp.zeros_like(s.node_residual);w=wp.zeros(len(u),dtype=real,device=s.device)
        wp.launch(scatter_reference,dim=s.n_ptc,inputs=[s.ptc_x,s.ptc_reference_x,s.ptc_F,s.ptc_m,s.block2bid,s.node2dof,u,w,s.dx],device=s.device)
        wp.launch(normalize_reference,dim=nn,inputs=[u,w],device=s.device)
        nodes=node_coordinates(s);x=nodes*s.dx;X=x-u[:nn].numpy()
        cdof=s.cdof2bijk[:nc].numpy();l=cdof[:,1]
        centers=s.block_xyz_by_id.numpy()[cdof[:,0]]*B+np.column_stack((l//B**2,(l//B)%B,l%B))
        V=s.center_vol[:,:s.bcn].numpy()[0][cdof[:,0],l//B**2,l%B**2]
        try:
            ids=patches(nodes,centers);P=projectors(X,ids,s.dx)
            if np.any(w[:nn].numpy()<=0):raise ValueError('unreconstructed node')
            self.ids=wp.array(ids,dtype=int,device=s.device);self.P=wp.array(P,dtype=real,device=s.device)
            self.weights=wp.array(s.aniso_params.mu*V/(s.dx**2*ids.shape[1]),dtype=real,device=s.device)
            self.origin=wp.array(x,dtype=vec3,device=s.device)
            self.reference_X=X
            self.reconstruction_stats=dict(patch_count=nc,patch_nodes=ids.shape[1],patch_degree=2,
                patch_polynomial_residual_max=float(np.max(abs(P@polynomial(X)[ids]))))
        except (ValueError,np.linalg.LinAlgError) as error:
            self.reference_valid=False;self.reconstruction_stats=dict(patch_failure=str(error))
        self.ready=True

    def correction(self,trial=True):
        self.prepare();s=self.s
        if not self.reference_valid:raise ValueError(self.reconstruction_stats)
        self.extra.zero_();self.energy.zero_();self.hg_energy.zero_()
        wp.launch(gather_values,dim=int(s.n_active_nodes.numpy()[0]),inputs=[s.ndof2bijk,s.grid_v_it,self.values],device=s.device)
        if not trial:self.values.zero_()
        wp.launch(patch_apply,dim=self.ids.shape,inputs=[self.ids,self.P,self.weights,self.origin,self.values,self.extra,self.hg_energy,s.dt,0],device=s.device)

    def tangent(self,p,out,pd=False):
        wp.launch(patch_apply,dim=self.ids.shape,inputs=[self.ids,self.P,self.weights,self.origin,p,out,self.hg_energy,self.s.dt,1],device=self.s.device)

    def memory_bytes(self):
        from warp._src.types import type_size_in_bytes
        return sum(a.size*type_size_in_bytes(a.dtype) for a in vars(self).values() if isinstance(a,wp.array))


class SelectiveHistoryLiteSolver(ResidualHistoryLiteSolver):
    """Particle history plus objective, quadratic-preserving patch energy."""
    def __init__(self,*args,**kwargs):
        if kwargs.pop('stabilization','selective_patch')!='selective_patch':
            raise ValueError('SelectiveHistoryLiteSolver requires selective_patch')
        if kwargs.get('stabilization_strength',1.)!=1.:
            raise ValueError('selective_patch coefficient is fixed at 1; no reaction fitting')
        super().__init__(*args,stabilization='none',**kwargs)
        if self.aniso_params.mu<=0:raise ValueError('positive matrix shear modulus required')
        self.stabilization='selective_patch';self.enhancements=PatchEnhancements(self)
