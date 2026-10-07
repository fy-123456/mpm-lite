"""Experimental material-carried quadratic patch energy.

Keep material patch labels, projector and reference weights. Their virtual
positions Y advance by a frozen Q1 interpolation N of grid velocity:
  E(v) = .5 sum w ||P (Y + dt N v)||^2.
A grid rebuild only changes N, so E(0) is unchanged. No offset, stiffness
rescaling or fitting is used. These carriers have no mass and only represent
stabilization history; particles retain the material constitutive history.
"""
import itertools
import numpy as np
import scipy.sparse as sp
import warp as wp
from engine.types import real,vec3
from .selective_patch import PatchEnhancements,SelectiveHistoryLiteSolver,patch_apply
from .enhancements import gather_values
from .stabilization_probe import node_coordinates

CORNERS=np.array(list(itertools.product((0,1),repeat=3)))


def carrier_map(Y,nodes,h):
    """Q1 map, with bounded linear extension for exterior virtual support."""
    Y=np.asarray(Y,float);nodes=np.asarray(nodes,int);lo=nodes.min(axis=0);hi=nodes.max(axis=0)
    if not np.isfinite(Y).all() or np.any(hi-lo<1):raise ValueError('finite carriers and complete cell support required')
    q=Y/h
    if np.any(q<lo-1) or np.any(q>hi+1):raise ValueError('material carriers exceed one-cell support extension')
    base=np.maximum(lo,np.minimum(np.floor(q).astype(int),hi-1));f=q-base
    lookup={tuple(n):i for i,n in enumerate(nodes)}
    ids=np.array([[lookup.get(tuple(n+c),-1) for c in CORNERS] for n in base],dtype=np.int32)
    if np.any(ids<0):raise ValueError('incomplete material carrier interpolation support')
    weights=np.prod(np.where(CORNERS[None,:,:],f[:,None,:],1-f[:,None,:]),axis=2)
    N=sp.csr_matrix((weights.ravel(),(np.repeat(np.arange(len(Y)),8),ids.ravel())),shape=(len(Y),len(nodes)))
    return ids,weights,N


@wp.kernel
def interpolate(ids:wp.array(dtype=int,ndim=2),weights:wp.array(dtype=real,ndim=2),v:wp.array(dtype=vec3),out:wp.array(dtype=vec3)):
    p=wp.tid();value=vec3(real(0))
    for k in range(8):value+=weights[p,k]*v[ids[p,k]]
    out[p]=value


@wp.kernel
def scatter(ids:wp.array(dtype=int,ndim=2),weights:wp.array(dtype=real,ndim=2),v:wp.array(dtype=vec3),out:wp.array(dtype=vec3)):
    p,k=wp.tid();wp.atomic_add(out,ids[p,k],weights[p,k]*v[p])


@wp.kernel
def advance(Y:wp.array(dtype=vec3),v:wp.array(dtype=vec3),dt:real):
    p=wp.tid();Y[p]+=dt*v[p]


class MaterialPatchEnhancements(PatchEnhancements):
    def __init__(self,s):
        super().__init__(s);self.initialized=False;self.last_origin=None;self.restart_state=None

    def prepare(self):
        if self.ready:return
        s=self.s
        if not self.initialized:
            super().prepare()
            if not self.reference_valid:return
            if self.restart_state is not None:
                z=self.restart_state
                self.ids=wp.array(z['ids'],dtype=int,device=s.device);self.P=wp.array(z['P'],dtype=real,device=s.device)
                self.weights=wp.array(z['weights'],dtype=real,device=s.device);self.origin=wp.array(z['Y'],dtype=vec3,device=s.device)
                self.reference_X=np.array(z['X'],copy=True)
            self.initialized=True
        else:
            self.values=wp.zeros_like(s.node_residual);self.extra=wp.zeros_like(s.node_residual)
        try:
            Y=self.origin.numpy();nodes=node_coordinates(s);ids,w,N=carrier_map(Y,nodes,s.dx)
            self.carrier_ids=wp.array(ids,dtype=int,device=s.device);self.carrier_weights=wp.array(w,dtype=real,device=s.device)
            self.N=N;self.carrier_velocity=wp.zeros(len(Y),dtype=vec3,device=s.device);self.carrier_force=wp.zeros_like(self.carrier_velocity)
            self.last_origin=Y.copy()
            self.reconstruction_stats=dict(patch_count=self.ids.shape[0],patch_nodes=self.ids.shape[1],patch_degree=2,
                material_carriers=len(Y),carrier_affine_error=float(np.max(abs(N@(nodes*s.dx)-Y))),
                carrier_partition_error=float(np.max(abs(N@np.ones(len(nodes))-1))),
                carrier_negative_weight_min=float(min(0.,w.min())),reference_history='material_carried')
            self.reference_valid=True
        except (ValueError,np.linalg.LinAlgError) as error:
            self.reference_valid=False;self.reconstruction_stats=dict(patch_failure=str(error))
        self.ready=True

    def correction(self,trial=True):
        self.prepare();s=self.s
        if not self.reference_valid:raise ValueError(self.reconstruction_stats)
        self.extra.zero_();self.carrier_force.zero_();self.energy.zero_();self.hg_energy.zero_()
        wp.launch(gather_values,dim=int(s.n_active_nodes.numpy()[0]),inputs=[s.ndof2bijk,s.grid_v_it,self.values],device=s.device)
        if not trial:self.values.zero_()
        wp.launch(interpolate,dim=len(self.origin),inputs=[self.carrier_ids,self.carrier_weights,self.values,self.carrier_velocity],device=s.device)
        wp.launch(patch_apply,dim=self.ids.shape,inputs=[self.ids,self.P,self.weights,self.origin,self.carrier_velocity,self.carrier_force,self.hg_energy,s.dt,0],device=s.device)
        wp.launch(scatter,dim=self.carrier_ids.shape,inputs=[self.carrier_ids,self.carrier_weights,self.carrier_force,self.extra],device=s.device)

    def tangent(self,p,out,pd=False):
        self.carrier_force.zero_()
        wp.launch(interpolate,dim=len(self.origin),inputs=[self.carrier_ids,self.carrier_weights,p,self.carrier_velocity],device=self.s.device)
        wp.launch(patch_apply,dim=self.ids.shape,inputs=[self.ids,self.P,self.weights,self.origin,self.carrier_velocity,self.carrier_force,self.hg_energy,self.s.dt,1],device=self.s.device)
        wp.launch(scatter,dim=self.carrier_ids.shape,inputs=[self.carrier_ids,self.carrier_weights,self.carrier_force,out],device=self.s.device)

    def commit(self):
        s=self.s
        wp.launch(gather_values,dim=int(s.n_active_nodes.numpy()[0]),inputs=[s.ndof2bijk,s.grid_v_new,self.values],device=s.device)
        wp.launch(interpolate,dim=len(self.origin),inputs=[self.carrier_ids,self.carrier_weights,self.values,self.carrier_velocity],device=s.device)
        wp.launch(advance,dim=len(self.origin),inputs=[self.origin,self.carrier_velocity,s.dt],device=s.device)

    def state(self):
        return dict(Y=self.origin.numpy().copy(),X=self.reference_X.copy(),P=self.P.numpy().copy(),ids=self.ids.numpy().copy(),weights=self.weights.numpy().copy())


class MaterialPatchLiteSolver(SelectiveHistoryLiteSolver):
    def __init__(self,*args,**kwargs):
        if kwargs.pop('stabilization','material_patch')!='material_patch':raise ValueError('material_patch required')
        super().__init__(*args,stabilization='selective_patch',**kwargs)
        self.stabilization='material_patch';self.enhancements=MaterialPatchEnhancements(self)
