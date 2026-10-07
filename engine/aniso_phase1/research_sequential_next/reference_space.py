"""Nested local h enrichment of the same carrier/Ks nonlinear model.

The ambient conforming Q4 grid stays fixed. These spaces are bounded local
reference diagnostics, not a continuum-reference certificate.
"""
from __future__ import annotations
import copy
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from ..tensor_reference import coordinates
from ..research_d.identity import digest
from ..research_d.stage2.contracts import array_digest
from ..research_sequential.condensation import FullCoordinates,Condensation
from ..research_d.common_kinetic import PointInertia


def local_h_family(parent,level):
    if level not in (1,2):raise ValueError('two bounded levels only')
    axes=coordinates(parent.edges,parent.p)
    x,y,z=np.meshgrid(*axes,indexing='ij')
    yy=2*(y-parent.edges[1][0])/(parent.edges[1][-1]-parent.edges[1][0])-1
    zz=2*(z-parent.edges[2][0])/(parent.edges[2][-1]-parent.edges[2][0])-1
    transverse=[np.ones_like(yy),yy,zz,yy*zz,.5*(3*yy**2-1),.5*(3*zz**2-1)]
    intervals=[(.25,.5),(.5,.75)]
    if level==2:intervals.extend([(.25,.375),(.375,.5),(.5,.625),(.625,.75)])
    columns=[];labels=[]
    for lo,hi in intervals:
        if not all(np.any(np.isclose(parent.edges[0],a,atol=1e-13,rtol=0)) for a in (lo,hi)):
            raise ValueError('local support boundaries must align with conforming cell edges')
        t=(x-lo)/(hi-lo);bubble=np.where((x>lo)&(x<hi),4*t*(1-t),0.)
        for j,p in enumerate(transverse):
            value=(bubble*p).ravel();scale=np.linalg.norm(value)
            columns.append(value/scale);labels.append(dict(x_support=[lo,hi],transverse_mode=j,normalization=scale))
    return np.column_stack(columns),labels


def extend(parent,basis,*,label):
    basis=np.asarray(basis,dtype=np.float64)
    if basis.ndim!=2 or basis.shape[0]!=int(np.prod(parent.shape)) or not np.isfinite(basis).all():
        raise ValueError('finite nodal enrichment required')
    obj=copy.copy(parent);extra=basis.shape[1]
    obj.raw=sp.hstack((parent.raw,sp.csr_matrix(basis)),format='csr')
    obj.transform=la.block_diag(parent.transform,np.eye(extra))
    obj.ndof=parent.ndof+extra
    obj.reference=np.vstack((parent.reference,np.zeros((extra,3))))
    obj.free_scalar_ids=np.r_[parent.free_scalar_ids,np.arange(parent.ndof,obj.ndof)]
    obj.fixed_scalar_ids=parent.fixed_scalar_ids.copy();obj.q_shape=(len(obj.free_scalar_ids),3)
    obj.lift=np.vstack((parent.lift,np.zeros((extra,3))))
    obj.q0=np.vstack((parent.q0,np.zeros((extra,3))))
    obj.signature=digest(dict(parent=parent.signature,extension=array_digest(basis),label=label,
        physical_model='same nonlinear material, original carrier Ks, original rigid grips',ambient_degree=parent.p))
    obj.metadata=dict(parent.metadata,parent_sha256=parent.signature,derived_space_label=label,
        producer='research_sequential_next',q_shape=list(obj.q_shape))
    for value in (obj.transform,obj.reference,obj.free_scalar_ids,obj.fixed_scalar_ids,obj.lift,obj.q0):value.setflags(write=False)
    for value in (obj.raw.data,obj.raw.indices,obj.raw.indptr):value.setflags(write=False)
    return obj


def extended_operators(parent_reduction,space,*,device='cuda:0',max_seconds=600,progress=None):
    """Reuse exact old blocks; compute every new cross term, never lump mass."""
    from ..research_d.stage2.gpu_operator import GPUOperator
    start=time.perf_counter();old=parent_reduction.parent.ndof;n=space.ndof;extra=n-old
    if extra<1:raise ValueError('an actual extension is required')
    M=np.zeros((n,n));M[:old,:old]=parent_reduction.original_mass
    inertia=PointInertia(space,order=5)
    for i in range(old,n,3):
        js=np.arange(i,min(i+3,n));v=np.zeros((n,3));v[js,np.arange(len(js))]=1.
        columns=inertia.apply(v)[:,:len(js)];M[:,js]=columns;M[js,:]=columns.T
        if time.perf_counter()-start>max_seconds:raise TimeoutError('local reference mass budget exhausted')
    op=GPUOperator(FullCoordinates(space),order=7,device=device)
    K=np.zeros((3*n,3*n));K[:3*old,:3*old]=parent_reduction.original_stiffness
    lin=op.prepare(np.zeros((n,3)))
    for j in range(3*old,3*n):
        d=np.zeros((n,3));d.ravel()[j]=1.
        col=op.action(lin,d).ravel();K[:,j]=col;K[j,:]=col
        if progress is not None and (j-3*old)%12==0:progress('new rest tangent columns',j-3*old,3*extra)
        if time.perf_counter()-start>max_seconds:raise TimeoutError('local reference tangent budget exhausted')
    # Verify block reuse with an independent complete action, including cross terms.
    rng=np.random.default_rng(20260930);d=rng.normal(size=(n,3))
    direct=inertia.apply(d);mass_error=float(la.norm(M@d-direct)/max(la.norm(direct),1e-30))
    direct=op.action(lin,d).ravel();stiffness_error=float(la.norm(K@d.ravel()-direct)/max(la.norm(direct),1e-30))
    if max(mass_error,stiffness_error)>2e-5:raise ValueError('extended operator cross terms failed')
    reduction=Condensation(space,M,K)
    return reduction,dict(seconds=time.perf_counter()-start,extra_scalar=extra,mass_action_relative=mass_error,
        stiffness_action_relative=stiffness_error,original_Ks_unchanged=np.array_equal(space.Ks,parent_reduction.parent.Ks),
        all_mass_cross_terms=True,ambient_degree=space.p,reference_qualification='bounded enrichment within fixed ambient Q4',
        rank=reduction.audit)


def select_locals(parent,indices,*,label):
    """Select an explicit scalar local subspace, preserving all carriers/Ks."""
    indices=np.asarray(indices,dtype=int)
    if indices.ndim!=1 or len(set(indices.tolist()))!=len(indices) or np.any(indices<0) or np.any(indices>=parent.ndof-parent.n):
        raise ValueError('distinct valid local function indices required')
    obj=copy.copy(parent);obj.transform=np.array(parent.transform[:,indices],copy=True)
    obj.ndof=obj.n+len(indices);obj.reference=np.vstack((parent.carrier_X,np.zeros((len(indices),3))))
    obj.free_scalar_ids=np.r_[parent.free_scalar_ids[parent.free_scalar_ids<parent.n],np.arange(obj.n,obj.ndof)]
    obj.fixed_scalar_ids=parent.fixed_scalar_ids.copy();obj.q_shape=(len(obj.free_scalar_ids),3)
    obj.lift=np.vstack((parent.lift[:parent.n],np.zeros((len(indices),3))))
    obj.q0=np.vstack((parent.q0[:parent.nfree_carrier],parent.q0[parent.nfree_carrier:][indices]))
    obj.signature=digest(dict(parent=parent.signature,selected=indices.tolist(),label=label,budget=len(indices)))
    obj.metadata=dict(parent.metadata,parent_sha256=parent.signature,derived_space_label=label,q_shape=list(obj.q_shape))
    for value in (obj.transform,obj.reference,obj.free_scalar_ids,obj.fixed_scalar_ids,obj.lift,obj.q0):value.setflags(write=False)
    full_ids=np.r_[np.arange(obj.n),obj.n+indices]
    return obj,full_ids
