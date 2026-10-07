"""Actual ambient h/p enrichment preserving the original carrier potential."""
from __future__ import annotations
import copy
import time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from engine.aniso_phase1.tensor_reference import coordinates
from engine.aniso_phase1.tensor_metrics import sampling
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_d.stage2.contracts import array_digest
from engine.aniso_phase1.research_d.common_kinetic import PointInertia
from engine.aniso_phase1.research_sequential.condensation import Condensation,FullCoordinates
from engine.aniso_phase1.research_sequential_next.reference_space import extend
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedGPUOperator


def ambient_space(parent, level, *, mode='h'):
    if mode not in ('h','p') or level not in (1,2):raise ValueError('bounded two-level ambient family required')
    e=parent.edges[0]
    ids=np.flatnonzero((e[:-1]>=parent.boundary['left_x']-1e-12)&(e[1:]<=parent.boundary['right_x']+1e-12))
    selected=[int(ids[0]),int(ids[-1])]
    new=copy.copy(parent)
    if mode=='h':
        x=np.unique(np.r_[e,*[np.linspace(e[i],e[i+1],2**level+1) for i in selected]])
        new.edges=(x,parent.edges[1],parent.edges[2]);new.p=parent.p
    else:
        new.edges=parent.edges;new.p=parent.p+level
    axes=coordinates(new.edges,new.p)
    prolong=[sampling(old,parent.p,x) for old,x in zip(parent.edges,axes)]
    tensor=sp.kron(sp.kron(prolong[0],prolong[1],format='csr'),prolong[2],format='csr')
    new.shape=tuple(len(x) for x in axes)
    new.raw=(tensor@parent.raw).tocsr()
    new.prolong=tuple(sampling(e,2,x) for e,x in zip(parent.oldedges,axes))
    new.signature=digest(dict(parent=parent.signature,mode=mode,level=level,edges=[x.tolist() for x in new.edges],p=new.p,
                              meaning='exact embedding of all old functions; same carrier Ks; local physical enrichment'))
    x,y,z=np.meshgrid(*axes,indexing='ij')
    yy=2*(y-new.edges[1][0])/(new.edges[1][-1]-new.edges[1][0])-1
    zz=2*(z-new.edges[2][0])/(new.edges[2][-1]-new.edges[2][0])-1
    transverse=[np.ones_like(yy),yy,zz,yy*zz,.5*(3*yy**2-1),.5*(3*zz**2-1)]
    columns=[];labels=[]
    for cell in selected:
        lo,hi=float(e[cell]),float(e[cell+1])
        if mode=='p':
            left,right=parent.boundary['left_x'],parent.boundary['right_x'];middle=(left+right)/2
            lo,hi=(left,middle) if cell==selected[0] else (middle,right)
        if mode=='h':
            mid=(lo+hi)/2
            intervals=[(lo,mid)] if level==1 else [(lo,mid),(lo,(lo+mid)/2),(mid,(mid+hi)/2)]
            for a,b in intervals:
                t=(x-a)/(b-a);bubble=np.where((x>a)&(x<b),4*t*(1-t),0.)
                for j,p in enumerate(transverse):
                    v=(bubble*p).ravel();scale=la.norm(v)
                    columns.append(v/scale);labels.append(dict(cell=cell,support=[a,b],transverse=j,normalization=float(scale)))
        else:
            t=(x-lo)/(hi-lo);inside=(x>lo)&(x<hi)
            for degree in range(parent.p+1,new.p+1):
                polynomial=np.polynomial.legendre.Legendre.basis(degree-2)(2*t-1)
                bubble=np.where(inside,t*(1-t)*polynomial,0.)
                for j,p in enumerate(transverse):
                    v=(bubble*p).ravel();scale=la.norm(v)
                    columns.append(v/scale);labels.append(dict(cell=cell,support=[lo,hi],degree=degree,transverse=j,normalization=float(scale)))
    basis=np.column_stack(columns)
    result=extend(new,basis,label=f'actual-ambient-{mode}{level}')
    # Sparse interpolation can leave unsorted indices. Canonicalize privately
    # before freezing; the legacy GPU uploader sorts its CSR input in place.
    raw=result.raw.copy();raw.sum_duplicates();raw.sort_indices()
    for array in (raw.data,raw.indices,raw.indptr):array.setflags(write=False)
    result.raw=raw
    result.metadata=dict(result.metadata,ambient_refinement=mode,ambient_degree=new.p,selected_cells=selected)
    # Extend's historical metadata names Q4; identity additionally binds the
    # actual ambient parent above and the complete explicit basis values.
    result.signature=digest(dict(embedded_parent=new.signature,basis=array_digest(basis),p=new.p,mode=mode,level=level))
    return result,dict(mode=mode,level=level,degree=new.p,selected_cells=selected,
                       original_shape=list(parent.shape),ambient_shape=list(new.shape),
                       original_cells=[len(e)-1 for e in parent.edges],ambient_cells=[len(e)-1 for e in new.edges],
                       extra_scalar=basis.shape[1],labels=labels,
                       all_old_functions_exactly_embedded=True,original_carrier_Ks=True,artificial_new_penalty=False)


def embedding_audit(old,new):
    rng=np.random.default_rng(20261001);q=rng.normal(size=(old.ndof,3))*.0001
    full=np.vstack((q,np.zeros((new.ndof-old.ndof,3))))
    axes=[np.linspace(e[0]+1e-8,e[-1]-1e-8,n) for e,n in zip(old.edges,[29,5,5])]
    a,ga=old._sample(old.nodes(q),axes);b,gb=new._sample(new.nodes(full),axes)
    carrier=old.reference[:old.n]+q[:old.n]
    invariant=dict(displacement_max=float(np.max(abs(a-b))),gradient_max=float(np.max(abs(ga-gb))),
        stabilization_energy_delta=float(.5*np.sum(carrier*((new.Ks-old.Ks)@carrier))),
        stabilization_gradient_max=float(np.max(abs((new.Ks-old.Ks)@carrier))),
        original_Ks_equal=bool(np.array_equal(old.Ks,new.Ks)))
    if invariant['gradient_max']>1e-8 or not invariant['original_Ks_equal']:raise ValueError('ambient embedding changed old physics')
    return invariant


def operators(parent_reduction,space,*,max_seconds=600,progress=None):
    start=time.perf_counter();n=space.ndof;old=parent_reduction.parent.ndof
    mass_order=space.p+1
    M=np.zeros((n,n));M[:old,:old]=parent_reduction.original_mass
    inertia=PointInertia(space,order=mass_order)
    for i in range(old,n,3):
        js=np.arange(i,min(i+3,n));v=np.zeros((n,3));v[js,np.arange(len(js))]=1.
        cols=inertia.apply(v)[:,:len(js)];M[:,js]=cols;M[js,:]=cols.T
        if time.perf_counter()-start>max_seconds:raise TimeoutError('ambient mass budget')
    op=SegmentedGPUOperator(FullCoordinates(space),order=max(7,space.p+1),device='cuda:0')
    lin=op.prepare(np.zeros((n,3)));K=np.zeros((3*n,3*n));K[:3*old,:3*old]=parent_reduction.original_stiffness
    for j in range(3*old,3*n):
        d=np.zeros((n,3));d.ravel()[j]=1.;col=op.action(lin,d).ravel();K[:,j]=col;K[j,:]=col
        if progress and (j-3*old)%18==0:progress('ambient tangent columns',j-3*old,3*(n-old))
        if time.perf_counter()-start>max_seconds:raise TimeoutError('ambient tangent budget')
    rng=np.random.default_rng(20261001);d=rng.normal(size=(n,3))
    direct=inertia.apply(d);mass_error=float(la.norm(M@d-direct)/la.norm(direct))
    higher=PointInertia(space,order=mass_order+1).apply(d)
    mass_order_difference=float(la.norm(direct-higher)/la.norm(higher))
    direct=op.action(lin,d).ravel();stiffness_error=float(la.norm(K@d.ravel()-direct)/la.norm(direct))
    if max(mass_error,stiffness_error,mass_order_difference)>2e-5:raise ValueError('new ambient full operator mismatch')
    # Diagnose the full free mass, beyond the legacy carrier-only nullspace.
    f=space.free_scalar_ids;Mf=M[np.ix_(f,f)];scale=1/np.sqrt(np.diag(Mf));ev=la.eigvalsh(scale[:,None]*Mf*scale[None,:])
    cutoff=64*len(f)*np.finfo(float).eps*max(ev[-1],1.)
    nullity=int(np.sum(ev<=cutoff))
    expected=parent_reduction.N.shape[1]
    if nullity!=expected:raise ValueError(f'new enrichment changed physical mass nullity {nullity} != {expected}')
    reduction=Condensation(space,M,K)
    return reduction,dict(seconds=time.perf_counter()-start,mass_order=mass_order,
        full_free_mass_nullity=nullity,rank=reduction.audit,mass_action_relative=mass_error,
        mass_order_difference=mass_order_difference,stiffness_action_relative=stiffness_error,
        original_mass_block_reused_after_verification=True,full_cross_terms=True)
