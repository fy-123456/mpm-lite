"""Nested internal h enrichment of the actually raised p=6 ambient space."""
import copy
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from engine.aniso_phase1.tensor_reference import coordinates
from engine.aniso_phase1.tensor_metrics import sampling
from engine.aniso_phase1.research_sequential_next.reference_space import extend
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_d.stage2.contracts import array_digest


def interior_h(parent):
    e=parent.edges[0]
    cells=[int(np.argmin(abs((e[:-1]+e[1:])/2-c))) for c in (.375,.625)]
    new=copy.copy(parent);new.edges=(np.unique(np.r_[e,*[(e[i]+e[i+1])/2 for i in cells]]),*parent.edges[1:])
    axes=coordinates(new.edges,parent.p)
    # Only x changed. Reinterpolating identical y/z nodal coordinates stores
    # seven entries per identity row, including zeros, then multiplies their
    # storage in the Kronecker product. Use exact identity on unchanged axes.
    px=sampling(parent.edges[0],parent.p,axes[0]);px.eliminate_zeros()
    tensor=sp.kron(px,sp.eye(parent.shape[1]*parent.shape[2],format='csr'),format='csr')
    new.raw=(tensor@parent.raw).tocsr();new.shape=tuple(len(x) for x in axes)
    new.prolong=tuple(sampling(e,2,x) for e,x in zip(parent.oldedges,axes))
    new.signature=digest(dict(parent=parent.signature,edges=[x.tolist() for x in new.edges],p=new.p,meaning='nested internal h on p6'))
    x,y,z=np.meshgrid(*axes,indexing='ij');yy=2*(y-axes[1][0])/(axes[1][-1]-axes[1][0])-1;zz=2*(z-axes[2][0])/(axes[2][-1]-axes[2][0])-1
    trans=[np.ones_like(yy),yy,zz,yy*zz,.5*(5*yy**3-3*yy),.5*(5*zz**3-3*zz)]
    columns=[];labels=[]
    for cell in cells:
        lo,hi=float(e[cell]),float((e[cell]+e[cell+1])/2);t=(x-lo)/(hi-lo);bubble=np.where((x>lo)&(x<hi),4*t*(1-t),0.)
        for j,mode in enumerate(trans):
            value=(bubble*mode).ravel();scale=la.norm(value);columns.append(value/scale)
            labels.append(dict(cell=cell,support=[lo,hi],transverse=j,normalization=float(scale)))
    basis=np.column_stack(columns);result=extend(new,basis,label='reference-next-internal-h-on-p6')
    raw=result.raw.copy();raw.sum_duplicates();raw.sort_indices()
    for a in (raw.data,raw.indices,raw.indptr):a.setflags(write=False)
    result.raw=raw;result.signature=digest(dict(ambient=new.signature,basis=array_digest(basis)))
    return result,dict(p=new.p,cells=cells,shape=result.shape,extra_scalar=12,labels=labels,
                       coverage='internal x half-cell bubbles; y/z cubic transverse modes; zero grip trace')


def nested_p_coefficients(full,old_definition,new_definition,base_dofs,new_dofs):
    result=np.zeros((new_dofs,3));result[:base_dofs]=full[:base_dofs]
    def key(v):return (v['cell'],v['degree'],v['transverse'])
    lookup={key(v):(i,v) for i,v in enumerate(new_definition['labels'])}
    for i,old in enumerate(old_definition['labels']):
        j,new=lookup[key(old)];result[base_dofs+j]=full[base_dofs+i]*new['normalization']/old['normalization']
    return result
