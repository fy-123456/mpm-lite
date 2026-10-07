"""One candidate: remove proven-zero transverse nodal columns from raw adjoints."""
import copy,time
from collections import OrderedDict
import numpy as np
import warp as wp
from .geometry import BlockGeometry
from engine.aniso_phase1.tensor_metrics import sampling
from engine.aniso_phase1.research_pressure_startup_next.reduced_geometry import BoundedCSR
from engine.aniso_phase1.research_d.common_state import StateTransaction
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_restoring_rt0_next.fixture import physical_payload

def transverse_segments(ptr,col,bounds,shape,chunk=128):
    if chunk<1 or int(chunk)!=chunk:raise ValueError('positive segment size')
    (x0,x1),(y0,y1),(z0,z1)=bounds;_,ny,nz=shape
    if any(lo<0 or hi<=lo or hi>n for (lo,hi),n in zip(bounds,shape)):raise ValueError('invalid nodal support block')
    lo=x0*ny*nz;hi=x1*ny*nz;starts=[];stops=[];rows=[0];used=0
    for a,b in zip(ptr[:-1],ptr[1:]):
        row=col[a:b];left=int(a)+int(np.searchsorted(row,lo));right=int(a)+int(np.searchsorted(row,hi));sub=col[left:right]
        yy=(sub//nz)%ny;zz=sub%nz;ids=np.flatnonzero((yy>=y0)&(yy<y1)&(zz>=z0)&(zz<z1));used+=len(ids)
        if len(ids):
            cuts=np.r_[0,np.flatnonzero(np.diff(ids)!=1)+1,len(ids)]
            for l,r in zip(cuts[:-1],cuts[1:]):
                begin=left+int(ids[l]);end=left+int(ids[r-1])+1
                for s in range(begin,end,chunk):starts.append(s);stops.append(min(s+chunk,end))
        rows.append(len(starts))
    return tuple(np.array(v,dtype=np.int32) for v in (starts,stops,rows)),int(used)

class MultirangeGeometry(BlockGeometry):
    @classmethod
    def from_block(cls,g):
        tick=time.perf_counter();s=g.model.parent;source=s.raw.tocsc();source.sort_indices();plans=[];used=[];bounds=[]
        for b,layout,mp,oldused in g.local:
            support=[]
            for a,(lo,hi) in enumerate(b):
                x=g.points[a][lo:hi];v=sampling(s.edges[a],s.p,x);d=sampling(s.edges[a],s.p,x,True)
                active=np.unique(np.r_[v.indices,d.indices]);support.append((int(active[0]),int(active[-1])+1))
            meta,n=transverse_segments(source.indptr,source.indices,support,s.shape);plans.append(meta);used.append(n);bounds.append(support)
        extra=sum(a.nbytes for p in plans for a in p)
        if g.local_bytes+2*extra>g.cap_bytes:raise MemoryError('multirange metadata construction cap')
        g.op.memory_budget.observe(extra);obj=cls.__new__(cls);obj.__dict__=g.__dict__.copy();obj.cache=OrderedDict();obj.profile={};obj.local=[]
        for (b,layout,old,_),meta,n in zip(g.local,plans,used):
            mp=copy.copy(old);mp.layouts=dict(old.layouts);raw=BoundedCSR.from_shared(g.op.maps.raw[1],meta);mp.raw=(old.raw[0],raw);obj.local.append((b,layout,mp,n))
        obj.local_bytes=g.local_bytes+extra
        obj.identity=dict(schema='pressure3d-multirange-v1',parent=g.identity,old_used_nnz=sum(x[3] for x in g.local),used_nnz=sum(used),metadata_extra_bytes=int(extra),construction_peak_bytes=int(g.local_bytes+2*extra),nodal_bounds=bounds,raw_values_and_P_unchanged=True)
        wp.synchronize_device(g.op.device);obj.additional_setup_s=time.perf_counter()-tick;return obj

def install(c):
    c.validate(c.state);s=c.state;before=physical_payload(s);old=copy.deepcopy(c.identity);g=MultirangeGeometry.from_block(c.geometry)
    c.core.geometry=g;c.core.identity=dict(c.core.identity,geometry=g.identity);c.core.signature=digest(c.core.identity)
    c.identity=dict(schema='pressure3d-multirange-coupling-v1',physical_parent=old,core=c.core.identity);c.signature=digest(c.identity)
    s.child_states['fluid']['model']=c.core.signature;s.child_states['explicit_pressure_grid']=c.signature
    if physical_payload(s)!=before:raise ValueError('multirange changed physical history')
    c.core._transaction=StateTransaction(s,validator=c.validate)
    return dict(physical_payload_exact=True,geometry=g.identity,additional_setup_s=g.additional_setup_s)
