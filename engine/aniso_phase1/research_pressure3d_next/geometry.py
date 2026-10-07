"""Bounded Cartesian cell blocks; no per-cell full-domain masks or dense basis."""
import copy
import time
from collections import OrderedDict
import numpy as np
import scipy.linalg as la
import warp as wp
from engine.aniso_phase1.tensor_metrics import sampling, quadrature_axis
from engine.aniso_phase1.research_d.stage2.gpu_space import CSR
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_cross_direction_next.geometry import CellGeometry
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology
from engine.aniso_phase1.research_pressure_startup_next.reduced_geometry import bounded_segments, BoundedCSR
from engine.aniso_phase1.research_restoring_rt0_next.device_rt0 import DeviceRT0
from engine.aniso_phase1.research_cost_phase_next.pressure import cell_geometry_kernel


def blocks(points, cuts):
    """Each local tensor is ordered exactly like a slice of the global tensor."""
    ranges=[]
    for x,e in zip(points,cuts):
        x=np.asarray(x);e=np.asarray(e);owner=np.searchsorted(e,x,side='right')-1
        if x.ndim!=1 or np.any(np.diff(x)<=0) or np.any(owner<0) or np.any(owner>=len(e)-1):
            raise ValueError('quadrature outside explicit cells')
        axis=[]
        for c in range(len(e)-1):
            ids=np.flatnonzero(owner==c)
            if not len(ids) or not np.array_equal(ids,np.arange(ids[0],ids[-1]+1)):
                raise ValueError('nonempty contiguous axis ownership required')
            axis.append((int(ids[0]),int(ids[-1]+1)))
        ranges.append(axis)
    return [tuple(ranges[a][idx[a]] for a in range(3)) for idx in np.ndindex(*(len(v) for v in ranges))]


def indices(bounds, shape):
    (a,b),(c,d),(e,f)=bounds;_,ny,nz=shape
    return ((np.arange(a,b,dtype=np.int64)[:,None,None]*ny+
             np.arange(c,d,dtype=np.int64)[None,:,None])*nz+
             np.arange(e,f,dtype=np.int64)[None,None,:]).ravel().astype(np.int32)


@wp.kernel
def gather(P:wp.array(dtype=wp.mat33d),weights:wp.array(dtype=wp.float64),
           out:wp.array(dtype=wp.mat33d),w:wp.array(dtype=wp.float64),
           x0:int,y0:int,z0:int,cy:int,cz:int,ny:int,nz:int):
    i=wp.tid();x=x0+i//(cy*cz);y=y0+(i//cz)%cy;z=z0+i%cz;k=(x*ny+y)*nz+z
    out[i]=P[k];w[i]=weights[k]


class BlockRT0(DeviceRT0):
    """Same authenticated integration kernels, without global XYZ ownership masks."""
    def __init__(self,g,cap_bytes,chunk=256):
        tick=time.perf_counter();t=g.topology;self.device=g.op.device
        self.count=g.count;self.shape=g.shape;self.faces=t.faces.copy();self.nflux=t.nflux;self.cell_count=t.cells
        starts=[];stops=[];cells=[];offsets=[0];perm=np.empty(g.count,dtype=np.int32);cursor=0
        for c,b in enumerate(g.blocks):
            ids=indices(b,g.shape);size=len(ids);perm[cursor:cursor+size]=ids
            for lo in range(0,size,chunk):starts.append(cursor+lo);stops.append(cursor+min(lo+chunk,size));cells.append(c)
            cursor+=size;offsets.append(len(starts))
        if cursor!=g.count:raise ValueError('cell block coverage differs')
        host=[np.array(v,dtype=np.int32) for v in (starts,stops,cells,offsets)]
        pairs=np.array([(a,b) for a in range(6) for b in range(a,6)],dtype=np.int32)
        owned=sum(x.nbytes for x in host)+perm.nbytes+g.total_weights.nbytes+sum(x.nbytes for x in g.points)+t.cells*7*8+pairs.nbytes
        transient=len(starts)*22*8+t.cells*22*8+4
        if owned+transient>cap_bytes:raise MemoryError('D3 RT0 metadata cap')
        self.guard=g.op.memory_budget;self.guard.observe(owned+transient)
        self.starts,self.stops,self.cells,self.offsets=[wp.array(x,dtype=wp.int32,device=self.device) for x in host]
        self.indices=wp.array(perm,dtype=wp.int32,device=self.device)
        self.weights=wp.array(g.total_weights,dtype=wp.float64,device=self.device)
        self.axes=[wp.array(x,dtype=wp.float64,device=self.device) for x in g.points]
        self.bounds=wp.array(np.array(t.cell_bounds).reshape(t.cells,6),dtype=wp.float64,device=self.device)
        self.volumes=wp.array(t.V0.copy(),dtype=wp.float64,device=self.device)
        self.pa=wp.array(pairs[:,0].copy(),dtype=wp.int32,device=self.device);self.pb=wp.array(pairs[:,1].copy(),dtype=wp.int32,device=self.device)
        self.pairs=pairs;self.invk=wp.mat33d(la.solve(g.mobility,np.eye(3),assume_a='pos'))
        self.blocks=len(starts);self.bytes=int(owned);self.transient_bytes=int(transient)
        self.construction_staging_bytes=perm.nbytes+sum(x.nbytes for x in host)
        self.identity=dict(schema='pressure3d-block-RT0-v1',shape=list(self.shape),cells=t.cells,chunk=chunk,bytes=self.bytes)
        wp.synchronize_device(self.device);self.build_seconds=time.perf_counter()-tick


class BlockGeometry(CellGeometry):
    """D3 baseline: exact 3-D blocks, conservative x-only raw-column pruning.

    The x bound retains transverse zero columns deliberately. It cannot drop a
    contributing node; multi-interval pruning is a separate performance choice.
    """
    def __init__(self,model,cuts,mobility,order=7):
        tick=time.perf_counter();self.model=model;self.op=model.operator;self.order=order
        self.topology=ReferenceTopology(cuts);t=self.topology
        if t.shape not in ((32,1,1),(32,2,1),(32,1,2),(32,2,2)):
            raise ValueError('D3 only supports registered 32x{1,2}x{1,2} research grids')
        self.cells=t.cells;self.V0=t.V0;self.B=t.B;self.cache=OrderedDict();self.profile={}
        self.mobility=np.array(mobility,dtype=float,copy=True)
        if self.mobility.shape!=(3,3) or not np.isfinite(self.mobility).all() or not np.allclose(self.mobility,self.mobility.T) or la.eigvalsh(self.mobility)[0]<=0:
            raise ValueError('finite full SPD mobility required')
        self.mobility.setflags(write=False);s=model.parent
        if any(not np.allclose([e[0],e[-1]],t.bounds[a],atol=1e-13,rtol=0) for a,e in enumerate(s.edges)):
            raise ValueError('pressure domain differs from solid')
        axes=[quadrature_axis(np.unique(np.r_[e,t.cuts[a]]),order) for a,e in enumerate(s.edges)]
        self.points=tuple(x[0] for x in axes);self.shape=tuple(map(len,self.points));self.count=int(np.prod(self.shape))
        if self.count>=2**31:raise ValueError('quadrature exceeds int32 scope')
        self.cap_bytes=min(256*2**20,int(.02*self.op.memory_budget.initial_free))
        # One CPU weight vector, one shared GPU vector + point permutation.
        lower_bound=self.count*(8+8+4)+((self.count+255)//256)*24*8
        if lower_bound>self.cap_bytes:raise MemoryError('D3 construction lower bound exceeds cap')
        self.op.memory_budget.observe(lower_bound)
        self.total_weights=(axes[0][1][:,None,None]*axes[1][1][None,:,None]*axes[2][1][None,None,:]).ravel()
        self.total_weights.setflags(write=False);self.blocks=blocks(self.points,t.cuts)
        self.layout=self.op.maps.layout(self.points);self.local=[];axis_cache={};raw_cache={}
        source=s.raw.tocsc();source.sort_indices();maps=self.op.maps
        if not np.array_equal(source.indptr,maps.raw[1].ptr.numpy()):raise ValueError('transpose row identity differs')
        local_bytes=0
        for c,b in enumerate(self.blocks):
            sl=tuple(slice(*v) for v in b);w=self.total_weights.reshape(self.shape)[sl]
            if not np.isclose(w.sum(),self.V0[c],rtol=1e-12,atol=1e-13):raise ValueError('D3 cell volume coverage')
            pairs=[]
            for a,(lo,hi) in enumerate(b):
                key=(a,lo,hi)
                if key not in axis_cache:
                    pair=[]
                    for deriv in (False,True):
                        mat=sampling(s.edges[a],s.p,self.points[a][lo:hi],deriv)
                        pair.append((CSR(mat,self.op.device),CSR(mat.T,self.op.device)))
                    axis_cache[key]=pair;local_bytes+=sum(x.bytes for pair0 in pair for x in pair0)
                pairs.append(axis_cache[key])
            xb=b[0]
            if xb not in raw_cache:
                x=self.points[0][slice(*xb)];a=sampling(s.edges[0],s.p,x);d=sampling(s.edges[0],s.p,x,True)
                active=np.unique(np.r_[a.indices,d.indices]);lo=int(active[0])*s.shape[1]*s.shape[2];hi=(int(active[-1])+1)*s.shape[1]*s.shape[2]
                meta,used=bounded_segments(source.indptr,source.indices,lo,hi)
                if self.total_weights.nbytes+local_bytes+sum(x.nbytes for x in meta)*2+lower_bound-self.total_weights.nbytes>self.cap_bytes:
                    raise MemoryError('D3 local metadata estimate exceeds cap')
                mp=copy.copy(maps);mp.layouts=dict(maps.layouts);bounded=BoundedCSR.from_shared(maps.raw[1],meta);mp.raw=(maps.raw[0],bounded)
                raw_cache[xb]=(mp,used);local_bytes+=bounded.extra_bytes
            mp,used=raw_cache[xb];self.local.append((b,(pairs,tuple(hi-lo for lo,hi in b)),mp,used))
        del source
        self.assembler=BlockRT0(self,self.cap_bytes)
        global_layout_bytes=sum(x.bytes for a in self.layout[0] for pair in a for x in pair)
        self.local_bytes=self.total_weights.nbytes+local_bytes+global_layout_bytes+self.assembler.bytes
        peak_metadata=self.local_bytes+self.assembler.construction_staging_bytes
        if peak_metadata>self.cap_bytes:raise MemoryError('D3 complete metadata construction cap')
        self.max_local_points=max(int(np.prod(layout[1])) for _,layout,_,_ in self.local)
        self.identity=dict(schema='pressure3d-local-Simpson-v1',solid=model.reduction.signature,cuts=[x.tolist() for x in t.cuts],mobility=self.mobility.tolist(),order=order,
            full_material_and_mass_unchanged=True,shape=self.shape,points=self.count,full_domain_masks=0,global_X_bytes=0,
            volume_gradient='full P, local tensor adjoints, conservative x envelope; Simpson exact',metadata_bytes=int(self.local_bytes),construction_metadata_peak_bytes=int(peak_metadata),cap_bytes=self.cap_bytes,
            raw_values_shared=True,raw_column_pruning='x envelope, retains all transverse nodes',cache_entries=6,owned_returns=True)
        wp.synchronize_device(self.op.device);self.build_seconds=time.perf_counter()-tick

    def _record(self,name,t):
        wp.synchronize_device(self.op.device);v=self.profile.setdefault(name,dict(calls=0,seconds=0.));v['calls']+=1;v['seconds']+=time.perf_counter()-t

    def evaluate(self,q):
        key=np.asarray(q).tobytes()
        if key in self.cache:
            self.cache.move_to_end(key);return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in self.cache[key].items()}
        self.op.memory_budget.observe(self.count*160+self.max_local_points*80)
        tick=time.perf_counter();F=self.field(q);self._record('field',tick)
        tick=time.perf_counter();cof=wp.empty_like(F);J=wp.empty(self.count,dtype=wp.float64,device=self.op.device);unused=wp.empty_like(J);bad=wp.zeros(1,dtype=wp.int32,device=self.op.device)
        wp.launch(cell_geometry_kernel,dim=self.count,inputs=[F,cof,J,unused,bad,.1],device=self.op.device)
        if bad.numpy()[0]:raise ValueError('D3 invalid current detF')
        jj=J.numpy().reshape(self.shape);self._record('cofactor_download',tick)
        tick=time.perf_counter();local_cof=wp.empty(self.max_local_points,dtype=wp.mat33d,device=self.op.device);local_w=wp.empty(self.max_local_points,dtype=wp.float64,device=self.op.device);V=[];G=[]
        for b,layout,mp,_ in self.local:
            n=int(np.prod(layout[1]));(x0,_),(y0,_),(z0,_)=b;_,cy,cz=layout[1]
            wp.launch(gather,dim=n,inputs=[cof,self.assembler.weights,local_cof,local_w,x0,y0,z0,cy,cz,self.shape[1],self.shape[2]],device=self.op.device)
            grad=mp.gradient_adjoint(local_cof[:n],layout,local_w[:n]).numpy().reshape(self.model.parent.ndof,3)
            G.append(self.model.reduction.P.T@grad);sl=tuple(slice(*v) for v in b);V.append(float(np.sum(self.total_weights.reshape(self.shape)[sl]*jj[sl])))
        self._record('local_gradient',tick);tick=time.perf_counter();H,minJ=self.assembler.assemble(F);self._record('H',tick)
        value=dict(volume=np.array(V),gradient=np.array(G),H=H,min_detF=min(float(jj.min()),minJ));self.cache[key]=value
        while len(self.cache)>6:self.cache.popitem(last=False)
        return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in value.items()}

    def discrete(self,q0,q1):
        return (self.evaluate(q0)['gradient']+4*self.evaluate((q0+q1)/2)['gradient']+self.evaluate(q1)['gradient'])/6

    def action(self,q,d):
        raise NotImplementedError('D3 tangent action is not qualified; coupled residual uses full G')
