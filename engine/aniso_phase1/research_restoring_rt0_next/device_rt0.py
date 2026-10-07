"""Bounded device assembly of the unchanged current-geometry RT0 tensor.

The point-wise deformation stays on device. Only 21 local integrals per cell
and small validity reductions are downloaded; no H survives a changed state.
"""
import time
import numpy as np
import scipy.linalg as la
import warp as wp
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_pressure_window_next.coupled import OwnedGeometry
from engine.aniso_phase1.research_cost_phase_next.pressure import cell_geometry_kernel


@wp.kernel
def integrate(F:wp.array(dtype=wp.mat33d),weights:wp.array(dtype=wp.float64),indices:wp.array(dtype=wp.int32),
              starts:wp.array(dtype=wp.int32),stops:wp.array(dtype=wp.int32),cells:wp.array(dtype=wp.int32),
              xs:wp.array(dtype=wp.float64),ys:wp.array(dtype=wp.float64),zs:wp.array(dtype=wp.float64),
              ny:int,nz:int,bounds:wp.array2d(dtype=wp.float64),volumes:wp.array(dtype=wp.float64),
              pa:wp.array(dtype=wp.int32),pb:wp.array(dtype=wp.int32),invk:wp.mat33d,
              partial:wp.array2d(dtype=wp.float64),mins:wp.array(dtype=wp.float64),bad:wp.array(dtype=wp.int32)):
    block,pair=wp.tid();cell=cells[block];a=pa[pair];b=pb[pair];aa=a//2;bb=b//2
    value=wp.float64(0.0);minimum=wp.float64(1.e100)
    for cursor in range(starts[block],stops[block]):
        i=indices[cursor];f=F[i];J=wp.determinant(f)
        if not wp.isfinite(J) or J<=wp.float64(.1):
            wp.atomic_add(bad,0,1)
        else:
            minimum=wp.min(minimum,J)
            ix=i//(ny*nz);iy=(i//nz)%ny;iz=i%nz
            X=wp.vec3d(xs[ix],ys[iy],zs[iz])
            phi_a=(bounds[cell,2*aa+1]-X[aa])/volumes[cell]
            phi_b=(bounds[cell,2*bb+1]-X[bb])/volumes[cell]
            if a%2==1:phi_a=(X[aa]-bounds[cell,2*aa])/volumes[cell]
            if b%2==1:phi_b=(X[bb]-bounds[cell,2*bb])/volumes[cell]
            ca=wp.vec3d(f[0,aa],f[1,aa],f[2,aa]);cb=wp.vec3d(f[0,bb],f[1,bb],f[2,bb])
            value+=weights[i]*phi_a*phi_b*wp.dot(ca,invk*cb)/J
    partial[block,pair]=value
    if pair==0:mins[block]=minimum


@wp.kernel
def collect(partial:wp.array2d(dtype=wp.float64),mins:wp.array(dtype=wp.float64),
            offsets:wp.array(dtype=wp.int32),local:wp.array2d(dtype=wp.float64),local_min:wp.array(dtype=wp.float64)):
    cell,pair=wp.tid();value=wp.float64(0.);minimum=wp.float64(1.e100)
    for b in range(offsets[cell],offsets[cell+1]):
        value+=partial[b,pair]
        minimum=wp.min(minimum,mins[b])
    local[cell,pair]=value
    if pair==0:local_min[cell]=minimum


class DeviceRT0:
    def __init__(self,geometry,*,cap_bytes,chunk=256):
        begun=time.perf_counter();g=geometry;t=g.topology;self.device=g.op.device;self.count=g.count;self.shape=tuple(g.shape);self.faces=np.array(t.faces,copy=True);self.nflux=t.nflux;self.cell_count=t.cells
        if chunk<1 or int(chunk)!=chunk:raise ValueError('positive integer chunk required')
        if not np.isfinite(g.mobility).all() or not np.allclose(g.mobility,g.mobility.T) or la.eigvalsh(g.mobility)[0]<=0:raise ValueError('full SPD mobility required')
        if self.count!=int(np.prod(self.shape)) or self.count>=2**31:raise ValueError('bounded tensor layout required')
        if g.X.shape!=(self.count,3) or not np.isfinite(g.total_weights).all() or np.any(g.total_weights<0):raise ValueError('invalid integration metadata')
        # Verify the tensor layout without constructing another complete X array.
        for axis,n in enumerate(self.shape):
            shaped=g.X[:,axis].reshape(self.shape)
            shape=[1,1,1];shape[axis]=n
            if not np.array_equal(np.broadcast_to(g.points[axis].reshape(shape),self.shape),shaped):raise ValueError('X is not the bound tensor grid')
        starts=[];stops=[];cells=[];offsets=[0];permutation=[];cursor=0
        for cell in range(t.cells):
            ix=np.flatnonzero(g.cell_ids==cell).astype(np.int32)
            if len(ix)==0:raise ValueError('pressure cell has no integration points')
            permutation.append(ix)
            for lo in range(0,len(ix),chunk):starts.append(cursor+lo);stops.append(cursor+min(lo+chunk,len(ix)));cells.append(cell)
            cursor+=len(ix);offsets.append(len(starts))
        if cursor!=self.count:raise ValueError('invalid integration cell ownership')
        host=[np.asarray(v,dtype=np.int32) for v in (starts,stops,cells,offsets)];perm=np.concatenate(permutation);pairs=np.array([(a,b) for a in range(6) for b in range(a,6)],dtype=np.int32)
        owned=sum(v.nbytes for v in host)+perm.nbytes+g.total_weights.nbytes+sum(x.nbytes for x in g.points)+t.cells*7*8+pairs.nbytes
        transient=len(starts)*22*8+t.cells*22*8+4
        if owned+transient>cap_bytes:raise MemoryError('device RT0 additional bytes exceed frozen cap')
        guard=getattr(g.op,'memory_budget',None)
        if guard:guard.observe(owned+transient)
        self.starts,self.stops,self.cells,self.offsets=[wp.array(v,dtype=wp.int32,device=self.device) for v in host]
        self.indices=wp.array(perm,dtype=wp.int32,device=self.device);self.weights=wp.array(np.array(g.total_weights,copy=True),dtype=wp.float64,device=self.device)
        self.axes=[wp.array(np.array(x,copy=True),dtype=wp.float64,device=self.device) for x in g.points]
        self.bounds=wp.array(np.array(t.cell_bounds).reshape(t.cells,6),dtype=wp.float64,device=self.device);self.volumes=wp.array(np.array(t.V0,copy=True),dtype=wp.float64,device=self.device)
        self.pa=wp.array(pairs[:,0].copy(),dtype=wp.int32,device=self.device);self.pb=wp.array(pairs[:,1].copy(),dtype=wp.int32,device=self.device);self.pairs=pairs
        self.invk=wp.mat33d(la.solve(g.mobility,np.eye(3),assume_a='pos'));self.blocks=len(starts);self.bytes=int(owned);self.transient_bytes=int(transient);self.guard=guard
        self.identity=dict(schema='current-device-RT0-chunk-v1',physical_geometry=g.identity,chunk=chunk,shape=list(self.shape),additional_peak_bytes=owned+transient,cap_bytes=int(cap_bytes),current_F_required=True)
        wp.synchronize_device(self.device);self.build_seconds=time.perf_counter()-begun

    def assemble(self,F):
        if F.dtype!=wp.mat33d or F.shape!=(self.count,) or F.device!=wp.get_device(self.device):raise ValueError('wrong current F layout or device')
        if self.guard:self.guard.observe(self.transient_bytes)
        partial=wp.empty((self.blocks,21),dtype=wp.float64,device=self.device);mins=wp.empty(self.blocks,dtype=wp.float64,device=self.device);bad=wp.zeros(1,dtype=wp.int32,device=self.device)
        local=wp.empty((self.cell_count,21),dtype=wp.float64,device=self.device);local_min=wp.empty(self.cell_count,dtype=wp.float64,device=self.device)
        wp.launch(integrate,dim=(self.blocks,21),inputs=[F,self.weights,self.indices,self.starts,self.stops,self.cells,*self.axes,self.shape[1],self.shape[2],self.bounds,self.volumes,self.pa,self.pb,self.invk,partial,mins,bad],device=self.device)
        wp.launch(collect,dim=(self.cell_count,21),inputs=[partial,mins,self.offsets,local,local_min],device=self.device)
        if bad.numpy()[0]:raise ValueError('current RT0 deformation invalid')
        values=local.numpy();minimum=float(local_min.numpy().min());H=np.zeros((self.nflux,self.nflux))
        if not np.isfinite(values).all():raise ValueError('nonfinite current RT0 tensor')
        for c,faces in enumerate(self.faces):
            mat=np.zeros((6,6))
            for pair,(a,b) in enumerate(self.pairs):mat[a,b]=values[c,pair];mat[b,a]=values[c,pair]
            H[np.ix_(faces,faces)]+=mat
        return H,minimum


class DeviceGeometry(OwnedGeometry):
    @classmethod
    def from_owned(cls,original,*,cap_bytes):
        obj=cls.__new__(cls);obj.__dict__=original.__dict__.copy();obj.cache=type(original.cache)()
        obj.assembler=DeviceRT0(original,cap_bytes=cap_bytes)
        obj.identity=dict(original.identity,implementation=obj.assembler.identity)
        return obj

    def evaluate(self,q):
        key=np.asarray(q).tobytes()
        if key in self.cache:
            self.cache.move_to_end(key);return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in self.cache[key].items()}
        F=self.field(q);cof=wp.empty_like(F);J=wp.empty(self.count,dtype=wp.float64,device=self.op.device);unused=wp.empty_like(J);bad=wp.zeros(1,dtype=wp.int32,device=self.op.device)
        wp.launch(cell_geometry_kernel,dim=self.count,inputs=[F,cof,J,unused,bad,.1],device=self.op.device)
        if bad.numpy()[0]:raise ValueError('cell geometry leaves positive J range')
        jj=J.numpy();V=[];G=[]
        for w,gw in zip(self.weights,self.gpu_weights):
            grad=self.op.maps.gradient_adjoint(cof,self.layout,gw).numpy().reshape(self.model.parent.ndof,3);G.append(self.model.reduction.P.T@grad);V.append(float(w@jj))
        H,minJ=self.assembler.assemble(F)
        value=dict(volume=np.array(V),gradient=np.array(G),H=H,min_detF=min(float(jj.min()),minJ));self.cache[key]=value
        while len(self.cache)>6:self.cache.popitem(last=False)
        return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in value.items()}
