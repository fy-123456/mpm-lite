"""D3 with only volume reduction moved to GPU, using existing cell ownership.

All current-state F/G/H and complete P are retained. No volume or H from one q
is reused at another q. The inherited owned six-entry cache is unchanged.
"""
import time
from collections import OrderedDict
import numpy as np
import warp as wp
from engine.aniso_phase1.research_pressure3d_next.geometry import BlockGeometry,gather
from engine.aniso_phase1.research_cost_phase_next.pressure import cell_geometry_kernel
from engine.aniso_phase1.research_d.identity import digest

@wp.kernel
def volume_partial(J:wp.array(dtype=wp.float64),w:wp.array(dtype=wp.float64),indices:wp.array(dtype=wp.int32),starts:wp.array(dtype=wp.int32),stops:wp.array(dtype=wp.int32),out:wp.array(dtype=wp.float64)):
    block=wp.tid();v=wp.float64(0.)
    for cursor in range(starts[block],stops[block]):
        i=indices[cursor];v+=w[i]*J[i]
    out[block]=v

@wp.kernel
def volume_collect(partial:wp.array(dtype=wp.float64),offsets:wp.array(dtype=wp.int32),out:wp.array(dtype=wp.float64)):
    cell=wp.tid();v=wp.float64(0.)
    for b in range(offsets[cell],offsets[cell+1]):v+=partial[b]
    out[cell]=v

class DeviceVolumeGeometry(BlockGeometry):
    @classmethod
    def adopt(cls,g):
        tick=time.perf_counter();obj=object.__new__(cls);obj.__dict__=g.__dict__.copy();obj.cache=OrderedDict();obj.profile={}
        obj.additional_bytes=8*(g.assembler.blocks+g.cells)
        if g.identity['construction_metadata_peak_bytes']+obj.additional_bytes>g.cap_bytes:raise MemoryError('volume workspace exceeds frozen geometry cap')
        g.op.memory_budget.observe(obj.additional_bytes)
        obj._original_assembler=g.assembler;obj._original_P=g.model.reduction.P
        obj.implementation=dict(schema='device-volume-block-reduction-v1',geometry=g.identity,space=g.model.reduction.signature,P_sha256=digest(g.model.reduction.P.tolist()),device=str(g.op.device),context=int(g.op.device.context or 0),dtype='float64',additional_peak_bytes=obj.additional_bytes,ownership='same RT0 permutation, block starts/stops/offsets; no new ownership metadata')
        # No persistent transient buffer: release on every evaluate, unchanged
        # failed-step/cache lifecycle, bounded by original 70 percent guard.
        wp.synchronize_device(g.op.device);obj.additional_setup_s=time.perf_counter()-tick
        return obj

    def evaluate(self,q):
        if self.assembler is not self._original_assembler or self.model.reduction.P is not self._original_P:raise ValueError('volume metadata owner changed; reconstruct adapter')
        key=np.asarray(q).tobytes()
        if key in self.cache:
            self.cache.move_to_end(key);return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in self.cache[key].items()}
        self.op.memory_budget.observe(self.count*160+self.max_local_points*80+self.additional_bytes)
        tick=time.perf_counter();F=self.field(q);self._record('field',tick)
        tick=time.perf_counter();cof=wp.empty_like(F);J=wp.empty(self.count,dtype=wp.float64,device=self.op.device);unused=wp.empty_like(J);bad=wp.zeros(1,dtype=wp.int32,device=self.op.device)
        wp.launch(cell_geometry_kernel,dim=self.count,inputs=[F,cof,J,unused,bad,.1],device=self.op.device)
        if bad.numpy()[0]:raise ValueError('D3 invalid current detF')
        a=self.assembler;partial=wp.empty(a.blocks,dtype=wp.float64,device=self.op.device);volume=wp.empty(self.cells,dtype=wp.float64,device=self.op.device)
        wp.launch(volume_partial,dim=a.blocks,inputs=[J,a.weights,a.indices,a.starts,a.stops,partial],device=self.op.device)
        wp.launch(volume_collect,dim=self.cells,inputs=[partial,a.offsets,volume],device=self.op.device)
        V=volume.numpy()
        if not np.isfinite(V).all() or np.any(V<=0):raise ValueError('invalid current cell volume')
        self._record('cofactor_volume_download',tick)
        tick=time.perf_counter();local_cof=wp.empty(self.max_local_points,dtype=wp.mat33d,device=self.op.device);local_w=wp.empty(self.max_local_points,dtype=wp.float64,device=self.op.device);G=[]
        for b,layout,mp,_ in self.local:
            n=int(np.prod(layout[1]));(x0,_),(y0,_),(z0,_)=b;_,cy,cz=layout[1]
            wp.launch(gather,dim=n,inputs=[cof,a.weights,local_cof,local_w,x0,y0,z0,cy,cz,self.shape[1],self.shape[2]],device=self.op.device)
            grad=mp.gradient_adjoint(local_cof[:n],layout,local_w[:n]).numpy().reshape(self.model.parent.ndof,3)
            G.append(self.model.reduction.P.T@grad)
        self._record('local_gradient',tick);tick=time.perf_counter();H,minJ=a.assemble(F);self._record('H',tick)
        value=dict(volume=V,gradient=np.array(G),H=H,min_detF=minJ);self.cache[key]=value
        while len(self.cache)>6:self.cache.popitem(last=False)
        return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in value.items()}
