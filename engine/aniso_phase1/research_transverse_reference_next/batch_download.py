"""DV with one owned parent-gradient download per geometry evaluation.

Only download batching changes. Each local adjoint and the complete P are retained.
The staging vector is call-local, so failed evaluation cannot leak buffered values.
"""
import time
import numpy as np
import warp as wp
from engine.aniso_phase1.research_transverse_next.device_volume import DeviceVolumeGeometry,volume_partial,volume_collect
from engine.aniso_phase1.research_pressure3d_next.geometry import gather
from engine.aniso_phase1.research_cost_phase_next.pressure import cell_geometry_kernel


class BatchDownloadGeometry(DeviceVolumeGeometry):
    @classmethod
    def adopt(cls,g):
        tick=time.perf_counter();obj=super().adopt(g)
        obj.gradient_staging_bytes=3*g.cells*g.model.parent.ndof*3*8
        obj.additional_bytes+=obj.gradient_staging_bytes
        if g.identity['construction_metadata_peak_bytes']+obj.additional_bytes>g.cap_bytes:
            raise MemoryError('batched gradient staging exceeds frozen cap')
        g.op.memory_budget.observe(obj.additional_bytes)
        obj.implementation=dict(obj.implementation,schema='batch-gradient-download-v1',gradient_staging_peak_bytes=obj.gradient_staging_bytes,additional_peak_bytes=obj.additional_bytes,gradient_downloads_per_uncached_evaluate=1,parent_adjoint_unchanged=True,complete_P=True)
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
        tick=time.perf_counter();local_cof=wp.empty(self.max_local_points,dtype=wp.mat33d,device=self.op.device);local_w=wp.empty(self.max_local_points,dtype=wp.float64,device=self.op.device);width=self.model.parent.ndof*3;gradients=wp.empty(self.cells*width,dtype=wp.float64,device=self.op.device)
        for cell,(b,layout,mp,_) in enumerate(self.local):
            n=int(np.prod(layout[1]));(x0,_),(y0,_),(z0,_)=b;_,cy,cz=layout[1]
            wp.launch(gather,dim=n,inputs=[cof,a.weights,local_cof,local_w,x0,y0,z0,cy,cz,self.shape[1],self.shape[2]],device=self.op.device)
            grad=mp.gradient_adjoint(local_cof[:n],layout,local_w[:n])
            wp.copy(gradients,grad,dest_offset=cell*width,count=width)
        raw=gradients.numpy().reshape(self.cells,self.model.parent.ndof,3)
        G=np.einsum('ij,cik->cjk',self.model.reduction.P,raw,optimize=True)
        self._record('local_gradient',tick);tick=time.perf_counter();H,minJ=a.assemble(F);self._record('H',tick)
        value=dict(volume=V,gradient=np.array(G),H=H,min_detF=minJ);self.cache[key]=value
        while len(self.cache)>6:self.cache.popitem(last=False)
        return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in value.items()}
