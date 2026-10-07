"""D3 algebra unchanged; split CPU restriction from GPU tensor adjoint timing."""
import time
import numpy as np
import warp as wp
from engine.aniso_phase1.research_pressure3d_next.geometry import BlockGeometry,gather
from engine.aniso_phase1.research_cost_phase_next.pressure import cell_geometry_kernel

class ProfiledGeometry(BlockGeometry):
    @classmethod
    def adopt(cls,g):
        obj=object.__new__(cls);obj.__dict__=g.__dict__.copy()
        obj.profile={};obj.cache_hits=0;obj.cache_misses=0
        return obj

    def evaluate(self,q):
        key=np.asarray(q).tobytes()
        if key in self.cache:
            self.cache_hits+=1;self.cache.move_to_end(key)
            return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in self.cache[key].items()}
        self.cache_misses+=1
        self.op.memory_budget.observe(self.count*160+self.max_local_points*80)
        tick=time.perf_counter();F=self.field(q);self._record('field',tick)
        tick=time.perf_counter();cof=wp.empty_like(F);J=wp.empty(self.count,dtype=wp.float64,device=self.op.device);unused=wp.empty_like(J);bad=wp.zeros(1,dtype=wp.int32,device=self.op.device)
        wp.launch(cell_geometry_kernel,dim=self.count,inputs=[F,cof,J,unused,bad,.1],device=self.op.device)
        if bad.numpy()[0]:raise ValueError('D3 invalid current detF')
        jj=J.numpy().reshape(self.shape);self._record('cofactor_download',tick)
        tick=time.perf_counter();local_cof=wp.empty(self.max_local_points,dtype=wp.mat33d,device=self.op.device);local_w=wp.empty(self.max_local_points,dtype=wp.float64,device=self.op.device)
        self._record('local_buffers',tick);V=[];G=[]
        for b,layout,mp,_ in self.local:
            tick=time.perf_counter();n=int(np.prod(layout[1]));(x0,_),(y0,_),(z0,_)=b;_,cy,cz=layout[1]
            wp.launch(gather,dim=n,inputs=[cof,self.assembler.weights,local_cof,local_w,x0,y0,z0,cy,cz,self.shape[1],self.shape[2]],device=self.op.device)
            grad=mp.gradient_adjoint(local_cof[:n],layout,local_w[:n]).numpy().reshape(self.model.parent.ndof,3)
            self._record('gather_local_adjoint_download',tick);tick=time.perf_counter()
            G.append(self.model.reduction.P.T@grad)
            sl=tuple(slice(*v) for v in b);V.append(float(np.sum(self.total_weights.reshape(self.shape)[sl]*jj[sl])))
            self._record('P_restriction_and_volume',tick)
        tick=time.perf_counter();H,minJ=self.assembler.assemble(F);self._record('H',tick)
        value=dict(volume=np.array(V),gradient=np.array(G),H=H,min_detF=min(float(jj.min()),minJ));self.cache[key]=value
        while len(self.cache)>6:self.cache.popitem(last=False)
        return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in value.items()}
