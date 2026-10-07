"""Factor the common x transpose in the exact local gradient adjoint.

D_x^T S_y^T S_z^T g_x + S_x^T(D_y^T S_z^T g_y + S_y^T D_z^T g_z).
All field maps, material rules and complete reduced P are unchanged. Temporary
arrays are call-owned; no graph, persistent staging or transaction state added.
"""
import time
import numpy as np
import warp as wp
from engine.aniso_phase1.research_d.stage2.gpu_space import column_kernel,add_kernel
from engine.aniso_phase1.research_transverse_reference_next.batch_download import BatchDownloadGeometry

class FactoredAdjoint:
    def __init__(self,parent):self.parent=parent
    def __getattr__(self,name):return getattr(self.parent,name)
    def gradient_adjoint(self,P,layout,weights):
        mp=self.parent;pairs,pshape=layout;intermediate=[]
        for j in range(3):
            a=wp.empty(P.size*3,dtype=wp.float64,device=mp.device)
            wp.launch(column_kernel,dim=a.size,inputs=[P,weights,a,j],device=mp.device)
            shape=(*pshape,3)
            for k in (2,1):a,shape=pairs[k][int(k==j)][1].apply(a,shape,k)
            intermediate.append(a)
        # All three branches have identical intermediate shape. Contract x only
        # twice, after summing the two branches which use S_x^T.
        first,_=pairs[0][1][1].apply(intermediate[0],shape,0)
        wp.launch(add_kernel,dim=intermediate[1].size,inputs=[intermediate[1],intermediate[2],intermediate[1]],device=mp.device)
        other,_=pairs[0][0][1].apply(intermediate[1],shape,0)
        wp.launch(add_kernel,dim=first.size,inputs=[first,other,first],device=mp.device)
        return mp.adjoint(first)

class FactoredGradientGeometry(BatchDownloadGeometry):
    @classmethod
    def adopt(cls,g):
        started=time.perf_counter();obj=super().adopt(g)
        if any(isinstance(mp,FactoredAdjoint) for _,_,mp,_ in g.local):raise ValueError('factoring already installed')
        obj.local=[(b,layout,FactoredAdjoint(mp),used) for b,layout,mp,used in g.local]
        # The three y/z intermediates coexist. Conservatively account for two
        # extra such buffers, even though the removed full nodal array is larger.
        extra=2*max(layout[1][0]*mp.space.shape[1]*mp.space.shape[2]*3*8 for _,layout,mp,_ in g.local)
        obj.additional_bytes+=extra
        if g.identity['construction_metadata_peak_bytes']+obj.additional_bytes>g.cap_bytes:raise MemoryError('factored workspace exceeds frozen cap')
        g.op.memory_budget.observe(obj.additional_bytes)
        obj.implementation=dict(obj.implementation,schema='factored-local-gradient-BD-v1',additional_peak_bytes=obj.additional_bytes,extra_intermediate_bytes=extra,complete_P=True,persistent_transient_buffers=False)
        wp.synchronize_device(g.op.device);obj.additional_setup_s=time.perf_counter()-started
        return obj
