"""One bounded batch-four contraction candidate; original sparse adjoints retained."""
import copy,time
from collections import OrderedDict
import numpy as np
import warp as wp
from engine.aniso_phase1.research_boundary_reference_next.geometry import SharedMidpointGeometry
from engine.aniso_phase1.research_cost_phase_next.pressure import cell_geometry_kernel
from engine.aniso_phase1.research_restoring_rt0_next.fixture import physical_payload
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_d.common_state import StateTransaction

@wp.kernel
def contract(P:wp.array(dtype=wp.float64),parents:wp.array(dtype=wp.float64),out:wp.array(dtype=wp.float64),n:int,nr:int):
    b,j=wp.tid();r=j//3;i=j%3;value=wp.float64(0.)
    for k in range(n):value+=P[k*nr+r]*parents[b*n*3+k*3+i]
    out[b*nr*3+j]=value

class BatchedGeometry(SharedMidpointGeometry):
    @classmethod
    def from_shared(cls,g):
        if not isinstance(g,SharedMidpointGeometry):raise ValueError('shared B geometry required')
        obj=cls.__new__(cls);obj.__dict__=g.__dict__.copy();obj.cache=OrderedDict();P=g.model.reduction.P;extra=P.nbytes;temporary=4*3*8*(P.shape[0]+P.shape[1]);cap=min(256*2**20,int(.02*g.op.memory_budget.initial_free))
        if g.local_bytes+2*extra>cap or temporary>64*2**20:raise MemoryError('G1 bounded metadata/workspace budget')
        g.op.memory_budget.observe(extra+temporary);obj.P_device=wp.array(np.ascontiguousarray(P).ravel(),dtype=wp.float64,device=g.op.device);obj.batch_parents=wp.empty(4*P.shape[0]*3,dtype=wp.float64,device=g.op.device);obj.batch_reduced=wp.empty(4*P.shape[1]*3,dtype=wp.float64,device=g.op.device);obj.local_bytes=g.local_bytes+extra
        obj.identity=dict(schema='continuous-batch-four-geometry-v1',parent=copy.deepcopy(g.identity),batch_size=4,parent_coefficients=P.shape[0],reduced_coefficients=P.shape[1],additional_static_bytes=extra,additional_temporary_bytes=temporary,metadata_bytes=obj.local_bytes,metadata_cap_bytes=cap,shared_nodal_arrays=False,full_P_and_cross_terms=True);return obj

    def evaluate(self,q):
        key=np.asarray(q).tobytes()
        if key in self.cache:
            self.cache.move_to_end(key);return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in self.cache[key].items()}
        F=self.field(q);cof=wp.empty_like(F);J=wp.empty(self.count,dtype=wp.float64,device=self.op.device);unused=wp.empty_like(J);bad=wp.zeros(1,dtype=wp.int32,device=self.op.device);wp.launch(cell_geometry_kernel,dim=self.count,inputs=[F,cof,J,unused,bad,.1],device=self.op.device)
        if bad.numpy()[0]:raise ValueError('G1 leaves positive J range')
        jj=J.numpy();V=[];G=[];n,nr=self.model.reduction.P.shape
        for base in range(0,len(self.local),4):
            group=self.local[base:base+4]
            for k,(start,stop,layout,weights) in enumerate(group):
                grad=self.local_maps[base+k].gradient_adjoint(cof[start:stop],layout,weights);wp.copy(self.batch_parents,grad,dest_offset=k*n*3,count=n*3);V.append(float(self.total_weights[start:stop]@jj[start:stop]))
            wp.launch(contract,dim=(len(group),nr*3),inputs=[self.P_device,self.batch_parents,self.batch_reduced,n,nr],device=self.op.device);G.extend(self.batch_reduced.numpy().reshape(4,nr,3)[:len(group)].copy())
        H,minJ=self.assembler.assemble(F);value=dict(volume=np.array(V),gradient=np.array(G),H=H,min_detF=min(float(jj.min()),minJ));self.cache[key]=value
        while len(self.cache)>6:self.cache.popitem(last=False)
        return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in value.items()}

def install(c,previous_bridge=None):
    c.validate(c.state);s=c.state;before=physical_payload(s);old=copy.deepcopy(c.identity);g=BatchedGeometry.from_shared(c.geometry);c.core.geometry=g;c.core.identity=dict(c.core.identity,geometry=g.identity,implementation='bounded-batch-four-gradient');c.core.signature=digest(c.core.identity);c.identity=dict(schema='continuous-batch-four-fixture-v1',physical_parent=old,core=c.core.identity);c.signature=digest(c.identity);s.child_states['fluid']['model']=c.core.signature;s.child_states['explicit_pressure_grid']=c.signature
    if physical_payload(s)!=before:raise ValueError('G1 changed physical state')
    c.core._transaction=StateTransaction(s,validator=c.validate);return dict(status='passed_scoped',prior_bridge=previous_bridge,physical_payload_exact=True,identity=c.identity,resources=g.identity)
