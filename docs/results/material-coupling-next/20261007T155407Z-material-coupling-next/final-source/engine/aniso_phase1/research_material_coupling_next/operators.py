import numpy as np
from engine.aniso_phase1.research_formal_pressure_next.material import Material as ParentMaterial
from engine.aniso_phase1.research_formal_pressure_next.geometry import Geometry as ParentGeometry

class Material(ParentMaterial):
    def __init__(self,model,order,budget):super().__init__(model,order);self.budget=budget
    def evaluate(self,q,direction=None):
        self.budget.check();cached=direction is None and q.tobytes() in self.cache
        if cached:return super().evaluate(q,direction)
        label=f'q{self.order}:'+('jv' if direction is not None else 'response');start=self.budget.begin('material',label);ok=False
        try:
            out=super().evaluate(q,direction)
            out['material_force']=self.model.r.P.T@out['material_force']
            if direction is not None:out['material_tangent_action']=self.model.r.P.T@out['material_tangent_action']
            ok=True;return out
        finally:self.budget.end('material',label,start,ok)

class Geometry(ParentGeometry):
    def __init__(self,model,budget,order=9):self.budget=budget;super().__init__(model,order)
    def evaluate(self,q,direction=None):self.budget.check();return super().evaluate(q,direction)
    def _evaluate(self,q,direction=None):
        label='jv' if direction is not None else 'response';start=self.budget.begin('geometry',label);ok=False
        try:out=super()._evaluate(q,direction);ok=True;return out
        finally:self.budget.end('geometry',label,start,ok)


def compare(model,low,high,limit=.03):
    """Disjoint x-slabs; constant/centered-linear stress moments in Pa.

    Absolute floors: 1e-8 N assembled force, 1e-5 Pa normalized stress moments,
    1e-10 J material energy. These are registered physical floors, not adaptive.
    """
    r=model.r;s=model.space
    rel=lambda x,y,floor:float(np.linalg.norm(x-y)/max(np.linalg.norm(y),floor))
    force={name:rel(low['force'][ids],high['force'][ids],1e-8) for name,ids in [('free',r.free),('fixed',r.fixed)]}
    force['all']=rel(low['force'],high['force'],1e-8)
    edges=s.edges[0];vol=np.diff(edges)*.25**2
    centers=np.c_[(edges[:-1]+edges[1:])/2,np.full(len(vol),.5),np.full(len(vol),.5)]
    lengths=np.c_[np.diff(edges),np.full(len(vol),.25),np.full(len(vol),.25)]
    def normalized(raw):
        weak=raw['weak_moments'];const=weak[:,0]/vol[:,None,None]
        linear=(weak[:,1:]-centers[:,:,None,None]*weak[:,0,None])/(vol[:,None,None,None]*lengths[:,:,None,None])
        return const,linear
    lc,ll=normalized(low);hc,hl=normalized(high)
    ce=[rel(x,y,1e-5) for x,y in zip(lc,hc)];le=[rel(x,y,1e-5) for x,y in zip(ll,hl)]
    groups={name:dict(slabs=np.flatnonzero(mask).tolist(),constant_max=max(np.array(ce)[mask],default=0.),linear_max=max(np.array(le)[mask],default=0.))
        for name,mask in [('left_grip',centers[:,0]<.25),('left_free',(centers[:,0]>=.25)&(centers[:,0]<.5)),('right_free',(centers[:,0]>=.5)&(centers[:,0]<.75)),('right_grip',centers[:,0]>=.75)]}
    energy=abs(low['material_U']-high['material_U'])/max(abs(high['material_U']),1e-10)
    errors=[*force.values(),*ce,*le,float(energy)];tangent=None
    if 'tangent_action' in low and 'tangent_action' in high:
        tangent={k:rel(low[k],high[k],1e-8) for k in ['tangent_action','material_tangent_action']};errors+=list(tangent.values())
    return dict(accepted=all(x<=limit for x in errors),limit=limit,force=force,constant_stress_max=max(ce),linear_stress_max=max(le),
        slab_constant=ce,slab_linear=le,regions=groups,material_energy_relative=float(energy),energy_absolute=abs(low['U']-high['U']),tangent=tangent,
        minJ_low=low['min_detF'],minJ_high=high['min_detF'],max_error=max(errors))
