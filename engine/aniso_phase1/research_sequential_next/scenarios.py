"""Explicit material/peak variants on the same geometry, basis, Ks and full M5."""
from __future__ import annotations
import copy
import numpy as np
import scipy.linalg as la
from ..types import AnisotropicMaterialParams
from ..history_increment import material_tangent
from ..research_d.identity import digest
from ..research_d.stage2.contracts import array_digest
from ..research_sequential.condensation import Condensation


def coefficient_matrix(params):
    H=np.empty((9,9));F=np.eye(3)[None];A=params.A0[None]
    for j in range(9):
        d=np.zeros_like(F);d.ravel()[j]=1.;H[:,j]=material_tangent(F,A,d,params).ravel()
    return H.reshape(3,3,3,3).transpose(0,2,1,3).reshape(9,9)


def material_reduction(original,angle_degrees):
    s=original.parent;angle=float(angle_degrees)
    old_angle=float(np.degrees(np.arctan2(s.params.fiber_direction[1],s.params.fiber_direction[0]))%180)
    if abs(angle-old_angle)<1e-12:return original,dict(material_unchanged=True)
    theta=np.deg2rad(angle);p=AnisotropicMaterialParams(s.params.mu,s.params.lam,s.params.k_f,[np.cos(theta),np.sin(theta),0.])
    n=s.ndof;stabilization=np.zeros((n,3,n,3))
    for a in range(3):stabilization[:s.n,a,:s.n,a]=s.Ks
    Km=original.original_stiffness-stabilization.reshape(3*n,3*n)
    C0=coefficient_matrix(s.params);C1=coefficient_matrix(p);condition=float(np.linalg.cond(C0))
    if condition>1e8:raise ValueError('rest constitutive reassembly is ill-conditioned; explicit Gram assembly required')
    # K_{ia,jb}=sum_mn H_{am,bn} G_{mn,ij}. Recover all nine
    # geometry-only derivative Gram blocks from the sealed full rest matrix.
    # The invertible 9x9 system is checked; no physical inverse/pseudoinverse.
    blocks=Km.reshape(n,3,n,3).transpose(1,3,0,2).reshape(9,n*n)
    grams=la.solve(C0,blocks);reconstruction=float(la.norm(C0@grams-blocks)/max(la.norm(blocks),1e-30))
    if reconstruction>2e-10:raise ValueError('old material rest operator reconstruction failed')
    K=(C1@grams).reshape(3,3,n,n).transpose(2,0,3,1).reshape(3*n,3*n)+stabilization.reshape(3*n,3*n)
    space=copy.copy(s);space.params=p;space.A=p.A0.copy();space.A.setflags(write=False)
    space.metadata=copy.deepcopy(s.metadata);space.metadata['material']=dict(mu=p.mu,lam=p.lam,k_f=p.k_f,fiber_angle_degrees=angle)
    space.signature=digest(dict(parent=s.signature,material=space.metadata['material'],basis='identical to parent',Ks=array_digest(s.Ks)))
    result=Condensation(space,original.original_mass,K)
    for name in ('P','offset','M'):
        if not np.array_equal(getattr(result,name),getattr(original,name)):raise ValueError('material-only scenario changed recovery or mass: '+name)
    return result,dict(material_unchanged=False,angle_degrees=angle,constitutive_reassembly_condition=condition,
        original_material_block_relative=reconstruction,mass_sha256=array_digest(result.M),
        complete_mass_and_recovery_unchanged=True,rest_stiffness='same derivative Gram blocks, new exact material tensor')


class ScaledBoundary:
    def __init__(self,parent,peak):
        if not np.isfinite(peak) or peak<=0:raise ValueError('finite positive peak displacement required')
        if parent.hold is not None:raise ValueError('peak variant requires the complete cyclic boundary')
        self.parent=parent;self.hold=None;self.unit=parent.unit;self.peak=float(peak);self.factor=self.peak/.005
        self.signature=digest(dict(parent=parent.signature,peak_m=self.peak,rule='scaled original cosine v1'))
    def lift(self,t):return self.factor*self.parent.lift(t)
    def speed(self,t):return self.factor*self.parent.speed(t)
    def validate(self,state,tol=1e-8):
        if any(x.shape!=self.unit.shape or not np.isfinite(x).all() for x in [state.q,state.velocity]):raise ValueError('invalid scaled-boundary state')
        ids=self.parent.reduction.fixed
        errors=[float(np.max(abs((x-target)[ids]))) for x,target in [(state.q,self.lift(state.time)),(state.velocity,self.speed(state.time))]]
        if max(errors)>tol:raise ValueError('incompatible scaled boundary phase')
        return errors


def install_peak(model,peak):
    if abs(float(peak)-.005)<1e-14:return model
    model.boundary=ScaledBoundary(model.boundary,peak)
    model.identity=dict(model.identity,boundary=model.boundary.signature);model.signature=digest(model.identity)
    return model
