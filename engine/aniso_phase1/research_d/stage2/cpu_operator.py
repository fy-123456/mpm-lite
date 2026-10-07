"""Explicit order-6 CPU reference, wrapping the unchanged parent operator."""
import numpy as np
from ...research_b.tensor import TensorRule,TensorMaterialOperator
from .contracts import OperatorContract,PARENT_SHA256,array_digest,digest

class CPUOperator:
    def __init__(self,space,*,order=6):
        self.space=space;self.rule=TensorRule.uniform(space.edges,order)
        self.operator=TensorMaterialOperator(space,self.rule)
    def evaluate(self,q,direction=None):
        s=self.space
        r=self.operator.evaluate(s.expand(q),None if direction is None else s.direction_coefficients(direction))
        out=dict(r,energy_J=r['U'],full_force=r['force'],force=s.restrict(r['force']))
        if direction is not None:out.update(full_tangent_action=r['tangent_action'],tangent_action=s.restrict(r['tangent_action']))
        return out

def contract_for(operator,q,*,extension_sources,device='cpu'):
    s=operator.space
    return OperatorContract(parent_bundle_sha256=PARENT_SHA256,space_sha256=s.signature,
        material_rule_sha256=operator.rule.signature,
        mass_rule_sha256=digest(dict(space=s.signature,order=5,density_kg_m3=1.,kind='full consistent point inertia; all cross terms')),
        material_sha256=digest(dict(model='Hencky + bilateral quadratic fiber',mu=s.params.mu,lam=s.params.lam,kf=s.params.k_f,A=array_digest(s.A),Ks=array_digest(s.Ks))),
        boundary_sha256=array_digest(s.lift),state_sha256=array_digest(q),extension_source_sha256=digest(extension_sources),
        q_shape=s.q_shape,full_shape=(s.ndof,3),device=device).validate()
