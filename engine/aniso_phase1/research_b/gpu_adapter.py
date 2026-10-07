"""Optional read-only D backend adapter for actual-direction sample rules.

General conditional A4 is deliberately rejected: D's current single-fiber
kernel does not evaluate that potential. No production/default GPU switch.
"""
import numpy as np
from .operator import LinearMaterialOperator
from .material import check_state


class RepresentativeGPUOperator(LinearMaterialOperator):
    def __init__(self, rule, gradient_map, params, space_signature, ndof, device='cuda:0'):
        super().__init__(rule, gradient_map, params, space_signature, ndof)
        A = rule.A2.reshape(-1, 9)
        if not np.allclose(rule.A4, np.einsum('pi,pj->pij', A, A), rtol=1e-10, atol=1e-12):
            raise ValueError('D backend currently supports actual single-direction samples only; conditional A4 requires a different kernel')
        from ..research_d.material import MaterialGPU
        self.backend = MaterialGPU(rule.A2, params, device=device)
        self.signature += '/D-CUDA-representative-v1'

    def evaluate(self, q, direction=None):
        F = check_state(self.deformation(q))
        dF = None if direction is None else np.einsum('pnj,ni->pij', self.D, self._check(direction))
        result = self.backend.evaluate(F, dF)
        if not all(np.isfinite(result[k]).all() for k in ('energy', 'stress')) or (
                direction is not None and not np.isfinite(result['tangent']).all()):
            raise ValueError('non-finite CUDA material response')
        out = dict(U=float(self.rule.weight @ result['energy']),PK1=result['stress'],
            force=np.einsum('p,pnj,pij->ni', self.rule.weight,self.D,result['stress']),
            min_detF=float(np.linalg.det(F).min()),material_calls=len(F),rule_signature=self.rule.signature)
        if direction is not None:
            out['tangent_action']=np.einsum('p,pnj,pij->ni',self.rule.weight,self.D,result['tangent'])
        return out

    def with_rule(self, rule):
        return RepresentativeGPUOperator(rule,self.gradient_map,self.params,self.space_signature,self.ndof,
                                         device=str(self.backend.device))
