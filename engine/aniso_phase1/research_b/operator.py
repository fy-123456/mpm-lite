"""Local schema-v1 displacement adapter and C-owned transaction handoff."""
from dataclasses import dataclass
import hashlib
import numpy as np
from .material import response
from .rules import readonly, digest_arrays, moment_audit


class LinearMaterialOperator:
    """q is displacement, point-major vector DOFs; positive force = dU/dq.

    gradient_map(X) returns (sample, scalar_dof, reference_component).
    The caller retains geometry/velocities/F histories. This hyperelastic B
    implementation owns only its frozen rule and a cache version.
    """
    def __init__(self, rule, gradient_map, params, space_signature, ndof):
        self.rule, self.params = rule, params
        self.gradient_map, self.space_signature = gradient_map, str(space_signature)
        self.ndof = int(ndof)
        self.D = readonly(gradient_map(rule.X))
        if self.D.shape != (len(rule.weight), self.ndof, 3) or not np.isfinite(self.D).all():
            raise ValueError('invalid space derivative map')
        self.signature = hashlib.sha256((rule.signature+self.space_signature+
            repr((params.mu, params.lam, params.k_f))+digest_arrays(self.D)).encode()).hexdigest()

    def with_rule(self, rule):
        return LinearMaterialOperator(rule, self.gradient_map, self.params, self.space_signature, self.ndof)

    def _check(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (self.ndof, 3) or not np.isfinite(q).all():
            raise ValueError('finite point-major displacement array required')
        return q

    def deformation(self, q):
        return np.eye(3) + np.einsum('pnj,ni->pij', self.D, self._check(q))

    def evaluate(self, q, direction=None):
        F = self.deformation(q)
        dF = None if direction is None else np.einsum('pnj,ni->pij', self.D, self._check(direction))
        values = response(F, self.rule.A2, self.rule.A4, self.params, dF)
        energy, P = values[:2]
        result = dict(U=float(self.rule.weight @ energy), PK1=P,
                      force=np.einsum('p,pnj,pij->ni', self.rule.weight, self.D, P),
                      min_detF=float(np.linalg.det(F).min()), material_calls=len(F),
                      rule_signature=self.rule.signature)
        if direction is not None:
            result['tangent_action'] = np.einsum('p,pnj,pij->ni', self.rule.weight, self.D, values[2])
        return result

    def energy(self, q):
        return self.evaluate(q)['U']

    def pk1(self, q):
        return self.evaluate(q)['PK1']

    def tangent_action(self, q, direction):
        return self.evaluate(q, direction)['tangent_action']

    def weak_moments(self, q):
        """Constant/linear stress probes, separately on every material region."""
        P = self.pk1(q)
        probes = np.column_stack((np.ones(len(P)), self.rule.X))
        return np.stack([np.einsum('p,pk,pij->kij', self.rule.weight[m], probes[m], P[m])
                         for region in np.unique(self.rule.partition)
                         for m in [self.rule.partition == region]])


@dataclass(frozen=True)
class RuleProposal:
    old_signature: str
    new_signature: str
    state_signature: str
    delta_U_rule: float
    force_change: float
    tangent_change: float
    stress_moment_change: float
    fallback_partitions: tuple
    admissible_budget: bool


class MaterialSession:
    """Explicit trial / commit / rollback; C controls accepted-step commits.

    Rules cannot change during a trial. Proposals are bound to old/new operator
    signatures and the current caller-owned displacement state. Rejected and
    accepted jumps are logged separately, and cumulative cost is sum(abs(dU)).
    """
    def __init__(self, operator, single_energy_budget, cumulative_energy_budget):
        if not np.isfinite([single_energy_budget, cumulative_energy_budget]).all() or min(single_energy_budget, cumulative_energy_budget) < 0:
            raise ValueError('finite nonnegative energy budgets required')
        self.operator = operator
        self.single_budget, self.total_budget = float(single_energy_budget), float(cumulative_energy_budget)
        self.cumulative_absolute = 0.
        self.cache_version = 0
        self.ledger = []
        self._active = False
        self._trial = None
        self._proposal = None

    def trial(self, q, direction=None):
        self._active = True
        self._trial = None
        # A failed later line-search trial must invalidate any earlier success.
        result = self.operator.evaluate(q, direction)
        self._trial = (digest_arrays(q), self.operator.signature)
        return result

    def commit(self, *, accepted_step):
        if not accepted_step or not self._active or self._trial is None:
            raise RuntimeError('C/caller must accept a valid trial before B commit')
        self.cache_version += 1
        self._active, self._trial, self._proposal = False, None, None

    def rollback(self):
        self._active, self._trial, self._proposal = False, None, None

    def propose(self, candidate, q, direction, fallback_partitions=()):
        if self._active:
            raise RuntimeError('sampling rule is frozen inside Newton/line search')
        if candidate.space_signature != self.operator.space_signature or candidate.ndof != self.operator.ndof or (
                candidate.params.mu, candidate.params.lam, candidate.params.k_f) != (
                self.operator.params.mu, self.operator.params.lam, self.operator.params.k_f):
            raise ValueError('rule proposal cannot change space or material law')
        audit = moment_audit(self.operator.rule, candidate.rule)
        volume = self.operator.rule.weight.sum()
        if max(audit['volume_absolute'], audit['first_moment_absolute']) > 1e-10*volume:
            raise ValueError('proposal changes material volume or first spatial moment')
        old, new = self.operator.evaluate(q, direction), candidate.evaluate(q, direction)
        delta = float(new['U']-old['U'])
        allowed = abs(delta) <= self.single_budget and self.cumulative_absolute+abs(delta) <= self.total_budget
        proposal = RuleProposal(self.operator.signature, candidate.signature, digest_arrays(q), delta,
            float(np.linalg.norm(new['force']-old['force'])),
            float(np.linalg.norm(new['tangent_action']-old['tangent_action'])),
            float(np.linalg.norm(candidate.weak_moments(q)-self.operator.weak_moments(q))),
            tuple(int(x) for x in fallback_partitions), bool(allowed))
        self._proposal = (proposal, candidate)
        return proposal

    def commit_rule(self, proposal, q, *, accepted_step):
        if self._active or not accepted_step or self._proposal is None or proposal is not self._proposal[0]:
            raise RuntimeError('only the caller may accept the current between-step proposal')
        if proposal.old_signature != self.operator.signature or proposal.state_signature != digest_arrays(q):
            raise RuntimeError('stale rule proposal or changed caller state')
        accepted = proposal.admissible_budget
        self.ledger.append(dict(old_signature=proposal.old_signature, new_signature=proposal.new_signature,
                                delta_U_rule=proposal.delta_U_rule, accepted=accepted,
                                reason='within frozen budgets' if accepted else 'rule energy budget exceeded'))
        if accepted:
            self.operator = self._proposal[1]
            self.cumulative_absolute += abs(proposal.delta_U_rule)
            self.cache_version += 1
        self._proposal = None
        return accepted

    def reject_rule(self):
        if self._proposal is not None:
            p = self._proposal[0]
            self.ledger.append(dict(old_signature=p.old_signature, new_signature=p.new_signature,
                                    delta_U_rule=p.delta_U_rule, accepted=False, reason='caller rejected'))
        self._proposal = None
