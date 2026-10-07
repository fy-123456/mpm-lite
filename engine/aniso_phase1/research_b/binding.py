"""Material-origin guard at the C/B integration boundary.

The numerical candidates stay unchanged. This factory confines between-step
proposals to rules derived from one authoritative reference-material source.
The low-level MaterialSession by itself only enforces transaction semantics.
"""
from .operator import LinearMaterialOperator, MaterialSession
from .rules import compress, local_fallback


class BoundMaterialFamily:
    def __init__(self, full_rule, gradient_map, params, space_signature, ndof):
        self.source_rule = full_rule
        self.source_signature = full_rule.signature
        self._map, self._params, self._space, self._ndof = gradient_map, params, space_signature, ndof
        self._members = []
        self.full = self._bind(full_rule)

    def _bind(self, rule):
        op = LinearMaterialOperator(rule, self._map, self._params, self._space, self._ndof)
        self._members.append((op, op.signature, op.rule.signature))
        return op

    def require_member(self, op):
        if not any(op is member and op.signature == signature and op.rule.signature == rule_signature
                   for member, signature, rule_signature in self._members):
            raise ValueError('proposal must be constructed from the same authoritative material family')

    def compressed(self, groups_per_partition, method='representatives'):
        return self._bind(compress(self.source_rule, groups_per_partition, method))

    def fallback(self, operator, partitions):
        self.require_member(operator)
        return self._bind(local_fallback(self.source_rule, operator.rule, partitions))

    def session(self, operator, single_energy_budget, cumulative_energy_budget):
        self.require_member(operator)
        return BoundMaterialSession(self, operator, single_energy_budget, cumulative_energy_budget)


class BoundMaterialSession(MaterialSession):
    def __init__(self, family, operator, single_energy_budget, cumulative_energy_budget):
        family.require_member(operator)
        super().__init__(operator, single_energy_budget, cumulative_energy_budget)
        self.family = family

    def propose(self, candidate, q, direction, fallback_partitions=()):
        self.family.require_member(self.operator)
        self.family.require_member(candidate)
        return super().propose(candidate, q, direction, fallback_partitions)

    def commit_rule(self, proposal, q, *, accepted_step):
        if self._proposal is not None:
            self.family.require_member(self._proposal[1])
        return super().commit_rule(proposal, q, accepted_step=accepted_step)
