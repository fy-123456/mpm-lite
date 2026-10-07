"""Owned proposals for C's whole-step transaction. B has no commit method."""
from dataclasses import dataclass
import copy
import secrets
import numpy as np
from .material import array_digest, digest


@dataclass(frozen=True)
class MaterialProposal:
    session: str
    token: int
    base_revision: int
    displacement_sha256: str
    source_sha256: str
    space_sha256: str
    rule_sha256: str
    cache_sha256: str
    energy_J: float
    min_detF: float

    def values(self):
        return copy.deepcopy(self.__dict__)


class MaterialSession:
    """A rule locked for one physical step/Newton/AVF path.

    prepare is pure. attach copies values into a C-owned StateTrial after
    checking the C token/revision and current q. C alone validates and commits
    the entire state. Reusing B across physical steps requires a new session.
    Rules (including local fallback) may be selected only before this session.
    """
    def __init__(self, operator, transaction):
        self.operator = operator
        self.transaction = transaction
        self.base_revision = transaction.revision
        self.session = secrets.token_hex(16)
        self.cache = operator.signature
        self.closed = False

    def _check(self, trial):
        if self.closed or self.transaction.revision != self.base_revision:
            raise ValueError('closed or stale material session')
        # C remains the token owner; this checks exact object ownership too.
        self.transaction._check_trial(trial)
        self.operator.check_identity(cache_sha256=self.cache)

    def prepare(self, trial):
        self._check(trial)
        result = self.operator.evaluate_full(trial.state.q)
        proposal = MaterialProposal(self.session, trial.token, trial.base_revision,
                    array_digest(trial.state.q), self.operator.source.signature,
                    self.operator.space.signature, self.operator.rule.signature,
                    self.cache, result['energy_J'], result['min_detF'])
        return proposal, result

    def attach(self, trial, proposal):
        self._check(trial)
        if not isinstance(proposal, MaterialProposal) or (proposal.session, proposal.token, proposal.base_revision) != (self.session, trial.token, self.base_revision):
            raise ValueError('foreign or stale material proposal')
        if proposal.displacement_sha256 != array_digest(trial.state.q):
            raise ValueError('displacement changed after material preparation')
        op = self.operator
        if (proposal.source_sha256, proposal.space_sha256, proposal.rule_sha256, proposal.cache_sha256) != (op.source.signature, op.space.signature, op.rule.signature, self.cache):
            raise ValueError('source/space/rule/cache mismatch')
        if not np.isfinite([proposal.energy_J, proposal.min_detF]).all() or proposal.min_detF <= op.min_detF:
            raise ValueError('invalid material proposal values')
        trial.state.child_states['B_material'] = proposal.values()
        return proposal.values()

    def reject(self, trial):
        self._check(trial)
        return self.transaction.rollback(trial)

    def switch_rule(self, _rule):
        raise ValueError('material rule cannot change during a physical step')

    def close(self):
        self.closed = True


def validate_material_child(state, operator):
    """C callback checks an attached proposal against the complete owned q."""
    child = state.child_states.get('B_material')
    if child is None:
        raise ValueError('missing B material preparation')
    expected = dict(displacement_sha256=array_digest(state.q), source_sha256=operator.source.signature,
                    space_sha256=operator.space.signature, rule_sha256=operator.rule.signature,
                    cache_sha256=operator.signature)
    if any(child.get(k) != value for k, value in expected.items()):
        raise ValueError('material child no longer matches the whole-step state')
    if not np.isfinite([child.get('energy_J', np.nan), child.get('min_detF', np.nan)]).all() or child['min_detF'] <= operator.min_detF:
        raise ValueError('invalid prepared material response')
    operator.check_identity(cache_sha256=child['cache_sha256'])
    return True
