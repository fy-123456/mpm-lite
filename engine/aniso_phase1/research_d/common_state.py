"""C-compatible semantics for the frozen A input; no time integrator is implied.

q and velocity contain EVERY carrier/local vector coefficient, including hard
boundary rows. q is displacement: x=X+Nq, F=I+grad_X(Nq). The package starts
from A's loaded static snapshot, not a stress-free dynamic initial condition.
An initial time of zero is a clock label only, not a prescribed loading phase.

The transaction owns no external B/E resources. Child states and transfer
packets must be owned value trees; external objects require explicit adapters.
Validation callbacks must not perform external side effects. They receive a
copy, and only the transaction's last assignment changes committed history.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import json
import threading
from typing import Callable

import numpy as np


def _shape(q_shape):
    shape = tuple(q_shape)
    if (len(shape) != 2 or any(isinstance(v, (bool, np.bool_))
            or not isinstance(v, (int, np.integer)) for v in shape)
            or shape[1] != 3 or shape[0] < 1):
        raise ValueError('full displacement shape must be (positive scalar basis count, 3)')
    return tuple(int(v) for v in shape)


def kinetic_state_contract(space_id, q_shape, mass_rule_id=None):
    """Return explicit JSON-compatible semantics, without claiming validation.

    The caller must bind space_id and mass_rule_id to sealed content hashes in
    the enclosing package. This function specifies physics; it does not build
    or certify a mass matrix, initial dynamic equilibrium, or time integrator.
    """
    shape = _shape(q_shape)
    if not isinstance(space_id, str) or not space_id:
        raise ValueError('nonempty frozen space identity required')
    if mass_rule_id is not None and (not isinstance(mass_rule_id, str) or not mass_rule_id):
        raise ValueError('mass rule identity must be a nonempty string or None')
    return {
        'schema_version': 1, 'space_id': space_id, 'dtype': 'float64',
        'full_q_shape': list(shape), 'dof_order': 'scalar basis row then Cartesian x,y,z',
        'q_convention': 'full displacement coefficients, including constrained carriers',
        'kinematics': {'x': 'X + N q', 'F': 'I + grad_X(N q)',
                       'v': 'N velocity', 'C': 'grad_X(N velocity) F^-1'},
        'inertia': {
            'model': 'reference continuum point inertia', 'density_kg_m3': 1.0,
            'reference_volume': 'positive dV in the original physical box',
            'scalar_mass': 'N.T @ diag(rho*dV) @ N',
            'vector_mass': 'kron(scalar_mass, I3)',
            'kinetic_energy': '0.5 * sum(velocity * (scalar_mass @ velocity))',
            'carrier_local_cross_blocks': 'required before boundary reduction; no enrichment static condensation or diagonal lumping',
            'independent_APIC_microinertia': False,
            'mass_geometry_dependence': 'constant for the frozen reference basis',
            'mass_rule_id': mass_rule_id,
            'mass_rule_may_follow_material_compression': False,
        },
        'initial_state': {
            'source': 'A static snapshot with right-grip displacement [0.005, 0, 0] m',
            'stress_free': False, 'velocity': 'zero initialization by explicit convention',
            'time_zero_meaning': 'local clock label only; no physical loading phase implied',
            'dynamic_equilibrium_or_loading_protocol_certified': False,
        },
        'transaction': {
            'fields': ['step', 'time', 'q', 'velocity', 'predictor', 'transfer', 'child_states'],
            'trial': 'owned deep copy of one committed revision',
            'commit': 'validate all fields; publish one owned state; invalidate sibling trials',
            'rollback': 'discard owned trial; committed state remains unchanged',
            'callbacks': 'receive copies; no external side effects',
            'stale_or_failed_trial': 'rejected; never reusable',
            'time_step_rule': 'step increments by one and local time strictly increases',
        },
        'scope': {
            'state_interface_only': True,
            'time_integrator_implemented': False,
            'dynamic_acceptance': False,
            'Eulerian_grid_force_solve': False,
            'spatial_accuracy_acceptance': False,
        },
    }


def _tree(value):
    """Validate owned numerical history and turn it into JSON-compatible data."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, np.bool_):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError('nonfinite history value')
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError('history arrays must contain finite real numeric values')
        return value.tolist()
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError('history dictionary keys must be strings')
        return {k: _tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tree(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        # C GridPacket is an owned dataclass of arrays. Its decoded JSON tree
        # needs an explicit GridPacket adapter before calling packet methods.
        return {f.name: _tree(getattr(value, f.name)) for f in fields(value)}
    raise TypeError('history must be a value tree, not an external resource')


def _real_array(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf' or not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite real coefficients')
    return np.array(array, dtype=np.float64, copy=True)


@dataclass
class CommonState:
    q: np.ndarray
    velocity: np.ndarray
    time: float = 0.0
    step: int = 0
    predictor: np.ndarray | None = None
    transfer: object = None
    child_states: dict = field(default_factory=dict)

    def clone(self):
        return copy.deepcopy(self)

    def to_dict(self):
        """Portable values; dataclass transfer packets become explicit fields."""
        return _tree(self)

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                              separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class StateTrial:
    token: int
    base_revision: int
    state: CommonState


class StateTransaction:
    """Copy-on-trial, all-field atomic history owned by a single transaction.

    ``begin_trial``/``commit`` are also exposed as ``trial``/``accept`` for the
    existing D protocol. A rejected commit consumes its token. ``snapshot``
    and ``committed`` always return copies, never mutable committed storage.
    A validator should raise ValueError on invalid detF, boundary conditions,
    or child physics. Returning False is also treated as a rejection.
    """
    def __init__(self, initial: CommonState, validator: Callable | None = None):
        if not isinstance(initial, CommonState):
            raise TypeError('CommonState initial value required')
        self._q_shape = _shape(np.shape(initial.q))
        if validator is not None and not callable(validator):
            raise TypeError('validator must be callable')
        self._validator = validator
        self._lock = threading.RLock()
        self._revision = 0
        self._next_token = 0
        self._active = {}
        self._committed = self._validated_copy(initial)

    def _validated_copy(self, state):
        if not isinstance(state, CommonState):
            raise TypeError('trial state must remain a CommonState')
        candidate = state.clone()
        candidate.q = _real_array(candidate.q, 'q')
        candidate.velocity = _real_array(candidate.velocity, 'velocity')
        if candidate.q.shape != self._q_shape or candidate.velocity.shape != self._q_shape:
            raise ValueError('full q/velocity shape mismatch')
        if candidate.predictor is not None:
            candidate.predictor = _real_array(candidate.predictor, 'predictor')
            if candidate.predictor.shape != self._q_shape:
                raise ValueError('predictor shape mismatch')
        if isinstance(candidate.step, (bool, np.bool_)) or not isinstance(candidate.step, (int, np.integer)) or candidate.step < 0:
            raise ValueError('step must be a nonnegative integer')
        if isinstance(candidate.time, (bool, np.bool_)) or not isinstance(candidate.time, (float, int, np.floating, np.integer)) or not np.isfinite(candidate.time) or candidate.time < 0:
            raise ValueError('time must be finite and nonnegative')
        if not isinstance(candidate.child_states, dict):
            raise TypeError('child states must be a dictionary')
        _tree(candidate)
        if self._validator is not None:
            result = self._validator(candidate.clone())
            if result is False or isinstance(result, np.bool_) and not result:
                raise ValueError('physical state validator rejected trial')
        return candidate

    @property
    def revision(self):
        with self._lock:
            return self._revision

    @property
    def committed(self):
        return self.snapshot()

    def snapshot(self):
        with self._lock:
            return self._committed.clone()

    def begin_trial(self):
        with self._lock:
            self._next_token += 1
            trial = StateTrial(self._next_token, self._revision, self._committed.clone())
            self._active[trial.token] = trial
            return trial

    def trial(self, q=None):
        trial = self.begin_trial()
        if q is not None:
            trial.state.q = np.array(q, copy=True)
        return trial

    def _check_trial(self, trial):
        if (not isinstance(trial, StateTrial) or trial.base_revision != self._revision
                or self._active.get(trial.token) is not trial):
            raise ValueError('foreign, stale, rolled-back, or failed trial token')

    def commit(self, trial):
        with self._lock:
            self._check_trial(trial)
            try:
                candidate = self._validated_copy(trial.state)
                if candidate.step != self._committed.step + 1 or candidate.time <= self._committed.time:
                    raise ValueError('accepted time step must increment step and strictly advance time')
                # Recheck after callback in case a callback re-entered this
                # transaction; no stale candidate can overwrite a newer state.
                self._check_trial(trial)
            except Exception:
                self._active.pop(trial.token, None)
                raise
            self._committed = candidate
            self._revision += 1
            self._active.clear()
            return candidate.clone()

    accept = commit

    def rollback(self, trial):
        with self._lock:
            self._check_trial(trial)
            del self._active[trial.token]
            return self._committed.clone()
