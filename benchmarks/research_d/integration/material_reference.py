"""Bounded material-quadrature certification for one frozen common space.

This audit changes only B's material rule. It never constructs or modifies a
mass rule. Order 6 is a *candidate* for the supplied states and directions;
passing against order 7 is neither exact integration nor a claim about future
states. The caller freezes and seals this protocol before running the audit.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import numpy as np

from engine.aniso_phase1.research_b.tensor import TensorMaterialOperator, TensorRule


_GROUPS = ('free', 'fixed', 'grip_left', 'grip_right')
_KINDS = ('energy', 'force', 'tangent')


def _array_hash(value):
    a = np.ascontiguousarray(value, dtype='<f8')
    h = hashlib.sha256()
    h.update(json.dumps(list(a.shape), separators=(',', ':')).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _states(values, ndof, label):
    if not isinstance(values, Mapping) or len(values) < 2:
        raise ValueError(f'{label} requires at least two named arrays')
    result = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f'{label} names must be nonempty strings')
        a = np.array(value, dtype=np.float64, copy=True)
        if a.shape != (ndof, 3) or not np.isfinite(a).all():
            raise ValueError(f'{label}/{name} must be finite with shape (ndof, 3)')
        if label == 'directions' and not np.any(a):
            raise ValueError('zero tangent direction cannot certify material integration')
        a.flags.writeable = False
        result[name] = a
    return result


def _validate(space, states, directions, protocol):
    if not isinstance(protocol, Mapping):
        raise ValueError('explicit frozen material-reference protocol required')
    p = copy.deepcopy(dict(protocol))
    if p.get('schema_version') != 1 or p.get('orders') != dict(wiring=5, candidate=6, check=7):
        raise ValueError('schema 1 requires wiring/candidate/check orders 5/6/7')
    if not isinstance(getattr(space, 'signature', None), str) or not space.signature:
        raise ValueError('space requires an explicit signature')
    ndof = getattr(space, 'ndof', None)
    if not isinstance(ndof, (int, np.integer)) or isinstance(ndof, bool) or ndof < 1:
        raise ValueError('space.ndof must be a positive scalar-basis count')
    n = getattr(space, 'n', None)
    if not isinstance(n, (int, np.integer)) or not 0 < n <= ndof:
        raise ValueError('space.n must identify the carrier scalar basis count')
    if np.shape(space.reference) != (ndof, 3) or not np.isfinite(space.reference).all():
        raise ValueError('finite carrier reference coordinates required')
    if len(space.edges) != 3 or any(np.ndim(e) != 1 or len(e) < 2 or
            not np.isfinite(e).all() or not np.all(np.diff(e) > 0) for e in space.edges):
        raise ValueError('three finite, strictly increasing material-cell axes required')
    if np.shape(space.A) != (3, 3) or not np.isfinite(space.A).all():
        raise ValueError('finite homogeneous fiber tensor required')
    groups_in = p.get('dof_groups', {})
    if set(groups_in) != set(_GROUPS):
        raise ValueError('explicit free/fixed/left-grip/right-grip groups required')
    groups = {'all': np.arange(3*ndof, dtype=np.int64)}
    for name in _GROUPS:
        raw = np.asarray(groups_in[name])
        if raw.ndim != 1 or raw.dtype.kind not in 'iu' or len(raw) == 0:
            raise ValueError(f'{name} must be a nonempty integer flat-DOF index array')
        a = raw.astype(np.int64)
        if np.any(a < 0) or np.any(a >= 3*ndof) or len(np.unique(a)) != len(a):
            raise ValueError(f'invalid or duplicate {name} DOF indices')
        groups[name] = a
    if np.intersect1d(groups['free'], groups['fixed']).size or not np.array_equal(
            np.sort(np.concatenate((groups['free'], groups['fixed']))), groups['all']):
        raise ValueError('free and fixed groups must partition every vector DOF')
    if np.intersect1d(groups['grip_left'], groups['grip_right']).size or any(
            not np.isin(groups[g], groups['fixed']).all() for g in ('grip_left', 'grip_right')):
        raise ValueError('grips must be disjoint subsets of fixed DOFs')
    budgets = p.get('budgets', {})
    if set(budgets) != set(_KINDS):
        raise ValueError('explicit energy, force and tangent budgets required')
    for name in _KINDS:
        budget = budgets[name]
        if not isinstance(budget, Mapping) or set(budget) != {'atol', 'rtol', 'scale'}:
            raise ValueError(f'{name} needs atol, rtol and diagnostic scale')
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and
                np.isfinite(v) and v > 0 for v in budget.values()):
            raise ValueError('budgets and diagnostic scales must be finite and positive')
    # Reject non-JSON protocol metadata rather than silently dropping provenance.
    try:
        json.dumps(p, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('protocol must be finite JSON data') from exc
    return p, groups, _states(states, ndof, 'states'), _states(directions, ndof, 'directions')


def _compare(candidate, check, budget):
    candidate, check = np.asarray(candidate), np.asarray(check)
    difference = float(np.linalg.norm((candidate-check).ravel()))
    check_norm = float(np.linalg.norm(check.ravel()))
    candidate_norm = float(np.linalg.norm(candidate.ravel()))
    limit = float(budget['atol'] + budget['rtol']*check_norm)
    return dict(candidate_norm=candidate_norm, check_norm=check_norm,
                difference_norm=difference, acceptance_limit=limit,
                norm_relative_error=None if check_norm == 0 else difference/check_norm,
                fixed_scale_relative_error=difference/budget['scale'],
                near_zero_reference=bool(check_norm <= budget['atol']/budget['rtol']),
                passed=bool(difference <= limit))


def _selected(value, groups):
    flat = np.asarray(value).ravel()
    result = {name: flat[index] for name, index in groups.items()}
    for name in ('grip_left', 'grip_right'):
        index = groups[name]
        result[name+'_net_xyz'] = np.array([flat[index[index % 3 == axis]].sum()
                                           for axis in range(3)])
    return result


def _summary(result, groups, *, tangent=False):
    keys = ('tangent_action', 'material_tangent_action') if tangent else ('force', 'material_force')
    data = {key: {group: dict(norm=float(np.linalg.norm(a)),
                              net_xyz=a.tolist() if group.endswith('_net_xyz') else None)
                  for group, a in _selected(result[key], groups).items()} for key in keys}
    if not tangent:
        data.update(total_energy=float(result['U']), material_energy=float(result['material_U']),
                    stabilization_energy=float(result['stabilization_U']),
                    minimum_detF=float(result['min_detF']),
                    material_calls_per_evaluation=int(result['material_calls']),
                    rule_signature=result['rule_signature'],
                    weak_moment_norm=float(np.linalg.norm(result['weak_moments'])))
    return data


def _pair(candidate, check, groups, budgets, *, tangent=False):
    metrics = {}
    if not tangent:
        for key in ('U', 'material_U', 'stabilization_U'):
            metrics[key] = _compare(candidate[key], check[key], budgets['energy'])
    keys = ('tangent_action', 'material_tangent_action') if tangent else ('force', 'material_force')
    for key in keys:
        left, right = _selected(candidate[key], groups), _selected(check[key], groups)
        for group in left:
            metrics[f'{key}/{group}'] = _compare(left[group], right[group], budgets['tangent' if tangent else 'force'])
    result = dict(metrics=metrics, passed=bool(all(m['passed'] for m in metrics.values())))
    if not tangent:
        # These are integrated weak moments, never same-F pointwise PK1 values.
        result['weak_moments_diagnostic'] = dict(
            check_norm=float(np.linalg.norm(check['weak_moments'])),
            difference_norm=float(np.linalg.norm(candidate['weak_moments']-check['weak_moments'])),
            acceptance_claim=False)
    return result


def audit_material_reference(space, states, directions, protocol):
    """Compare order 6 against 7 at every named state/direction pair.

    ``states`` and ``directions`` are maps to finite point-major displacement
    coefficient arrays shaped ``(space.ndof, 3)``; all directions are evaluated
    at all states. The space implements B TensorMaterialOperator's streaming
    mapping. Protocol schema 1 specifies orders, flat DOF groups, and budgets
    for energy, force and tangent (``atol``, ``rtol``, ``scale`` each).

    Every accepted metric satisfies ``norm(candidate-check) <= atol +
    rtol*norm(check)``. ``scale`` is diagnostic only and cannot relax acceptance.
    No file is written, no mass rule is changed, and returned data are JSON-safe.
    """
    protocol, groups, states, directions = _validate(space, states, directions, protocol)
    operators = {role: TensorMaterialOperator(space, TensorRule.uniform(space.edges, order))
                 for role, order in protocol['orders'].items()}
    records = {}
    calls, stabilization_invariant, all_passed = 0, True, True
    for state_name, state in states.items():
        values, tangents = {}, {}
        for role, operator in operators.items():
            tangents[role] = {}
            for direction_name, direction in directions.items():
                evaluated = operator.evaluate(state, direction)
                calls += int(evaluated['material_calls'])
                if role not in values:
                    values[role] = evaluated
                elif evaluated['U'] != values[role]['U'] or not np.array_equal(
                        evaluated['force'], values[role]['force']):
                    raise ValueError('energy/force changed when only the tangent direction changed')
                tangents[role][direction_name] = evaluated
        stable = len({float(v['stabilization_U']) for v in values.values()}) == 1
        stabilization_invariant &= stable
        pair_records = {}
        for label, role in (('candidate_vs_check', 'candidate'), ('wiring_vs_check', 'wiring')):
            pair = _pair(values[role], values['check'], groups, protocol['budgets'])
            pair['directions'] = {name: _pair(tangents[role][name], tangents['check'][name],
                groups, protocol['budgets'], tangent=True) for name in directions}
            pair['passed'] &= all(t['passed'] for t in pair['directions'].values())
            pair['participates_in_acceptance'] = role == 'candidate'
            pair_records[label] = pair
        accepted = pair_records['candidate_vs_check']['passed'] and stable
        all_passed &= accepted
        records[state_name] = dict(
            input_sha256=_array_hash(state),
            orders={role: dict(order=protocol['orders'][role],
                **_summary(value, groups),
                directions={name: _summary(v, groups, tangent=True)
                            for name, v in tangents[role].items()}) for role, value in values.items()},
            comparisons=pair_records, stabilization_invariant=bool(stable), passed=bool(accepted))
    output = dict(schema_version=1, space_signature=space.signature,
        protocol=protocol,
        protocol_sha256=hashlib.sha256(json.dumps(protocol, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest(),
        states=records,
        directions={name: dict(sha256=_array_hash(a), norm=float(np.linalg.norm(a)))
                    for name, a in directions.items()},
        operator_evaluations=len(states)*len(directions)*len(operators),
        material_calls_total=calls,
        stabilization_invariant=bool(stabilization_invariant),
        passed=bool(all_passed and stabilization_invariant),
        scope=dict(candidate_order=6, independent_check_order=7, wiring_control_order=5,
            state_names=list(states), direction_names=list(directions),
            acceptance='atol + rtol * check norm; fixed scale is diagnostic only',
            material_rule_only=True, mass_rule_changed=False,
            exact_integration_claim=False, future_states_certified=False,
            independent_constitutive_wiring_certified=False,
            pointwise_stress_accuracy_claim=False,
            space_accuracy_certified=False,
            explanation='Finite frozen-state/direction comparison using the existing B evaluator; '
                        'independent constitutive wiring and spatial reference accuracy require separate checks.'))
    # An explicit serialization gate prevents NaN/Inf or numpy values escaping.
    json.dumps(output, allow_nan=False)
    return output
