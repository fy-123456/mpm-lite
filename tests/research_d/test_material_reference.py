"""Small physical fixtures for the common-input quadrature audit."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.research_d.integration.material_reference import audit_material_reference
from engine.aniso_phase1.types import AnisotropicMaterialParams
from engine.aniso_phase1.tensor_reference import coordinates
from engine.aniso_phase1.research_b.tensor import TensorMaterialOperator


class SmallSpace:
    """One Q2 tensor cell with identity displacement reconstruction."""
    def __init__(self):
        self.edges = (np.array([0., 1.]),)*3
        self.p = 2
        self.reference = np.array(np.meshgrid(*coordinates(self.edges, self.p), indexing='ij')).reshape(3, -1).T
        self.ndof = self.n = len(self.reference)
        self.Ks = np.zeros((self.n, self.n))
        self.params = AnisotropicMaterialParams(10., 20., 200.)
        a = np.array([1., 1., 0.])/np.sqrt(2.)
        self.A = np.outer(a, a)
        self.signature = 'small-q2-material-reference-fixture-v1'
        self.mass_rule = {'order': 4, 'identity': 'unchanged-test-mass-rule'}

    def nodes(self, q):
        return np.asarray(q)

    def adjoint(self, f):
        return np.asarray(f)


def inputs():
    s = SmallSpace()
    flat = np.arange(3*s.ndof).reshape(-1, 3)
    left = flat[s.reference[:, 0] == 0.].ravel()
    right = flat[s.reference[:, 0] == 1.].ravel()
    fixed = np.concatenate((left, right))
    free = np.setdiff1d(flat.ravel(), fixed)
    protocol = dict(schema_version=1, orders=dict(wiring=5, candidate=6, check=7),
        dof_groups=dict(free=free.tolist(), fixed=fixed.tolist(),
            grip_left=left.tolist(), grip_right=right.tolist()),
        budgets=dict(energy=dict(atol=1e-10, rtol=1e-4, scale=.01),
            force=dict(atol=1e-8, rtol=1e-4, scale=1.),
            tangent=dict(atol=1e-7, rtol=1e-3, scale=1.)))
    q = .002*s.reference*np.array([1., -.1, .3])
    d1 = .001*s.reference*np.array([.4, .2, -.3])
    d2 = .001*s.reference[:, ::-1].copy()
    return s, {'rest': np.zeros_like(q), 'affine': q}, {'stretch': d1, 'mixed': d2}, protocol


class MaterialReferenceChecks(unittest.TestCase):
    def test_affine_certification_and_bounded_scope_are_json_serializable(self):
        s, states, directions, protocol = inputs()
        mass_before = copy.deepcopy(s.mass_rule)
        snapshots = {k: a.copy() for k, a in states.items()}
        result = audit_material_reference(s, states, directions, protocol)
        json.dumps(result, allow_nan=False)
        self.assertTrue(result['passed'])
        self.assertEqual(result['operator_evaluations'], 12)
        self.assertEqual(result['material_calls_total'], 4*(5**3+6**3+7**3))
        self.assertFalse(result['scope']['future_states_certified'])
        self.assertFalse(result['scope']['space_accuracy_certified'])
        self.assertFalse(result['scope']['mass_rule_changed'])
        self.assertEqual(mass_before, s.mass_rule)
        for name, value in snapshots.items():
            np.testing.assert_array_equal(states[name], value)
        metrics = result['states']['rest']['comparisons']['candidate_vs_check']['metrics']
        self.assertTrue(metrics['material_U']['near_zero_reference'])
        self.assertIn('force/grip_left_net_xyz', metrics)
        self.assertEqual(result['states']['affine']['orders']['candidate']['stabilization_energy'], 0.)

    def test_candidate_error_fails_without_hiding_in_total_stabilization(self):
        s, states, directions, protocol = inputs()
        original = TensorMaterialOperator.evaluate
        def perturbed(operator, q, direction=None):
            result = original(operator, q, direction)
            if operator.rule.orders == (6,):
                result['material_force'] = result['material_force'].copy()
                result['material_force'][0, 0] += .1
            return result
        with patch.object(TensorMaterialOperator, 'evaluate', perturbed):
            result = audit_material_reference(s, states, directions, protocol)
        self.assertFalse(result['passed'])
        row = result['states']['affine']['comparisons']['candidate_vs_check']
        self.assertFalse(row['metrics']['material_force/fixed']['passed'])

    def test_order_five_is_a_diagnostic_and_does_not_veto_candidate(self):
        s, states, directions, protocol = inputs()
        original = TensorMaterialOperator.evaluate
        def perturbed(operator, q, direction=None):
            result = original(operator, q, direction)
            if operator.rule.orders == (5,):
                result['material_U'] += 1.
                result['U'] += 1.
            return result
        with patch.object(TensorMaterialOperator, 'evaluate', perturbed):
            result = audit_material_reference(s, states, directions, protocol)
        self.assertTrue(result['passed'])
        self.assertFalse(result['states']['affine']['comparisons']['wiring_vs_check']['passed'])
        self.assertFalse(result['states']['affine']['comparisons']['wiring_vs_check']['participates_in_acceptance'])

    def test_illegal_protocol_and_inputs_fail_before_evaluation(self):
        s, states, directions, protocol = inputs()
        bad_protocols = []
        for mutate in (
            lambda p: p['orders'].update(candidate=4),
            lambda p: p['budgets']['force'].update(scale=float('nan')),
            lambda p: p['dof_groups']['free'].append(p['dof_groups']['fixed'][0]),
            lambda p: p['dof_groups'].update(grip_left=p['dof_groups']['free']),
        ):
            p = copy.deepcopy(protocol); mutate(p); bad_protocols.append(p)
        with patch.object(TensorMaterialOperator, 'evaluate', side_effect=AssertionError('must not run')):
            for bad in bad_protocols:
                with self.assertRaises(ValueError): audit_material_reference(s, states, directions, bad)
            with self.assertRaises(ValueError): audit_material_reference(s, {'one': states['rest']}, directions, protocol)
            bad = dict(states); bad['rest'] = np.full_like(states['rest'], np.nan)
            with self.assertRaises(ValueError): audit_material_reference(s, bad, directions, protocol)
            bad_directions = dict(directions); bad_directions['stretch'] = np.zeros_like(states['rest'])
            with self.assertRaises(ValueError): audit_material_reference(s, states, bad_directions, protocol)

    def test_nonpositive_jacobian_is_not_reported_as_an_accuracy_pass(self):
        s, states, directions, protocol = inputs()
        states['inverted'] = -2*s.reference
        with self.assertRaises(ValueError):
            audit_material_reference(s, states, directions, protocol)


if __name__ == '__main__':
    unittest.main()
