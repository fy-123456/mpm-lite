import unittest
from unittest.mock import patch
import numpy as np
from tests.research_a.helpers import small_problem
from engine.aniso_phase1.research_a.adaptive import enrich
from engine.aniso_phase1.research_a.support_family import support_family
from engine.aniso_phase1.research_a.estimators import correction_score, ranked_indices
from engine.aniso_phase1.research_a.export_adapter import FixedSpace


class SpaceInvariantsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = small_problem()
        # Block file/reference access throughout actual selection and balancing.
        with patch("builtins.open", side_effect=AssertionError("Selection attempted file access")), \
             patch("io.open", side_effect=AssertionError("Selection attempted Path access")), \
             patch("numpy.load", side_effect=AssertionError("Selection attempted archive access")):
            cls.result = enrich(cls.problem, support_family("v22-overlap"), rounds=2,
                                patches_per_round=2, budget=12)
        cls.space = FixedSpace(cls.problem, cls.result["W"])
        field = cls.result["field"]
        cls.q = np.vstack((cls.problem.Q.T@(field["y"]-cls.problem.lift), field["local_coefficients"]))
        cls.points = [np.array([.28, .49, .73]), np.array([.41, .58]), np.array([.4, .57])]

    def test_actual_rebalance_has_no_mass_shift_zero_modes_or_energy_growth(self):
        previous = np.inf
        for r in self.result["records"]:
            self.assertFalse(r["reference_field_used"])
            self.assertLessEqual(r["materials"]["F45"]["energy_J"], previous+1e-10)
            previous = r["materials"]["F45"]["energy_J"]
            for m in r["materials"].values():
                self.assertFalse(m["mass_included"])
                self.assertEqual(m["stiffness_shift"], 0.)
                self.assertGreater(m["schur_min"], 0.)
        self.assertLess(self.result["basis_reconstruction_relative"], 1e-7)
        errors = self.problem.invariant_errors(self.result["W"])
        self.assertLess(max(errors.values()), 1e-7)

    def test_reference_free_score_positive_stable_and_validation(self):
        rhs = np.array([1., -2.])
        w = rhs/2
        self.assertAlmostEqual(correction_score(rhs, w, "energy"), 1.25)
        self.assertAlmostEqual(correction_score(rhs, w, "stress-correction", lambda v: 4*v), 5.)
        self.assertEqual(ranked_indices([1., 1., 0.]).tolist(), [0, 1, 2])
        with self.assertRaises(ValueError):
            correction_score(rhs, np.array([np.nan, 1.]))

    def test_displacement_mapping_direction_and_force_adjoint(self):
        space = self.space
        rng = np.random.default_rng(10)
        v = rng.normal(scale=1e-4, size=space.shape)
        dx, dF = space.jvp(v, self.points)
        x1, F1 = space.evaluate(self.q+v, self.points)
        x0, F0 = space.evaluate(self.q, self.points)
        np.testing.assert_allclose(x1-x0, dx, atol=2e-9)
        np.testing.assert_allclose(F1-F0, dF, atol=2e-8)
        fx, fF = rng.normal(size=dx.shape), rng.normal(size=dF.shape)
        lhs = np.sum(fx*dx)+np.sum(fF*dF)
        rhs = np.sum(space.vjp(fx, fF, self.points)*v)
        self.assertLess(abs(lhs-rhs), 1e-8)
        # Check exactly one identity matrix against the legacy total-position map.
        from engine.aniso_phase1.tensor_metrics import evaluate_gradient
        legacy = space.potential.T@space.total_coefficients(self.q)
        exact = evaluate_gradient((self.problem.edges, 2, legacy), self.points)
        np.testing.assert_allclose(F0, exact, atol=2e-8)

    def test_nonlinear_force_tangent_and_quadratic_bending(self):
        rng = np.random.default_rng(4)
        v = rng.normal(size=self.space.shape)
        v /= np.linalg.norm(v)
        base = self.space.response(self.q, v)
        eps = 1e-6
        plus, minus = self.space.response(self.q+eps*v), self.space.response(self.q-eps*v)
        self.assertLess(abs((plus["energy_J"]-minus["energy_J"])/(2*eps)-np.sum(base["force"]*v)), 1e-8)
        relative = np.linalg.norm((plus["force"]-minus["force"])/(2*eps)-base["tangent_action"])/np.linalg.norm(base["tangent_action"])
        self.assertLess(relative, 5e-4)
        self.assertLess(self.problem.invariant_errors(self.space.W)["quadratic_polynomial_error"], 1e-8)
