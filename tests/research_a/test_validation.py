import unittest
from engine.aniso_phase1.research_a.validation import nonlinear_checks


class NonlinearGateTests(unittest.TestCase):
    def setUp(self):
        self.gates = dict(energy_derivative_absolute=1e-8, tangent_relative=5e-4,
                          rotation_energy_absolute=1e-9, rotation_force_relative=1e-6,
                          derivative_relative_diagnostic=2e-4)
        self.row = dict(epsilon=3e-7, energy_derivative_absolute_error=2.6e-10,
                        energy_derivative_relative_error=2.4e-4, tangent_relative_error=4e-9)

    def test_small_absolute_error_does_not_hide_relative_diagnostic(self):
        result = nonlinear_checks([self.row], 1e-15, 1e-11, self.gates)
        self.assertTrue(result["engineering_passed"])
        self.assertFalse(result["relative_derivative_diagnostic_passed"])

    def test_each_physical_gate_can_reject(self):
        for key, value in (("energy_derivative_absolute_error", 2e-8), ("tangent_relative_error", 1e-3)):
            bad = dict(self.row, **{key: value})
            self.assertFalse(nonlinear_checks([bad], 0., 0., self.gates)["engineering_passed"])
        self.assertFalse(nonlinear_checks([self.row], 2e-9, 0., self.gates)["engineering_passed"])
        self.assertFalse(nonlinear_checks([self.row], 0., 2e-6, self.gates)["engineering_passed"])

    def test_missing_and_nonfinite_evidence_rejected(self):
        with self.assertRaises(ValueError):
            nonlinear_checks([], 0., 0., self.gates)
        with self.assertRaises(ValueError):
            nonlinear_checks([dict(self.row, energy_derivative_absolute_error=float("nan"))], 0., 0., self.gates)
