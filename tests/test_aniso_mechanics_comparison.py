import unittest
import numpy as np
from scipy import sparse
from engine.aniso_phase1.mechanics_comparison import Q1Static, load_and_trace, solve_metrics, mode_audit, field_errors
from benchmarks.aniso_mechanics_comparison import prototype_check


class MechanicsComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.space=Q1Static(17)

    def test_static_operator_and_trace_match_consistent_prototype(self):
        result=prototype_check(17, self.space)
        self.assertLess(result['matrix_relative'], 1e-12)
        self.assertLess(result['value_absolute'], 1e-12)
        self.assertLess(result['gradient_absolute'], 1e-10)

    def test_uniform_face_load_is_integrated_not_equal_per_node(self):
        trace, area=load_and_trace(self.space)
        self.assertAlmostEqual(area, .125**2)
        self.assertAlmostEqual(trace.sum(), 1.)
        weights=trace.reshape(-1,3)[:,1]
        nonzero=weights[weights>1e-12]
        self.assertAlmostEqual(nonzero.max()/nonzero.min(), 4.)
        other,_=load_and_trace(self.space, order=5)
        np.testing.assert_allclose(trace,other,atol=1e-14)

    def test_static_balance_mean_constraint_and_work_identity(self):
        u,r=solve_metrics(self.space)
        self.assertLess(r['force_balance_relative'],1e-9)
        self.assertLess(r['moment_balance_relative'],1e-9)
        self.assertLess(r['free_residual_relative'],1e-9)
        self.assertLess(r['clamp_max'],1e-12)
        self.assertLess(r['work_identity_relative'],1e-10)
        trace,_=load_and_trace(self.space)
        scaled=u*r['target_mean_displacement']/r['tip_displacement']
        self.assertAlmostEqual(trace@scaled,r['target_mean_displacement'],places=13)
        np.testing.assert_allclose((self.space.K@scaled)[self.space.free],
                                   (trace*r['force_at_target_mean'])[self.space.free],atol=1e-12)

    def test_common_physical_modes_and_self_comparison(self):
        audit=mode_audit(self.space)
        self.assertLess(audit['clamp_max'],1e-12)
        for name in ('stretch','shear','twist'):
            self.assertAlmostEqual(audit['modes'][name]['stiffness_ratio'],1.,places=10)
        self.assertGreater(audit['modes']['bend_y']['stiffness_ratio'],1.)
        u,_=solve_metrics(self.space)
        errors=field_errors(self.space,u,self.space,u)
        self.assertLess(errors['displacement_l2_relative'],1e-12)
        self.assertLess(errors['strain_energy_norm_relative'],1e-12)

    def test_unsupported_domain_and_outside_query_rejected(self):
        with self.assertRaises(ValueError):Q1Static(18)
        with self.assertRaises(ValueError):self.space.read(np.array([[0.,0.,0.]]))


if __name__=='__main__':unittest.main()
