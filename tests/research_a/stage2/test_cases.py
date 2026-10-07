import unittest
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.research_a.stage2.cases import CaseSpec, registered_cases, supports
from engine.aniso_phase1.research_a.stage2.problem import Problem
from engine.aniso_phase1.research_a.baseline_adapter import Problem as ParentProblem
from engine.aniso_phase1.high_order_space import local_box, BoxElastic
from engine.aniso_phase1.research_a.stage2.reference import solve, pilot_edges, estimate
from engine.aniso_phase1.research_a.export_adapter import FixedSpace


class CasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = Problem.from_archive(CaseSpec(), 2, 'construction')

    def test_original_F45_energy_and_reaction_match_parent(self):
        p = self.base
        old = ParentProblem(p.edges, p.degree, p.A, p.carrier_X, p.Ks, p.params, p.fiber_tensor, p.initial_u)
        empty = np.empty((p.n, 0))
        f, r = p.equilibrate(empty); oldf, oldr = old.equilibrate(empty, labels=('F45',))
        np.testing.assert_allclose(f['u'], oldf['u'], atol=1e-9, rtol=1e-5)
        self.assertAlmostEqual(r['energy_J'], oldr['F45']['energy_J'], delta=1e-8)
        self.assertAlmostEqual(r['reaction_N'], oldr['F45']['reaction_N'], delta=1e-6)

    def test_real_case_materials_boundaries_and_patch(self):
        cases = registered_cases(); self.assertEqual(len(cases), 9)
        fingerprints = set()
        for name, spec in cases.items():
            with self.subTest(name=name):
                p = self.base if name == 'F45' else Problem.from_archive(spec, 2, 'construction')
                f, r = p.equilibrate(np.empty((p.n, 0)))
                self.assertTrue(r['static_passed'])
                self.assertLess(r['boundary_error_m'], 1e-8)
                self.assertLess(r['work_identity_relative'], 2e-5)
                errors = p.invariant_errors(np.empty((p.n, 0)))
                self.assertLess(errors['quadratic_polynomial_error'], 1e-8)
                self.assertLess(errors['patch_affine_residual'], 1e-8)
                fingerprints.add(spec.signature)
                if name == 'tall-shear':
                    self.assertAlmostEqual(spec.volume, .0703125)
                    self.assertFalse(p.construction['carrier_identity_reused'])
                    self.assertGreater(abs(r['reaction_vector_N'][1]), .001)
                    self.assertFalse(np.allclose(p.Ks, self.base.Ks))
                if name == 'bending':
                    X=p.carrier_X; right=X[:,0]>=.75
                    rotated=X[right]+p.lift[right]-spec.rotation_center
                    np.testing.assert_allclose(np.linalg.norm(rotated,axis=1),np.linalg.norm(X[right]-spec.rotation_center,axis=1),atol=1e-10)
                    self.assertGreater(abs(r['reaction_moment_Nm'][1]), 1e-5)
        self.assertEqual(len(fingerprints), 9)
        with self.assertRaises(ValueError): CaseSpec(loading='not-a-case')

    def test_all_supports_construct_in_physical_domain(self):
        for name in ('F45', 'angle15', 'tall-shear'):
            c=registered_cases()[name]; edges=pilot_edges(c)
            for family in ('v22-original','v22-overlap','wide-overlap','fiber-rect'):
                for patch in supports(c,family):
                    self.assertTrue(np.all(np.asarray(patch['lo'])>=np.asarray(c.box)[:,0]))
                    self.assertTrue(np.all(np.asarray(patch['hi'])<=np.asarray(c.box)[:,1]))
                    ids,op,meta=local_box(edges,2,patch,c.H)
                    self.assertEqual(len(ids),np.prod(op.free_shape))
                    self.assertEqual(tuple(meta['dirichlet_faces'][0]),(True,True))

    def test_reference_loading_and_full_domain(self):
        for name in ('F45','tall-shear','bending'):
            c=registered_cases()[name]; edges=pilot_edges(c); u,r=solve(c,edges,2)
            self.assertTrue(r['solve_passed'])
            self.assertLess(r['free_span_work_identity_relative'], 1e-5)
            self.assertEqual(len(u),estimate(edges,2)['nodes'])
            self.assertGreater(np.linalg.norm(r['reaction_vector_N']), 1e-4)

    def test_linear_rigid_modes_and_nonlinear_objectivity(self):
        p=self.base; op=BoxElastic(p.edges,p.degree,p.case.H); X=p.nodes()
        skew=np.array([[0,-.02,.01],[.02,0,-.01],[-.01,.01,0]])
        rigid=X@skew.T+[.01,-.02,.03]
        self.assertLess(la.norm(op.apply(rigid.T.ravel())),1e-8)
        space=FixedSpace(p,np.empty((p.n,0)))
        q=np.zeros(space.shape); Y=space.total_coefficients(q)
        R=la.expm(skew*10); Z=Y@R.T; Z[:len(p.carrier_X)]+=[.03,.02,-.01]
        a=space.potential.evaluate(Y,order=3);b=space.potential.evaluate(Z,order=3)
        self.assertLess(abs(a['U']-b['U']),1e-7)
        self.assertLess(la.norm(b['force']-a['force']@R.T)/max(la.norm(a['force']),1e-8),1e-4)

if __name__ == '__main__': unittest.main()
