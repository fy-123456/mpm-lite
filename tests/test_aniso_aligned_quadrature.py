"""Meaningful integration and map invariants for the physical-domain reference."""
import unittest
import numpy as np
from engine.aniso_phase1.aligned_quadrature import (partitions,box_rule,face_rule,evaluate,LOW,HIGH)
from benchmarks.aniso_quadrature_validation import setup


class AlignedQuadratureTests(unittest.TestCase):
    def test_physical_volume_and_polynomial_exactness(self):
        rule=box_rule(1/16,4)
        self.assertEqual(len(rule.weights),81*64)
        self.assertTrue(np.all(rule.weights>0))
        self.assertTrue(np.all((rule.points>LOW)&(rule.points<HIGH)))
        self.assertAlmostEqual(rule.weights.sum(),np.prod(HIGH-LOW),places=15)
        # Coordinatewise degree 6, independent of the stiffness implementation.
        powers=np.array([6,4,2]);expected=np.prod((HIGH**(powers+1)-LOW**(powers+1))/(powers+1))
        self.assertAlmostEqual(np.sum(rule.weights*np.prod(rule.points**powers,axis=1))/expected,1,places=12)
        q,w=face_rule(1/16);self.assertAlmostEqual(w.sum(),.125**2,places=15)
        self.assertTrue(np.all(q[:,0]==.75))

    def test_knot_alignment_is_needed_for_piecewise_polynomials(self):
        rule=box_rule(1/16,4);knot=.40625
        measured=rule.weights@np.abs(rule.points[:,0]-knot)
        expected=.5*((knot-LOW[0])**2+(HIGH[0]-knot)**2)*np.prod((HIGH-LOW)[1:])
        self.assertAlmostEqual(measured/expected,1,places=13)
        # Single box high-order quadrature crosses the kink and is not exact.
        z,w=np.polynomial.legendre.leggauss(4);x=.5+.25*z
        wrong=.25*(w@np.abs(x-knot))*.125**2
        self.assertGreater(abs(wrong/expected-1),1e-3)

    def test_batched_map_matches_independent_pointwise_implementation(self):
        _,nodes,blend=setup(17)
        rng=np.random.default_rng(88);points=LOW+(HIGH-LOW)*rng.random((19,3))
        points=np.vstack([points,[.25,.5,.5],[.75,.5,.5],[.40625,.46875,.53125]])
        N,G=evaluate(blend,points);oldN,oldG=blend.evaluate(points)
        np.testing.assert_allclose(N.toarray(),oldN.toarray(),atol=2e-14)
        for a,b in zip(G,oldG):np.testing.assert_allclose(a.toarray(),b.toarray(),atol=3e-13)
        np.testing.assert_allclose(N@nodes,points,atol=2e-13)
        for d in range(3):np.testing.assert_allclose(G[d]@nodes,np.tile(np.eye(3)[d],(len(points),1)),atol=1e-11)

    def test_checked_pcg_rejects_negative_curvature(self):
        from scipy.sparse.linalg import aslinearoperator
        from engine.aniso_phase1.quadrature_probe import checked_pcg
        M=aslinearoperator(np.eye(2));b=np.array([1.,0.])
        _,info,_=checked_pcg(aslinearoperator(np.diag([-1.,2.])),b,M)
        self.assertEqual(info,-1)
        x,info,_=checked_pcg(aslinearoperator(np.diag([2.,3.])),np.ones(2),M)
        self.assertEqual(info,0);np.testing.assert_allclose(x,[.5,1/3])

    def test_low_order_can_miss_positive_strain_energy(self):
        # A quadratic displacement has nonzero strain away from the midpoint.
        z,w=np.polynomial.legendre.leggauss(1);reduced=w@(2*z)**2
        z,w=np.polynomial.legendre.leggauss(4);full=w@(2*z)**2
        self.assertEqual(reduced,0.)
        self.assertAlmostEqual(full,8/3,places=13)


if __name__=='__main__':unittest.main()
