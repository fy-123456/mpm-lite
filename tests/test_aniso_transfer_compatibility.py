"""Protect the production-promotion gate and its positive controls."""
import unittest
import numpy as np
from engine.aniso_phase1.transfer_compatibility import mass_audit,history_read
from engine.aniso_phase1.aligned_quadrature import box_rule
from benchmarks.aniso_quadrature_validation import setup


class TransferCompatibilityTests(unittest.TestCase):
    def test_signed_mls_lumping_is_not_a_positive_mass_replacement(self):
        snap,_,blend=setup(17);r=mass_audit(blend,snap.X,snap.volume,17)
        self.assertAlmostEqual(r['mls_lumped_sum'],r['total_mass'],places=13)
        self.assertGreater(r['negative_lumped_nodes'],0)
        self.assertGreater(r['negative_mass_fraction'],.05)
        self.assertGreater(r['lite_min_lumped'],0.)
        self.assertGreater(r['consistent_mass_soft_modes'],0)

    def test_affine_history_reproduces_exact_rigid_rotation_and_stretch(self):
        snap,_,blend=setup(17);a=.43
        Q=np.array([[np.cos(a),-np.sin(a),0.],[np.sin(a),np.cos(a),0.],[0,0,1.]])
        F=Q@np.diag([1.04,.99,1.]);x=(snap.X-.5)@F.T+.5
        q=box_rule(blend.h,2).points;query=(q-.5)@F.T+.5
        recovered,_=history_read(x,snap.X,np.broadcast_to(F,(len(x),3,3)),snap.volume,query,blend.h)
        np.testing.assert_allclose(recovered,np.broadcast_to(F,recovered.shape),rtol=1e-10,atol=1e-10)


if __name__=='__main__':unittest.main()
