import unittest
import os
import numpy as np

from engine.aniso_phase1.refinement import CachedFrozenQ1, TensorReferenceQ1
from engine.aniso_phase1.convergence_reference import (
    FrozenResidualQ1, geometry, knots, union_knots, tensor_rule, sites, state_from_field, rms)
from engine.aniso_phase1.history_increment import HistoryField, ReferenceBasis, frozen
from engine.aniso_phase1.consistent_transfer import bent_nodes


class RefinementTests(unittest.TestCase):
    def test_tensor_reference_rejects_inexact_mass_samples(self):
        source=geometry(17)
        X,w=tensor_rule(knots(source),2)
        q=sites(X,w)
        disturbed=X.copy(); disturbed[0,0]+=.001
        particles=sites(disturbed,w)
        field=HistoryField(((ReferenceBasis(source),frozen(bent_nodes(source))),))
        state=state_from_field(field,particles,q)
        with self.assertRaisesRegex(ValueError,'exactly integrate'):
            TensorReferenceQ1(source,state)

    @unittest.skipUnless(os.environ.get('ANISO_TEST_CUDA'), 'optional real CUDA audit')
    def test_device_material_and_trajectory_match_cpu(self):
        import warp as wp
        from engine.aniso_phase1.refinement_gpu import DeviceMaterial
        from engine.aniso_phase1.consistent_transfer import material_response
        from engine.aniso_phase1.convergence_reference import directions
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        rng=np.random.default_rng(471)
        F=np.eye(3)+rng.normal(size=(256,3,3))*.08
        F[0]=np.eye(3)
        F[1]=[[0,-1,0],[1,0,0],[0,0,1]]
        X=rng.uniform([.25,.4375,.4375],[.75,.5625,.5625],size=(len(F),3))
        A=directions(X); source=geometry(17)
        gpu=DeviceMaterial(A,source.params)
        for cpu,device in zip(material_response(F,A,source.params),gpu(F)):
            np.testing.assert_allclose(cpu,device,atol=2e-11,rtol=2e-11)
        bad=F.copy(); bad[0,0,0]=-1
        with self.assertRaises(ValueError): gpu(bad)
        q=sites(*tensor_rule(knots(source),5))
        field=HistoryField(((ReferenceBasis(source),frozen(bent_nodes(source))),))
        s=state_from_field(field,q,q)
        a=TensorReferenceQ1(source,s); b=TensorReferenceQ1(source,s,device='cuda:0')
        sa,sb=s,s; va=np.zeros_like(s.xp); vb=va.copy()
        for _ in range(4):
            sa,va,_=a.step(sa,va,.000015625)
            sb,vb,_=b.step(sb,vb,.000015625)
        np.testing.assert_allclose(va,vb,atol=1e-10)
        np.testing.assert_allclose(sa.Fq,sb.Fq,atol=1e-12)
        a.close(); b.close()
    def test_tensor_mass_and_dynamics_match_full_cholesky(self):
        source = geometry(17)
        q = sites(*tensor_rule(knots(source), 3))
        field = HistoryField(((ReferenceBasis(source), frozen(bent_nodes(source))),))
        s = state_from_field(field, q, q)
        tensor = TensorReferenceQ1(source,s)
        dense = CachedFrozenQ1(source,s,knots(source),enrich=False)
        rhs = np.random.default_rng(413).normal(size=(tensor.R.shape[1],3))
        np.testing.assert_allclose(tensor.M.toarray(),dense.M.toarray(),atol=1e-17)
        np.testing.assert_allclose(tensor.mass_solve(rhs),dense.mass_solve(rhs),rtol=1e-11,atol=1e-9)
        np.testing.assert_allclose(tensor.mass_from_y(tensor.mass_to_y(rhs)),rhs,atol=1e-13)
        np.testing.assert_allclose(tensor.mass_to_y(rhs),dense.mass_to_y(rhs),atol=1e-14)
        sa,sb = s,s
        va = np.zeros_like(s.xp); vb=va.copy()
        for _ in range(3):
            sa,va,_=tensor.step(sa,va,.00003125)
            sb,vb,_=dense.step(sb,vb,.00003125)
        np.testing.assert_allclose(va,vb,atol=1e-10)
        np.testing.assert_allclose(sa.Fq,sb.Fq,atol=1e-12)
        tensor.close()

    def test_cached_step_matches_original_and_rejects_bad_commit(self):
        source = geometry(17)
        axes = knots(source, True)
        q = sites(*tensor_rule(union_knots(knots(source), axes), 3))
        field = HistoryField(((ReferenceBasis(source), frozen(bent_nodes(source))),))
        s = state_from_field(field, q, q)
        a = FrozenResidualQ1(source, s, axes)
        b = CachedFrozenQ1(source, s, axes)
        sa, sb = s, s
        va = np.zeros_like(s.xp)
        vb = va.copy()
        for _ in range(5):
            sa, va, _ = a.step(sa, va, .00003125)
            sb, vb, info = b.step(sb, vb, .00003125)
            self.assertLess(rms(va-vb, s.mass), 1e-10)
            np.testing.assert_allclose(sa.Fq, sb.Fq, atol=1e-12)
            self.assertLess(abs(info['energy_budget_residual']), 1e-16)
        v = np.zeros((b.ncoeff, 3))
        v[:b.nnode, 0] = -100*(b.X[:, 0]-b.X[:, 0].min())
        with self.assertRaises(ValueError):
            b.commit(sb, v, 1.)


if __name__ == '__main__':
    unittest.main()
