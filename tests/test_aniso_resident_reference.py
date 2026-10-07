import os
import unittest
import numpy as np

from engine.aniso_phase1.convergence_reference import geometry,knots,tensor_rule,sites,state_from_field
from engine.aniso_phase1.history_increment import HistoryField,ReferenceBasis,frozen
from engine.aniso_phase1.consistent_transfer import bent_nodes
from engine.aniso_phase1.refinement import TensorReferenceQ1


@unittest.skipUnless(os.environ.get('ANISO_TEST_CUDA'),'optional real CUDA audit')
class ResidentTests(unittest.TestCase):
    def test_nonuniform_consistent_pic_matches_cpu_and_nodal_with_rollback(self):
        from unittest.mock import patch
        from tests.test_aniso_nonuniform_reference import small
        from engine.aniso_phase1.convergence_reference import union_knots
        from engine.aniso_phase1.resident_reference import ResidentReference
        old=geometry(17);source=small();basis=ReferenceBasis(old)
        initial=dict(position=HistoryField(((basis,frozen(bent_nodes(old))),)),
                     velocity=HistoryField(((basis,frozen(old.X*0)),)))
        axes=union_knots(knots(old),knots(source))
        p=sites(*tensor_rule(axes,2));q=sites(*tensor_rule(axes,5))
        gpu=ResidentReference(source,initial,p,q,transfer_mode='consistent_pic')
        nodal=ResidentReference(source,initial,p,q)
        state=state_from_field(initial['position'],p,q);cpu=TensorReferenceQ1(source,state)
        try:
            rng=np.random.default_rng(401)
            v=rng.normal(size=source.X.shape)*.01;v[source.fixed]=0
            vp=cpu.particles.N @ v
            np.testing.assert_allclose(cpu.project(vp),v,rtol=1e-11,atol=1e-14)
            self.assertAlmostEqual(float(np.sum(cpu.mass[:,None]*vp*vp)),float(np.sum(v*(cpu.M@v))),delta=1e-18)
            # Nonzero velocity exercises actual G2P/P2G on the very first step.
            for model in (gpu,nodal):
                model.velocity=v.copy();model.kinetic=.5*float(np.sum(v*(cpu.M@v)))
            for _ in range(8):
                info=gpu.step(.00003125);nodal.step(.00003125)
                state,vp,_=cpu.step(state,vp,.00003125)
                self.assertLess(info['transfer_roundtrip_relative'],1e-11)
                self.assertLess(abs(info['energy_budget_residual']),1e-16)
            for siteset,x,F in ((p,state.xp,state.Fp),(q,state.xq,state.Fq)):
                gx,gF=gpu.fields()['position'].evaluate(siteset.X)
                np.testing.assert_allclose(gx,x,atol=2e-12)
                np.testing.assert_allclose(gF,F,atol=2e-11)
            np.testing.assert_allclose(gpu.velocity,nodal.velocity,atol=2e-11)
            np.testing.assert_allclose(gpu.coefficients,nodal.coefficients,atol=2e-12)
            np.testing.assert_allclose(gpu.fields()['velocity'].evaluate(p.X)[0],vp,atol=2e-10)
            saved=[gpu.coefficients.copy(),gpu.velocity.copy(),gpu.energy,gpu.kinetic]
            with patch('engine.aniso_phase1.resident_reference.minimize_with_backtracking',side_effect=RuntimeError('forced failure')):
                with self.assertRaisesRegex(RuntimeError,'forced failure'):gpu.step(.001)
            for a,b in zip(saved,[gpu.coefficients,gpu.velocity,gpu.energy,gpu.kinetic]):np.testing.assert_array_equal(a,b)
            before=gpu.fields()['position'].evaluate(q.X)[1]
            bad=np.zeros_like(source.X);bad[:,0]=-2*(source.X[:,0]-.25)
            with self.assertRaises(ValueError):gpu.elastic(bad)
            np.testing.assert_array_equal(gpu.fields()['position'].evaluate(q.X)[1],before)
            # A rejected trial must not poison the next accepted PIC step.
            gpu.step(.00003125);nodal.step(.00003125)
            np.testing.assert_allclose(gpu.coefficients,nodal.coefficients,atol=2e-12)
        finally:
            gpu.close();nodal.close();cpu.close()

    def test_analytic_velocity_projection_uses_independent_quadrature(self):
        import warp as wp
        from engine.aniso_phase1.resident_reference import ResidentReference
        from engine.aniso_phase1.initial_controls import velocity_moments
        from engine.aniso_phase1.smooth_history import PolynomialHistoryBasis
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        source=geometry(17);basis=PolynomialHistoryBasis()
        c=np.zeros((len(basis.powers),3))
        c[np.all(basis.powers==[5,0,0],axis=1),1]=.01
        velocity=HistoryField(((basis,frozen(c)),))
        initial=dict(position=HistoryField(((ReferenceBasis(source),frozen(source.X)),)),velocity=velocity)
        p=sites(*tensor_rule(knots(source),2));q=sites(*tensor_rule(knots(source),5))
        k7,r7=velocity_moments(source,velocity,7);k9,r9=velocity_moments(source,velocity,9)
        model=ResidentReference(source,initial,p,q,velocity_moments=(k7,r7))
        np.testing.assert_allclose((model.base.M @ model.velocity)[source.free],r9[source.free],rtol=1e-12,atol=1e-18)
        self.assertAlmostEqual(model.kinetic,k9,delta=1e-20)
        k2,_=velocity_moments(source,velocity,2)
        self.assertGreater(abs(k2/k9-1),1e-5)
        info=model.step(.00003125)
        self.assertLessEqual(info['projection_delta'],0)
        self.assertLess(abs(info['energy_budget_residual']),1e-18)
        model.close()

    def test_constant_stress_balances_reference_surface_traction(self):
        import warp as wp
        from engine.aniso_phase1.resident_reference import ResidentReference
        from engine.aniso_phase1.initial_controls import initial_dead_traction
        from engine.aniso_phase1.types import AnisotropicMaterialParams
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        source=geometry(17);source.params=AnisotropicMaterialParams(10.,20.,0.)
        x=source.X.copy();x[:,0]+=.01*(x[:,0]-.25)
        initial=dict(position=HistoryField(((ReferenceBasis(source),frozen(x)),)),
                     velocity=HistoryField(((ReferenceBasis(source),frozen(x*0)),)))
        p=sites(*tensor_rule(knots(source),2));q=sites(*tensor_rule(knots(source),5))
        model=ResidentReference(source,initial,p,q)
        load=initial_dead_traction(source,initial['position'])
        force=model.elastic(model.coefficients)[1]
        np.testing.assert_allclose(force[source.free],load[source.free],atol=1e-13)
        model.step(.00003125,external=load)
        np.testing.assert_allclose(model.velocity,0,atol=1e-12)
        model.close()

    def test_force_derivative_trajectory_and_invalid_trials(self):
        import warp as wp
        from engine.aniso_phase1.resident_reference import ResidentReference
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        source=geometry(17)
        p=sites(*tensor_rule(knots(source),2)); q=sites(*tensor_rule(knots(source),5))
        initial=dict(position=HistoryField(((ReferenceBasis(source),frozen(bent_nodes(source))),)),
                     velocity=HistoryField(((ReferenceBasis(source),frozen(source.X*0)),)))
        gpu=ResidentReference(source,initial,p,q)
        state=state_from_field(initial['position'],p,q); cpu=TensorReferenceQ1(source,state)
        rng=np.random.default_rng(923)
        du=rng.normal(size=source.X.shape)*1e-4; du[source.fixed]=0
        U,f=gpu.elastic(du); Uc,fc=cpu.elastic(state,du)
        self.assertAlmostEqual(U,Uc,delta=1e-16)
        np.testing.assert_allclose(f,fc,atol=1e-13,rtol=1e-10)
        direction=du*10
        ep=gpu.elastic(du+1e-4*direction)[0]; em=gpu.elastic(du-1e-4*direction)[0]
        self.assertAlmostEqual((ep-em)/2e-4,np.sum(f*direction),delta=1e-11)
        bad=source.X*0; bad[:,0]=-2*(source.X[:,0]-.25)
        with self.assertRaises(ValueError): gpu.elastic(bad)
        vp=np.zeros_like(state.xp)
        for _ in range(16):
            info=gpu.step(.00003125)
            state,vp,_=cpu.step(state,vp,.00003125)
        x,F=gpu.fields()['position'].evaluate(q.X)
        np.testing.assert_allclose(x,state.xq,atol=2e-12)
        np.testing.assert_allclose(F,state.Fq,atol=2e-11)
        np.testing.assert_allclose(gpu.fields()['velocity'].evaluate(p.X)[0],vp,atol=2e-10)
        self.assertLess(abs(info['energy_budget_residual']),1e-16)
        # External load is a potential term; discrete mechanical work closes.
        before=gpu.energy+gpu.kinetic
        external=np.zeros_like(source.X); external[source.free,1]=1e-6
        info=gpu.step(.00003125,external=external)
        self.assertLess(abs(info['energy_budget_residual']),1e-16)
        self.assertNotEqual(info['external_work'],0)
        self.assertAlmostEqual(gpu.energy+gpu.kinetic-before,
            sum(info[k] for k in ('projection_delta','inertia_remainder','elastic_remainder','solver_work','external_work')),
            delta=1e-16)
        cpu.close();gpu.close()


if __name__=='__main__':unittest.main()
