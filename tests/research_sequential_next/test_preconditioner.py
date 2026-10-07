"""Exact AVF equations, factor invalidation and true-state rollback."""
import inspect
import unittest
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.research_c.stage2.dynamics import AVF
from engine.aniso_phase1.research_sequential_next.preconditioner import StaticFactors,ReusedAVF
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF,StepRejected
from benchmarks.research_sequential_next.config import protocol,validate,identity
from .test_runtime import ToyModel


class FactorTests(unittest.TestCase):
    def test_sealed_equations_only_factor_provider_changes(self):
        original=inspect.getsource(AVF._compute).replace('def _compute(self,','def _advance(self,',1)
        original=original.replace('factor = la.lu_factor(H)  # search only; no SPD assumption or physical shift',
                                  'factor = self._factor(H, dt)  # same matrix; only its exact LU is reused')
        self.assertEqual(original,inspect.getsource(ReusedAVF._advance))

    def test_factor_content_model_dt_and_bounded_storage(self):
        factors=StaticFactors(max_entries=2);H=np.array([[2.,.1],[.1,1.]])
        a=factors.get(H,.025,'M');b=factors.get(H.copy(),.025,'M');self.assertIs(a,b)
        np.testing.assert_allclose(H@la.lu_solve(b,np.ones(2)),np.ones(2),atol=1e-14)
        for value,dt,model in [(H,.05,'M'),(H,.05,'N'),(H+.1,.05,'N')]:factors.get(value,dt,model)
        self.assertEqual(factors.report()['misses'],4);self.assertEqual(factors.report()['hits'],1)
        self.assertEqual(factors.report()['entries'],2);self.assertGreater(factors.report()['evictions'],0)
        with self.assertRaises(MemoryError):StaticFactors(max_bytes=1).get(H,.1,'M')

    def test_new_protocol_and_legacy_interpretation(self):
        old=protocol(1.,device='cpu');del old['implementation'];self.assertNotIn('implementation',validate(old))
        base=protocol(1.);cached=protocol(1.,linearization_cache=True)
        self.assertNotEqual(identity(base),identity(cached))
        with self.assertRaises(ValueError):protocol(1.,device='cpu',operator_mode='segmented')
        with self.assertRaises(ValueError):protocol(1.,operator_mode='original',linearization_cache=True)

    def test_reuse_preserves_steps_and_failure_retry(self):
        m=ToyModel();m.signature='toy-physical-model';cfg=protocol(1.,device='cpu')
        a=ValidatedAVF(m,cfg);b=ReusedAVF(m,cfg)
        for h in [.025,.025,.05]:
            ra=a.step(h);rb=b.step(h)
            np.testing.assert_allclose(a.state.q,b.state.q,rtol=0,atol=1e-14)
            np.testing.assert_allclose(a.state.velocity,b.state.velocity,rtol=0,atol=1e-14)
            self.assertAlmostEqual(ra['true_residual'],rb['true_residual'])
        before=b.state.digest()
        def fail(where,state):
            if where=='after_prepare':raise ValueError('intentional rejected factor-cache step')
        with self.assertRaises(StepRejected):b.step(.025,inject=fail)
        self.assertEqual(before,b.state.digest());a.step(.025);b.step(.025)
        np.testing.assert_allclose(a.state.q,b.state.q,rtol=0,atol=1e-14)
        self.assertGreater(b.factors.hits,0)


if __name__=='__main__':unittest.main()
