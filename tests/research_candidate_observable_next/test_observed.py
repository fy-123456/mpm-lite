import unittest
from types import SimpleNamespace
from benchmarks.research_candidate_observable_next.observed import sample_guard
from engine.aniso_phase1.research_sequential_next.resources import GPUMemoryBudget

class Device:
    def __init__(self,values):self.values=iter(values);self.reads=0
    @property
    def free_memory(self):self.reads+=1;return next(self.values)

class ExactDecisionTests(unittest.TestCase):
    def guard(self,free):
        d=Device([free,999999]);g=SimpleNamespace(device=d,initial_free=1000,budget=700,minimum_free=1000,samples=0,peak_required=0)
        return g,d
    def test_success_sample_is_original_decision(self):
        g,d=self.guard(800);value,sample,error=sample_guard(g,GPUMemoryBudget.observe,400)
        self.assertEqual((value,sample,error),(800,800,None));self.assertEqual(d.reads,1);self.assertIs(g.device,d);self.assertEqual(g.peak_required,600)
    def test_required_rejection_retains_exact_sample_and_guard_counters(self):
        g,d=self.guard(500);value,sample,error=sample_guard(g,GPUMemoryBudget.observe,300)
        self.assertIsInstance(error,MemoryError);self.assertIsNone(value);self.assertEqual(sample,500);self.assertEqual(d.reads,1);self.assertEqual(g.samples,1);self.assertEqual(g.peak_required,800);self.assertIs(g.device,d)
    def test_reserve_free_rejection_even_with_budget_available(self):
        g,d=self.guard(1200);value,sample,error=sample_guard(g,GPUMemoryBudget.observe,1201)
        self.assertIsInstance(error,MemoryError);self.assertEqual(sample,1200);self.assertIs(g.device,d)
    def test_original_exception_restores_device(self):
        g,d=self.guard(800)
        def broken(*args):raise ValueError('driver error')
        with self.assertRaisesRegex(ValueError,'driver error'):sample_guard(g,broken,0)
        self.assertIs(g.device,d)
    def test_exact_boundary_allowed_next_byte_rejected(self):
        for reserve,reject in ((500,False),(501,True)):
            g,d=self.guard(800);value,sample,error=sample_guard(g,GPUMemoryBudget.observe,reserve)
            self.assertEqual(error is not None,reject);self.assertEqual(d.reads,1)
    def test_free_memory_recovery_preserves_high_water(self):
        g,d=self.guard(500);d.values=iter([500,900])
        sample_guard(g,GPUMemoryBudget.observe,100)
        value,free,error=sample_guard(g,GPUMemoryBudget.observe,100)
        self.assertEqual((value,free,error),(900,900,None));self.assertEqual(g.minimum_free,500);self.assertEqual(g.peak_required,600);self.assertEqual(g.samples,2)

if __name__=='__main__':unittest.main()
