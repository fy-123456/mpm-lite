"""Cache correctness is state identity and ownership, not approximate equality."""
import unittest
from types import SimpleNamespace
import numpy as np
from engine.aniso_phase1.research_sequential_next.linearization import LinearizationCache


class FakeOperator:
    def __init__(self,signature='A'):
        self.signature=signature;self.device='cpu';self.count=1;self.space=SimpleNamespace(q_shape=(2,3),_check=self.check)
        self.calls=0
    @staticmethod
    def check(value,shape,name):
        a=np.asarray(value,dtype=float)
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError(name)
        return a
    def prepare(self,q):
        self.calls+=1
        return SimpleNamespace(F=q.copy(),Q=q.copy(),c=q.copy(),response={'force':q.copy(),'U':float(np.sum(q*q))})
    def action(self,lin,d):return lin.F*d


class CacheTests(unittest.TestCase):
    def test_exact_state_only_and_foreign_handles(self):
        op=FakeOperator();cache=LinearizationCache(op,max_bytes=1<<20,max_entries=2);q=np.ones((2,3))
        token,created=cache.prepare(q);self.assertTrue(created)
        same,created=cache.prepare(q.copy());self.assertFalse(created);self.assertEqual(token,same);self.assertEqual(op.calls,1)
        with self.assertRaises(ValueError):cache.action(token,q+1e-12,q)
        other=LinearizationCache(FakeOperator('B'),max_bytes=1<<20)
        with self.assertRaises(ValueError):other.action(token,q,q)
        np.testing.assert_array_equal(cache.action(token,q,np.full_like(q,2.)),2*q)
        q[0,0]=9.
        with self.assertRaises(ValueError):cache.action(token,q,q)

    def test_response_cannot_poison_cache_and_eviction_is_bounded(self):
        cache=LinearizationCache(FakeOperator(),max_bytes=1<<20,max_entries=1);q=np.ones((2,3))
        token,_=cache.prepare(q);response=cache.response(token,q)
        with self.assertRaises(ValueError):response['force'][0,0]=99.
        response['force'].setflags(write=True);response['force'][:]=0.
        np.testing.assert_array_equal(cache.response(token,q)['force'],q)
        cache.prepare(2*q)
        with self.assertRaises(ValueError):cache.action(token,q,q)
        self.assertLessEqual(cache.used_bytes,cache.max_bytes);self.assertEqual(cache.report()['entries'],1)
        cache.clear();self.assertEqual(cache.used_bytes,0)
        with self.assertRaises(ValueError):cache.response(token,q)

    def test_nonfinite_and_unusable_budget_rejected(self):
        with self.assertRaises(MemoryError):LinearizationCache(FakeOperator(),max_bytes=32)
        cache=LinearizationCache(FakeOperator(),max_bytes=1<<20)
        with self.assertRaises(ValueError):cache.prepare(np.full((2,3),np.nan))


if __name__=='__main__':unittest.main()
