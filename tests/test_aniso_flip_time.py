import unittest
import numpy as np
from benchmarks.aniso_flip_time import scaled_flip


class FlipTimeTests(unittest.TestCase):
    def test_equal_physical_interval_mixing(self):
        target=np.array([.3,-.2,.7]);initial=np.array([-.1,.5,.2]);results=[]
        for substeps in (1,2,4,10):
            dt=.001/substeps;beta=scaled_flip(dt);v=initial.copy()
            for _ in range(substeps):v=beta*v+(1-beta)*target
            np.testing.assert_allclose(v,.9*initial+.1*target,atol=1e-15,rtol=0)
            results.append(v)
        self.assertEqual(scaled_flip(.001),.9)
        self.assertAlmostEqual(scaled_flip(.00025)**4,.9,places=15)
        self.assertNotAlmostEqual(.9**4,.9,places=3)

    def test_limits_invalid_inputs_and_nonuniform_partition(self):
        self.assertEqual(scaled_flip(.0005,0.),0.)
        self.assertEqual(scaled_flip(.0005,1.),1.)
        self.assertAlmostEqual(scaled_flip(.0002)*scaled_flip(.0008),scaled_flip(.001),places=15)
        for args in ((0.,),(-1.,),(float('nan'),),(float('inf'),),(.001,-.1),(.001,1.1),(.001,.9,0.),(.001,.9,float('inf'))):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):scaled_flip(*args)


if __name__=='__main__':unittest.main()
