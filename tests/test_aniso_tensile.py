import unittest
import numpy as np
import warp as wp
from demos.aniso import Config, Scene
from engine.aniso_phase1.tensile import loading_displacement


class TensileTests(unittest.TestCase):
    def test_loading_reversal_and_finish(self):
        self.assertAlmostEqual(loading_displacement(.2,.01,.2),.002)
        self.assertAlmostEqual(loading_displacement(.3,.01,.2),.001)
        self.assertEqual(loading_displacement(.4,.01,.2),0.)
        self.assertEqual(loading_displacement(.5,.01,.2),0.)

    def test_discrete_reaction_balances_momentum(self):
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        s=Scene(Config('tensile',9,.002,loading_time=.002), 'cpu')
        self.assertTrue(s.step())
        first=s.loading_rows[-1]
        self.assertGreater(first['right_force'],0.)
        self.assertLess(abs(first['momentum_balance_error']),1e-7)
        self.assertAlmostEqual(first['loading_work'],first['right_force']*first['displacement'])
        self.assertTrue(s.step())
        self.assertEqual(s.loading_rows[-1]['displacement'],0.)
        self.assertLess(s.loading_rows[-1]['loading_velocity'],0.)
        self.assertAlmostEqual(s.solver.ptc_m.numpy().sum(),.046875)


if __name__=='__main__':unittest.main()
