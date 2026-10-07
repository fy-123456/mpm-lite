import unittest
from benchmarks.research_restoring_rt0_next.reaction import impulse

class ReactionIntegration(unittest.TestCase):
    def test_signed_impulse_with_clipped_variable_intervals(self):
        rows=[dict(time=.25,dt=.25,reaction_N=4.),dict(time=1.,dt=.75,reaction_N=-2.)]
        self.assertAlmostEqual(impulse(rows,.125,.75),-.5)
        self.assertAlmostEqual(impulse(rows,0.,1.),-.5)

    def test_missing_interval_rejected(self):
        with self.assertRaises(ValueError):impulse([dict(time=1.,dt=.5,reaction_N=2.)],0.,1.)
