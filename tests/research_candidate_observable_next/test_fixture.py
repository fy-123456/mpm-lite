import copy
import unittest
from benchmarks.research_candidate_observable_next import __doc__
from engine.aniso_phase1.research_candidate_observable_next.fixture import parameters

class ExplicitFixtureTests(unittest.TestCase):
    def setUp(self):self.protocol={'parameters':dict(alpha=.8,storage=.0002,pressure0_Pa=.2,reservoir_Pa=.002,mobility_scale=1e-7,source_density_s_inv=0.)}
    def test_returns_owned_parameters(self):
        p=parameters(self.protocol);p['pressure0_Pa']=1.;self.assertEqual(self.protocol['parameters']['pressure0_Pa'],.2)
    def test_every_parameter_must_be_explicit(self):
        for k in self.protocol['parameters']:
            p=copy.deepcopy(self.protocol);del p['parameters'][k]
            with self.assertRaises(ValueError):parameters(p)
    def test_reject_changed_coupling_strength_or_negative_mobility(self):
        for k,v in [('alpha',.4),('storage',0.),('mobility_scale',-1.),('pressure0_Pa',-1.)]:
            p=copy.deepcopy(self.protocol);p['parameters'][k]=v
            with self.assertRaises(ValueError):parameters(p)
    def test_reject_nonfinite_parameter(self):
        for k in self.protocol['parameters']:
            p=copy.deepcopy(self.protocol);p['parameters'][k]=float('nan')
            with self.assertRaises(ValueError):parameters(p)

if __name__=='__main__':unittest.main()
