import unittest
from unittest.mock import patch
import numpy as np
from engine.aniso_phase1.research_zero_source_next.coupled import ZeroSourceCoupling,explicit_zero
from benchmarks.research_zero_source_next.continuation import mass_budget,extended_identity
from engine.aniso_phase1.research_pressure_startup_next.theta import schedule
class Contract(unittest.TestCase):
    def test_source_rejected_before_model(self):
        with patch('engine.aniso_phase1.research_pressure_startup_next.coupled.StartupCoupling.__init__') as parent:
            for kwargs in ({},{'source_m3_s':None},{'source_m3_s':float('nan')},{'source_m3_s':1.}):
                with self.assertRaises(ValueError):ZeroSourceCoupling(None,None,[0,1],**kwargs)
            parent.assert_not_called()
        self.assertEqual(float(explicit_zero(0.)),0.)
    def test_longer_window_tightens_mass_tolerance(self):
        a=mass_budget([1.,3.],[.0002,.0006],[.2,.2],[.1,.1],1e-5,1.5e-4)
        b=mass_budget([1.,3.],[.0002,.0006],[.2,.2],[.1,.1],1e-5,2e-4)
        self.assertTrue(np.all(b<a));self.assertAlmostEqual(b[1]/b[0],3.)
    def test_switch_is_physical_time(self):
        t=np.linspace(0,2e-4,17);fine=np.linspace(0,2e-4,33)
        np.testing.assert_array_equal(schedule(t,'startup'),[1]*2+[.5]*14)
        np.testing.assert_array_equal(schedule(fine,'startup'),[1]*4+[.5]*28)
    def test_extension_does_not_erase_source_or_material(self):
        a={'core':{'window_s':1.,'alpha':.8},'source_m3_s':[0.]}
        b={'core':{'window_s':2.,'alpha':.8},'source_m3_s':[1.]}
        self.assertNotEqual(extended_identity(a),extended_identity(b))
if __name__=='__main__':unittest.main()
