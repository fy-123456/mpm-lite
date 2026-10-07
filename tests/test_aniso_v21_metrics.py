import unittest
import numpy as np
from benchmarks.aniso_v21_common import compare,mark
from benchmarks.aniso_v21_metrics import compare_many
from engine.aniso_phase1.tensor_reference import coordinates
class SpatialMetricTests(unittest.TestCase):
    def test_batch_matches_independent_gradient_integration_and_regions(self):
        e=[np.array([.125,.25,.3125,.5,.6875,.75,.875]),np.array([.375,.5,.625]),np.array([.375,.5,.625])];x=np.array(np.meshgrid(*coordinates(e,2),indexing='ij')).reshape(3,-1).T;u=np.column_stack((.01*x[:,0]**2,.003*x[:,0]*x[:,1],-.002*x[:,1]*x[:,2]));v=u+np.column_stack((.001*x[:,0]*x[:,1],-.0007*x[:,1]**2,.0003*x[:,2]**2));a=(e,2,u);b=(e,2,v);s=compare(a,b)['regions'];m=compare_many({'case':a},b)['case']
        for region in s:
            for key in s[region]:self.assertAlmostEqual(s[region][key],m[region][key],places=12)
        self.assertAlmostEqual(s['global']['volume'],.046875,places=13);self.assertAlmostEqual(s['grip']['volume']+s['interior']['volume'],s['global']['volume'],places=13)
    def test_indicator_marks_local_error_and_preserves_existing_mesh(self):
        e=[np.linspace(.125,.875,25),np.linspace(.375,.625,9),np.linspace(.375,.625,9)];hist=[np.zeros(len(x)-1) for x in e];hist[0][10]=10.;hist[1][4]=10.;hist[2][3]=10.;new,ids=mark(e,hist)
        for k in range(3):self.assertTrue(np.all(np.isin(e[k],new[k])))
        self.assertIn(10,ids[0]);self.assertIn(4,ids[1]);self.assertIn(3,ids[2]);self.assertIn(.25,new[0]);self.assertIn(.75,new[0])
if __name__=='__main__':unittest.main()
