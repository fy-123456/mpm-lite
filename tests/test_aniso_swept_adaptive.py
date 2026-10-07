import unittest
import numpy as np
from engine.aniso_phase1.swept_quadrature import sweep

class AdaptiveMovingCutTest(unittest.TestCase):
    def test_curved_crossing_refinement_against_analytic_integral(self):
        # Trilinear motion has a curved kernel-crossing surface in reference
        # coordinates. Integrate a kink exactly in x; outer refinement must
        # converge to the analytic y/z integral, not just preserve volume.
        def physical(X):
            y=X.copy();y[:,0]=X[:,0]*(1+.3*X[:,1])+.2*X[:,2];return y
        exact=.175+(.25-.1+.04/3)*np.log(1.3)/.3;errors=[]
        # The first midpoint split coincides with existing physical y/z
        # kernel cuts; levels 2 and 3 add genuinely finer intervals.
        for level in (0,2,3):
            X,V,_=sweep([np.array([0.,1.])]*3,physical,1.,order=2,outer_refine=level);errors.append(abs(float(V@abs(physical(X)[:,0]-.5))-exact));self.assertTrue(np.all(V>0))
        self.assertLess(errors[1],errors[0]/4);self.assertLess(errors[2],errors[1]/4);self.assertLess(errors[2],3e-8)
if __name__=='__main__':unittest.main()
