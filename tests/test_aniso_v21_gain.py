import unittest
import numpy as np
from engine.aniso_phase1 import local_reference as ref
from engine.aniso_phase1.stress_gain import reference_stress_load,predicted_reduction
from benchmarks.aniso_v21_common import compare,hessian
class CorrectableStressTests(unittest.TestCase):
    def test_predicted_reduction_equals_independent_integrated_stress_difference(self):
        e=[np.array([.125,.25,.3125,.5,.6875,.75,.875]),np.array([.375,.5,.625]),np.array([.375,.5,.625])];H=hessian('F45');x,K=ref.assemble(e,2,H.T@H);u=np.column_stack((.01*x[:,0]**2,.002*x[:,0]*x[:,1],-.001*x[:,1]*x[:,2]));v=u+.0005*x*x;reference=(e,2,v);g,norm=reference_stress_load(e,reference,H);r=K@u.T.ravel()-g;w=np.random.default_rng(31).normal(scale=1e-5,size=u.shape);ids=np.arange(3*len(x));pred=predicted_reduction(ids,w.T.ravel(),r,K);before=compare((e,2,u),reference)['regions']['global']['stress_error_squared'];after=compare((e,2,u+w),reference)['regions']['global']['stress_error_squared'];self.assertAlmostEqual(before-after,pred,places=12);self.assertGreater(norm,0)
if __name__=='__main__':unittest.main()
