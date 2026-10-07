import unittest
import numpy as np
from benchmarks.aniso_v21_common import compare,hessian
from engine.aniso_phase1.tensor_reference import coordinates
from engine.aniso_phase1.tensor_metrics import compare_fields,evaluate_gradient
from benchmarks.aniso_local_q3 import gradient

class TensorMetricTests(unittest.TestCase):
    def test_tensor_quadrature_matches_independent_pointwise_regional_integrals(self):
        rng=np.random.default_rng(22);edges=[np.array([.125,.25,.5,.75,.875]),np.array([.375,.5,.625]),np.array([.375,.48,.625])];other=[np.union1d(e,(e[:-1]+e[1:])/2) for e in edges]
        fields=[]
        for e,p in [(edges,3),(other,4)]:
            X=np.array(np.meshgrid(*coordinates(e,p),indexing='ij')).reshape(3,-1).T;u=np.column_stack((X[:,0]**p*.002,X[:,1]**3*.001,X[:,0]*X[:,2]*.003));u+=rng.normal(scale=1e-6,size=u.shape);fields.append((e,p,u))
        fast=compare_fields(*fields,hessian('F45'),np.array([1,1,0])/np.sqrt(2),indicators=True)
        slow=compare(*fields,indicators=True)
        for k in fast['regions']:
            for n in fast['regions'][k]:np.testing.assert_allclose(fast['regions'][k][n],slow['regions'][k][n],rtol=2e-11,atol=1e-13)
        for a,b in zip(fast['axis_indicators'],slow['axis_indicators']):np.testing.assert_allclose(a,b,rtol=2e-11,atol=1e-16)
    def test_tensor_gradient_on_nonmatching_quartic_mesh(self):
        e=[np.array([.125,.25,.75,.875]),np.array([.375,.47,.625]),np.array([.375,.5,.625])];p=4;X=np.array(np.meshgrid(*coordinates(e,p),indexing='ij')).reshape(3,-1).T;u=np.random.default_rng(9).normal(size=(len(X),3));q=[np.linspace(v[0]+.001,v[-1]-.001,9) for v in e];P=np.array(np.meshgrid(*q,indexing='ij')).reshape(3,-1).T
        np.testing.assert_allclose(evaluate_gradient((e,p,u),q).reshape(-1,3,3),gradient(P,e,p,u),atol=1e-11,rtol=2e-12)
if __name__=='__main__':unittest.main()
