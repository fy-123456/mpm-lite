import unittest
import numpy as np
from engine.aniso_phase1.research_observable_boundary_next.rt0 import CartesianTopology
from benchmarks.research_restoring_rt0_next.grid_transfer import restriction

class GridTests(unittest.TestCase):
    def test_extensive_and_intensive(self):
        a=CartesianTopology(([0.,.2,1.],[0.,1.],[0.,1.]));b=CartesianTopology(([0.,.1,.2,.5,1.],[0.,1.],[0.,1.]));P,C,Z=restriction(a,b)
        np.testing.assert_allclose(P@np.full(b.cells,2.),2.);np.testing.assert_allclose(C@b.V0,a.V0)
        rng=np.random.default_rng(2);z=rng.normal(size=b.nflux);np.testing.assert_allclose(a.B@(Z@z),C@(b.B@z),atol=1e-12)
        np.testing.assert_allclose(np.sum(C,axis=0),1.)
    def test_transverse_subdivision(self):
        a=CartesianTopology(([0.,1.],[0.,1.],[0.,1.]));b=CartesianTopology(([0.,.5,1.],[0.,.5,1.],[0.,1.]));P,C,Z=restriction(a,b)
        np.testing.assert_allclose(a.B@Z,C@b.B)
    def test_foreign_domain(self):
        a=CartesianTopology(([0.,1.],[0.,1.],[0.,1.]));b=CartesianTopology(([0.,.5],[0.,1.],[0.,1.]))
        with self.assertRaises(ValueError):restriction(a,b)

if __name__=='__main__':unittest.main()
