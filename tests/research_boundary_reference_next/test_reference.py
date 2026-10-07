import unittest
import numpy as np
from engine.aniso_phase1.research_boundary_reference_next.reference import slab_reference,boundary_grid
class Reference(unittest.TestCase):
    def test_cell_content_matches_two_boundary_integrals(self):
        v=slab_reference([0.,1e-5,1e-4,.1,.9999,.99999,1.],[0.,1e-7,1e-5,2e-4],mobility=1e-8,storage=.0002,p0=.2,reservoir=.002,area=.04)
        np.testing.assert_allclose((.2-v['pressure'])@v['cell_capacity'],2*v['boundary_cumulative_per_end_m3'],atol=1e-17,rtol=1e-6)
        self.assertTrue(np.isfinite(v['boundary_interval_per_end_m3_s']).all())
        np.testing.assert_array_equal(v['pressure'][0],.2)
    def test_first_interval_scaling_and_telescope(self):
        v=slab_reference([0.,.5,1.],[0.,1e-6,4e-6],mobility=1e-8,storage=.0002,p0=.2,reservoir=.002,area=.04)
        Q=v['boundary_cumulative_per_end_m3'];self.assertAlmostEqual(Q[2]/Q[1],2.)
        self.assertAlmostEqual(np.diff([0,1e-6,4e-6])@v['boundary_interval_per_end_m3_s'],Q[-1])
    def test_invalid_scope_and_nonfinite_rejected(self):
        for times in ([1.,2.],[0.,float('nan')],[0.,100000.]):
            with self.assertRaises(ValueError):slab_reference([0.,1.],times,mobility=1e-8,storage=.0002,p0=.2,reservoir=.002,area=.04)
    def test_boundary_grid_preserves_domain_and_symmetry(self):
        x=np.array(boundary_grid([[.125,.875],[0.,1.],[0.,1.]],4e-6)[0]);self.assertEqual(len(x),17);self.assertTrue(np.all(np.diff(x)>0));np.testing.assert_allclose(x+x[::-1],1.);self.assertAlmostEqual(x[1]-x[0],4e-6)
class GeometryWork(unittest.TestCase):
    def test_simpson_gradient_integrates_cubic_volume_exactly(self):
        from engine.aniso_phase1.research_boundary_reference_next.geometry import SharedMidpointGeometry
        g=SharedMidpointGeometry.__new__(SharedMidpointGeometry)
        g.evaluate=lambda F:dict(gradient=np.linalg.det(F)*np.linalg.inv(F).T)
        A=np.array([[1.02,.12,.03],[.01,.98,.07],[.02,-.03,1.04]])
        B=np.array([[.94,-.08,.05],[.11,1.05,.02],[.04,.08,.99]])
        G=g.discrete(A,B)
        self.assertAlmostEqual(float(np.sum(G*(B-A))),np.linalg.det(B)-np.linalg.det(A),places=12)
        self.assertTrue(np.allclose(g.discrete(A,A),g.evaluate(A)['gradient']))

if __name__=='__main__':unittest.main()
