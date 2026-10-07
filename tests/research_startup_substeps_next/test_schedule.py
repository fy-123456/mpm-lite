import unittest
import numpy as np
from engine.aniso_phase1.research_startup_substeps_next.schedule import *

class Observations(unittest.TestCase):
    def test_registered_tables_and_switch(self):
        for c,n in [('U2',20),('U4',28)]:
            for fine in (False,True):
                t=make_times(c,fine);self.assertEqual(len(t)-1,n*(2 if fine else 1));ix=indices(t,OBSERVATIONS)
                np.testing.assert_array_equal(t[ix],OBSERVATIONS)
                theta=theta_schedule(t,'startup');self.assertTrue(np.all(theta[t[1:]<=25e-6]==1));self.assertTrue(np.all(theta[t[:-1]>=25e-6]==.5))
    def test_signed_unequal_intervals_conserve_flux(self):
        t=np.array([0.,1.,3.]);z=np.array([[2.,-1.],[4.,-3.]]);Q=np.vstack([np.zeros(2),np.cumsum(z*np.diff(t)[:,None],axis=0)])
        v=aggregate(dict(flux=z,cumulative=Q,pressure=np.array([[5.],[3.],[1.]])),t,[0.,3.])
        np.testing.assert_allclose(v['flux'][0],[10/3,-7/3]);np.testing.assert_array_equal(v['pressure'][-1],[1.])
    def test_partial_interval_and_recovery_do_not_double_count(self):
        rows=[dict(time=1.,dt=1.,R=2.,D=.1),dict(time=3.,dt=2.,R=-1.,D=.3)]
        self.assertEqual(interval_rows(rows[:1],[0.,3.],average=['R'],summed=['D']),[])
        v=interval_rows(rows,[0.,3.],average=['R'],summed=['D']);self.assertEqual(v[0]['R'],0.);self.assertAlmostEqual(v[0]['D'],.4)
        self.assertEqual(v,interval_rows(rows,[0.,3.],average=['R'],summed=['D']))
    def test_bad_coverage_and_nonfinite_rejected(self):
        for t in ([0.,0.,1.],[1.,2.],[0.,np.nan],[0.,-1.]):
            with self.assertRaises(ValueError):validate_times(t)
        with self.assertRaises(ValueError):indices([0.,1.,3.],[0.,2.])
        with self.assertRaises(ValueError):interval_rows([dict(time=2.,dt=1.)],[0.,2.])
        with self.assertRaises(ValueError):aggregate(dict(flux=np.ones((2,1)),cumulative=np.zeros((3,1))),[0.,1.,2.],[0.,2.])
    def test_zero_flow_and_two_equal_step_restriction(self):
        t=np.arange(5.);z=np.zeros((4,2));v=dict(flux=z,cumulative=np.zeros((5,2)),pressure=np.arange(5.)[:,None])
        a=aggregate(v,t,t[::2]);np.testing.assert_array_equal(a['flux'],0);np.testing.assert_array_equal(a['pressure'],v['pressure'][::2])

if __name__=='__main__':unittest.main()
