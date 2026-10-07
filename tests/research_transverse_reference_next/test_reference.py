import unittest
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.research_transverse_reference_next.reference import algebra,exact,integrate,initial,mode_increment_check


class ReferenceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cuts=[[0.,.5,1.],[0.,.5,1.],[0.,1.]]
        cls.mobility=np.array([[.1,.02,.01],[.02,.08,-.015],[.01,-.015,.06]])*1e-7
        cls.a=algebra(cls.cuts,cls.mobility,.0002,.002,source=0.)

    def test_explicit_initial_and_source(self):
        for v in (.2,[.2]*3,[.2,.2,np.nan,.2]):
            with self.assertRaises(ValueError):initial(v,4)
        with self.assertRaises(ValueError):algebra(self.cuts,self.mobility,.0002,.002,source=1e-9)
        p=np.arange(4,dtype=float);owned=initial(p,4);p[:]=99
        np.testing.assert_array_equal(owned,np.arange(4))

    def test_matrix_exponential_and_mass(self):
        a=self.a;p=np.array([.17,.19,.21,.23]);t=np.array([0.,.001,.003]);v=exact(a,p,t)
        matrix=np.zeros((5,5));matrix[:4,:4]=-a['L']/a['C'][:,None];matrix[:4,4]=a['rhs']/a['C']
        q=np.array([(la.expm(s*matrix)@np.r_[p,1.])[:4] for s in t])
        np.testing.assert_allclose(v['pressure'],q,rtol=1e-6,atol=1e-8)
        np.testing.assert_allclose((v['pressure']-p)@a['C']+v['cumulative']@a['top'].boundary_sign,0,atol=1e-12)

    def test_equilibrium_and_interval_mean(self):
        a=self.a;t=np.array([0.,.001,.003]);v=exact(a,np.full(4,.002),t)
        np.testing.assert_allclose(v['pressure'],.002,atol=1e-10)
        np.testing.assert_allclose(v['cumulative'],0,atol=1e-12)
        w=exact(a,np.array([.17,.19,.21,.23]),t)
        np.testing.assert_allclose(np.sum(w['flux']*np.diff(t)[:,None],axis=0),w['cumulative'][-1],atol=1e-12)

    def test_theta_ledger_and_switch(self):
        a=self.a;t=[0.,12.5e-6,25e-6,37.5e-6];v=integrate(a,[.17,.19,.21,.23],t)
        np.testing.assert_array_equal(v['theta'],[1.,1.,.5])
        self.assertTrue(all(abs(r['energy_balance_J'])<1e-9 and r['source_work_J']==0 for r in v['ledger']))
        self.assertEqual(v['ledger'][-1]['numerical_dissipation_J'],0.)

    def test_signal_is_separate_from_passing_error(self):
        v=mode_increment_check(np.array([.04,.039999]),np.array([.04,.039998]))
        self.assertTrue(v['passed']);self.assertFalse(v['signal_resolved'])
        self.assertTrue(mode_increment_check(np.array([.04,.039]),np.array([.04,.039]))['signal_resolved'])


if __name__=='__main__':unittest.main()
