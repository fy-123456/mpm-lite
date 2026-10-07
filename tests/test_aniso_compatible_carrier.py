import unittest
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v17_modes import controlled_case
from benchmarks.aniso_carrier_joint import spectrum
from engine.aniso_phase1.compatible_carrier import CompatibleReconstruction,make_case
from engine.aniso_phase1.carrier_joint import gradient
from engine.aniso_phase1.compatible_avf import CompatibleAVF,CompatibleGeometry

class CompatibleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s,e,m,h,meta=controlled_case();cls.rec=CompatibleReconstruction(s.Y,h,16);cls.case=make_case(cls.rec,3)
    def test_quadratic_and_grip_compatibility(self):
        s,e,m,h,_=self.case;X=e.reference_particles;Y=s.Y;T,*G=self.rec.maps(X)
        pol=lambda x:np.column_stack([np.ones(len(x)),x,x*x,x[:,0]*x[:,1],x[:,0]*x[:,2],x[:,1]*x[:,2]])
        np.testing.assert_allclose(T@pol(Y),pol(X),atol=1e-12)
        for j in range(3):
            d=np.zeros((len(X),10));d[:,1+j]=1;d[:,4+j]=2*X[:,j]
            for k,(a,b) in enumerate(((0,1),(0,2),(1,2))):
                if j==a:d[:,7+k]=X[:,b]
                if j==b:d[:,7+k]=X[:,a]
            np.testing.assert_allclose(G[j]@pol(Y),d,atol=1e-12)
        free=(Y[:,0]>.25)&(Y[:,0]<.75);grip=(X[:,0]<.25)|(X[:,0]>.75)
        for b in [T]+G:self.assertLess(np.max(abs(b[grip][:,free])),1e-12)
    def test_objectivity_affine_and_static(self):
        s,e,m,h,_=self.case;Q=CompatibleGeometry(s,e,m,h).Q
        self.assertTrue(spectrum(e.tangent(s.Y,Q))['passed'])
        t=.73;R=np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1]])
        out=e.evaluate(s.Y@R.T+np.array([.1,.2,.3]));self.assertLess(abs(out['U']),1e-12)
        F=np.array([[1.02,.01,0],[.002,.99,.003],[0,0,1.]])
        np.testing.assert_allclose(gradient(e.B,s.Y@F.T),np.broadcast_to(F,(len(m),3,3)),atol=1e-12)
    def test_force_tangent_from_energy(self):
        s,e,m,h,_=self.case;rng=np.random.default_rng(123);d=rng.normal(size=s.Y.shape);d/=la.norm(d);Y=s.Y+1e-4*rng.normal(size=s.Y.shape);eps=1e-6
        a=e.evaluate(Y);p=e.evaluate(Y+eps*d);n=e.evaluate(Y-eps*d)
        self.assertLess(abs((p['U']-n['U'])/(2*eps)-np.sum(a['force']*d)),1e-8)
        exact=e.tangent(Y)@d.T.ravel();fd=((p['force']-n['force'])/(2*eps)).T.ravel()
        self.assertLess(la.norm(exact-fd)/la.norm(exact),1e-7)
    def test_moving_joint_history(self):
        s,e,m,h,_=self.case;so=CompatibleAVF(s,e,m,h,mode='driven')
        for k in range(4):
            row=so.step(.00025)
            self.assertLess(abs(row['budget_defect_J']),1e-14);self.assertLess(row['endpoint_velocity_constraint'],1e-10)
            np.testing.assert_allclose(e.position_basis@so.state.Y,so.state.x,atol=1e-13)
            self.assertLess(row['history_commit_max'],1e-12)

if __name__=='__main__':unittest.main()
