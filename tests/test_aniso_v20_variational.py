import unittest
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v20_common import BASE,controlled_case
from benchmarks.aniso_v20_modes import gram
from engine.aniso_phase1.integrated_avf import IntegratedAVF,material_null_condensation,boundary_impulse
from engine.aniso_phase1.compatible_carrier import CompatibleReconstruction,shape_matrices
from engine.aniso_phase1.grip_enrichment import local_basis,EnrichedPotential,static_solve
from engine.aniso_phase1.swept_quadrature import sweep,line_breaks
from benchmarks.aniso_boundary_reference import hessian

class VariationalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s,cls.e,cls.m,cls.h,_=controlled_case();r=CompatibleReconstruction.__new__(CompatibleReconstruction);r.nodes=cls.s.Y.copy();r.h=cls.h;r.resolution=16
        with np.load(BASE/'v19/space/reconstruction16.npz') as z:r.A=z['A'];r.fe_nodes=z['nodes'];r.edges=[z[f'axis{k}'] for k in range(3)]
        cls.rec=r
    def test_static_elimination_preserves_original_reaction(self):
        s,e,m,h=self.s,self.e,self.m,self.h;H,R,Z,C=material_null_condensation(e,s.Y,h);e.carrier_reference=s.Y.copy();so=IntegratedAVF(s,e,m,h,condense=True)
        Q=np.eye(e.n)[:,(s.Y[:,0]>.25)&(s.Y[:,0]<.75)];K=e.tangent(s.Y);lift=np.zeros_like(s.Y);lift[s.Y[:,0]>=.75,0]=.005;vals=[]
        for q,lift_ in ((Q,lift),(so.geometry.Q,R@lift)):
            B=la.block_diag(q,q,q);u=lift_.T.ravel();u+=B@la.solve(B.T@K@B,-B.T@K@u,assume_a='pos');vals.append((.5*u@K@u,u))
        np.testing.assert_allclose(vals[0][0],vals[1][0],rtol=1e-11)
        np.testing.assert_allclose(vals[0][1],vals[1][1],atol=2e-12)
    def test_condensed_impulse_preserves_physical_momentum_duality(self):
        s,e,m,h=self.s.clone(),self.e,self.m,self.h;e.carrier_reference=s.Y.copy();so=IntegratedAVF(s,e,m,h,condense=True);g=so.geometry;z=np.random.default_rng(6).normal(scale=1e-4,size=(4*len(m),3));_,d=boundary_impulse(g,z,0.,h)
        np.testing.assert_allclose(d['impulse'].sum(0),m@d['delta'][:len(m)],atol=1e-13)
        torque=np.sum(np.cross(s.x,m[:,None]*d['delta'][:len(m)]),axis=0)
        for j in range(3):torque+=np.sum(np.cross(np.eye(3)[j],g.metric[(j+1)*len(m):(j+2)*len(m),None]*d['delta'][(j+1)*len(m):(j+2)*len(m)]),axis=0)
        np.testing.assert_allclose(np.sum(np.cross(s.Y,d['impulse']),axis=0),torque,atol=1e-13);self.assertLessEqual(d['energy_change_J'],1e-20)
    def test_enrichment_vanishes_inside_grips_and_keeps_polynomials(self):
        r=self.rec;Z=local_basis(r.fe_nodes,3);fixed=(r.fe_nodes[:,0]<=.25)|(r.fe_nodes[:,0]>=.75);np.testing.assert_allclose(Z[fixed],0,atol=1e-15)
        x=r.nodes;f=np.column_stack((np.ones(len(x)),x,x*x,x[:,0]*x[:,1],x[:,0]*x[:,2],x[:,1]*x[:,2]));X=r.fe_nodes;exact=np.column_stack((np.ones(len(X)),X,X*X,X[:,0]*X[:,1],X[:,0]*X[:,2],X[:,1]*X[:,2]));np.testing.assert_allclose(r.A@f,exact,atol=2e-12)
    def test_local_energy_force_and_tangent_share_potential(self):
        r=self.rec;e=self.e;pot=EnrichedPotential(r,e.Ks,1,e.params,e.A[0]);rng=np.random.default_rng(7);Y=np.vstack((r.nodes,np.zeros((6,3))))+rng.normal(scale=1e-4,size=(len(r.nodes)+6,3));d=rng.normal(size=Y.shape);d/=la.norm(d);a=pot.evaluate(Y,d);eps=2e-6;plus=pot.evaluate(Y+eps*d);minus=pot.evaluate(Y-eps*d)
        self.assertLess(abs((plus['U']-minus['U'])/(2*eps)-np.sum(a['force']*d)),1e-7)
        np.testing.assert_allclose((plus['force']-minus['force'])/(2*eps),a['tangent_action'],rtol=2e-5,atol=2e-7)
    def test_nonlinear_rigid_rotation_objectivity_with_local_modes(self):
        r=self.rec;e=self.e;pot=EnrichedPotential(r,e.Ks,1,e.params,e.A[0]);Y=np.vstack((r.nodes,np.zeros((6,3))));rng=np.random.default_rng(8);Y+=rng.normal(scale=1e-4,size=Y.shape);Q=la.expm(np.array([[0.,-.7,.1],[.7,0.,-.2],[-.1,.2,0.]]));a=pot.evaluate(Y);b=pot.evaluate(Y@Q.T)
        self.assertLess(abs(a['U']-b['U']),1e-13);np.testing.assert_allclose(b['force'],a['force']@Q.T,atol=3e-12)
    def test_positive_enriched_static_and_nested_energy(self):
        last=np.inf
        for level in range(4):
            *_,r=static_solve(self.rec,self.e.Ks,hessian('F45'),level);self.assertTrue(r['static_passed']);self.assertLessEqual(r['energy_J'],last*(1+1e-10));last=r['energy_J']
    def test_swept_quadrature_positive_volume_and_polynomial_moments(self):
        edges=[np.array([0.,.25])]*3;A=np.array([[1.,.21,0.],[0.,1.,.13],[.08,0.,1.]])
        for order in (3,4):
            X,V,info=sweep(edges,lambda X:X@A.T,.125,order);self.assertTrue(np.all(V>0));self.assertAlmostEqual(V.sum(),.25**3,places=13)
            np.testing.assert_allclose(V@X,np.ones(3)*.25**4/2,atol=1e-12);np.testing.assert_allclose(V@(X*X),np.ones(3)*.25**5/3,atol=1e-12)
    def test_line_cuts_track_all_physical_coordinates(self):
        ends=np.array([[[0.,.1,.0],[.8,-.2,.4]]]);cuts=line_breaks(0.,1.,ends,.125)
        for l,r in zip(cuts[:-1],cuts[1:]):
            t=np.array([l+1e-8*(r-l),r-1e-8*(r-l)]);x=ends[0,0]+t[:,None]*(ends[0,1]-ends[0,0]);np.testing.assert_array_equal(np.floor(x[0]/.125-.5),np.floor(x[1]/.125-.5))
    def test_reaction_gram_cross_terms_and_degenerate_rotation(self):
        omega=np.array([2.,2.,7.]);T=.83;c=np.array([.3,-.1,.6]);s=np.array([.4,.7,.2]);cc,ss,cs=gram(omega,T);value=c@cc@c+s@ss@s+2*c@cs@s;t=np.linspace(0,T,100001);signal=c@np.cos(omega[:,None]*t)+s@np.sin(omega[:,None]*t)
        self.assertAlmostEqual(value,np.trapezoid(signal**2,t)/T,places=8)
        # Equal-frequency amplitudes must combine coherently, not sum powers.
        self.assertAlmostEqual(cc[0,0]*sum(c[:2])**2+ss[0,0]*sum(s[:2])**2+2*cs[0,0]*sum(c[:2])*sum(s[:2]),c[:2]@cc[:2,:2]@c[:2]+s[:2]@ss[:2,:2]@s[:2]+2*c[:2]@cs[:2,:2]@s[:2],places=13)
if __name__=='__main__':unittest.main()
