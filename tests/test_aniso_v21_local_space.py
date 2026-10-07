import unittest
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v21_common import BASE,controlled_case,hessian
from engine.aniso_phase1 import local_reference as ref
from engine.aniso_phase1.stress_local_space import patches,patch_mask,patch_corrections,scalar_grams,reduced_solve,LocalPotential

class StressLocalSpaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s,e,_,_,_=controlled_case();cls.state=s;cls.energy=e
        with np.load(BASE/'v19/space/reconstruction16.npz') as z:cls.A=z['A'];cls.nodes=z['nodes'];cls.edges=[z[f'axis{k}'] for k in range(3)]
        with np.load(BASE/'v20/space/F45-r16-e0.npz') as z:u=z['u']
        _,K=ref.assemble(cls.edges,2,hessian('F45'));cls.K=K;cls.patches=[patches()[i] for i in (0,4,22,49)];cls.W,cls.records=patch_corrections(cls.nodes,K,u,cls.patches)
        cls.free=(s.Y[:,0]>.25)&(s.Y[:,0]<.75);cls.Q=np.eye(len(s.Y))[:,cls.free];cls.nf=cls.Q.shape[1];cls.lift=np.zeros_like(s.Y);cls.lift[s.Y[:,0]>=.75,0]=.005;V=np.column_stack((cls.A@cls.Q,cls.W,cls.A@cls.lift[:,0]));cls.G=scalar_grams(cls.edges,V);cls.patch=cls.Q.T@e.Ks@cls.Q;cls.patchlift=cls.Q.T@e.Ks@cls.lift;cls.patchconstant=.5*np.sum(cls.lift*(e.Ks@cls.lift));cls.Wgram=cls.W.T@cls.W
    @classmethod
    def solution(cls,label,ids):return reduced_solve(cls.G,hessian(label),cls.patch,cls.patchlift,cls.patchconstant,cls.nf,ids,cls.Wgram)
    def test_bounded_support_and_rotation_closed_component_space(self):
        fixed=(self.nodes[:,0]<=.25)|(self.nodes[:,0]>=.75);np.testing.assert_allclose(self.W[fixed],0,atol=0.)
        for p,r in zip(self.patches,self.records):
            np.testing.assert_allclose(self.W[~patch_mask(self.nodes,p)][:,r['columns']],0,atol=0.);self.assertLess(r['relative_residual'],1e-8)
    def test_static_no_mass_four_materials_and_nested_energy(self):
        for label in ('ISO','F0','F45','F90'):
            prev=np.inf
            for n in (0,len(self.records[0]['columns']),self.W.shape[1]):
                coef,T,r,K=self.solution(label,list(range(n)));self.assertTrue(r['static_passed'],r);self.assertLess(r['free_residual'],1e-9);self.assertLess(r['work_identity_relative'],1e-8);self.assertLessEqual(r['energy_J'],prev*(1+1e-10));prev=r['energy_J']
                if n:self.assertGreater(r['local_stiffness_min'],0);self.assertGreater(r['schur_min'],0)
    def test_solution_matches_independent_physical_energy_and_reaction(self):
        ids=list(range(self.W.shape[1]));coef,T,r,K=self.solution('F45',ids);y=self.lift+self.Q@coef[:self.nf];u=self.A@y+self.W@coef[self.nf:];f=(self.K@u.T.ravel()).reshape(3,-1).T;energy=.5*np.sum(u*f)+.5*np.sum(y*(self.energy.Ks@y));R=(self.A.T@f+self.energy.Ks@y)[self.state.Y[:,0]>=.75,0].sum();self.assertAlmostEqual(energy,r['energy_J'],places=13);self.assertAlmostEqual(R,r['reaction_N'],places=11)
    def test_rigid_affine_and_quadratic_bending_exact(self):
        x=self.state.Y;f=np.column_stack((np.ones(len(x)),x,x*x,x[:,0]*x[:,1],x[:,0]*x[:,2],x[:,1]*x[:,2]));z=self.nodes;target=np.column_stack((np.ones(len(z)),z,z*z,z[:,0]*z[:,1],z[:,0]*z[:,2],z[:,1]*z[:,2]));np.testing.assert_allclose(self.A@f,target,atol=3e-12)
        bending=np.column_stack((-x[:,0]*x[:,1],.5*x[:,0]**2,np.zeros(len(x))));X=np.array([[.28,.42,.46],[.48,.51,.57],[.72,.59,.41]]);L=ref.gradient(X,self.edges,2,self.A@bending);exact=np.zeros_like(L);exact[:,0,0]=-X[:,1];exact[:,0,1]=-X[:,0];exact[:,1,0]=X[:,0];np.testing.assert_allclose(L,exact,atol=3e-12)
    def test_same_nonlinear_potential_force_tangent_and_objectivity(self):
        Z=self.W;pot=LocalPotential(self.edges,self.A,Z,self.energy.Ks,self.energy.params,self.energy.A[0]);Y=np.vstack((self.state.Y,np.zeros((Z.shape[1],3))));rng=np.random.default_rng(55);Y+=rng.normal(scale=1e-5,size=Y.shape);d=rng.normal(size=Y.shape);d/=la.norm(d);a=pot.evaluate(Y,d);eps=1e-6;b=pot.evaluate(Y+eps*d);c=pot.evaluate(Y-eps*d);self.assertLess(abs((b['U']-c['U'])/(2*eps)-np.sum(a['force']*d)),2e-7);np.testing.assert_allclose((b['force']-c['force'])/(2*eps),a['tangent_action'],atol=3e-7,rtol=2e-5)
        R=la.expm(np.array([[0,-.6,.2],[.6,0,.1],[-.2,-.1,0]]));v=pot.evaluate(Y@R.T);self.assertAlmostEqual(a['U'],v['U'],places=12);np.testing.assert_allclose(v['force'],a['force']@R.T,atol=1e-11)
    def test_native_nonlinear_tangent_matches_independent_static_hessian(self):
        ids=list(range(self.W.shape[1]));_,T,_,K=self.solution('F45',ids);Z=self.W@T[self.nf:,self.nf:];pot=LocalPotential(self.edges,self.A,Z,self.energy.Ks,self.energy.params,self.energy.A[0]);Y=np.vstack((self.state.Y,np.zeros((Z.shape[1],3))));m=self.nf+len(ids);v=np.random.default_rng(56).normal(size=(m,3));d=np.vstack((self.Q@v[:self.nf],v[self.nf:]));action=pot.evaluate(Y,d)['tangent_action'];reduced=np.vstack((self.Q.T@action[:len(self.state.Y)],action[len(self.state.Y):]));np.testing.assert_allclose(reduced.T.ravel(),K@v.T.ravel(),atol=3e-10,rtol=2e-10)
if __name__=='__main__':unittest.main()
