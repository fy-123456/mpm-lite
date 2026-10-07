import unittest,tempfile,json,subprocess,sys
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_absolute_state_next.state import MaterialStateModel,State
from engine.aniso_phase1.research_absolute_state_next.dynamics import DryMidpoint

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=MaterialStateModel()
    def test_absolute_fields_match_original_tensor(self):
        m=self.m;r=m.r;s=m.space;state=m.rest();rng=np.random.default_rng(31)
        state.q[r.free]=rng.normal(size=(len(r.free),3))*1e-7
        axes=[np.array([.261,.493,.721]),np.array([.401,.537]),np.array([.437,.561])]
        X=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
        x,F,v=m.fields(state,X);u,g=s._sample(s.nodes(r.expand(state.q)),axes)
        np.testing.assert_allclose(x,X+u.reshape(-1,3),atol=1e-12,rtol=1e-9)
        np.testing.assert_allclose(F,np.eye(3)+g.reshape(-1,3,3),atol=1e-11,rtol=1e-8)
    def test_material_derivative_from_shadow_positions(self):
        m=self.m;state=m.rest();rng=np.random.default_rng(32);state.q[m.r.free]=rng.normal(size=(len(m.r.free),3))*1e-7
        X=rng.uniform([.26,.39,.39],[.74,.61,.61],(15,3));eps=1e-6
        F=m.fields(state,X)[1]
        fd=np.stack([(m.fields(state,X+np.eye(3)[a]*eps)[0]-m.fields(state,X-np.eye(3)[a]*eps)[0])/(2*eps) for a in range(3)],axis=-1)
        np.testing.assert_allclose(F,fd,rtol=1e-5,atol=1e-7)
    def test_corrected_transfer_after_grid_crossing(self):
        m=self.m;state=m.rest();full=np.zeros((m.space.ndof,3));full[:m.space.n,0]=.03;state.q=m.r.project(full)
        X=np.array([[.26,.4,.43],[.51,.54,.58],[.72,.59,.48]])
        x,_,_=m.fields(state,X);self.assertTrue(np.any(np.floor(x*64)!=np.floor(X*64)))
        inc=np.zeros_like(state.q);inc[m.r.free]=np.random.default_rng(33).normal(size=(len(m.r.free),3))*1e-7
        t=m.corrected_transfer(state,X,inc)
        np.testing.assert_allclose(t['displacement'],t['exact'],atol=1e-12)
        np.testing.assert_allclose(t['material_gradient'],t['exact_material_gradient'],atol=1e-11)
        self.assertGreater(np.linalg.norm(t['raw']-t['exact']),1e-12)
    def test_checkpoint_foreign_and_corruption(self):
        m=self.m;s=m.rest()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'state.npz';m.save(s,p);self.assertEqual(m.load(p).digest(),s.digest())
            with np.load(p) as z:d={k:z[k].copy() for k in z.files}
            meta=json.loads(str(d['metadata']));meta['identity']['schema']='foreign';d['metadata']=np.array(json.dumps(meta));np.savez(p,**d)
            with self.assertRaisesRegex(ValueError,'foreign'):m.load(p)
            m.save(s,p)
            with np.load(p) as z:d={k:z[k].copy() for k in z.files}
            d['q'][0,0]+=1e-7;np.savez(p,**d)
            with self.assertRaisesRegex(ValueError,'digest'):m.load(p)
    def test_trial_and_failure_leave_input_unchanged(self):
        m=self.m;s=m.rest();old=s.digest();inc=np.ones_like(s.q)*1e-8
        with self.assertRaises(RuntimeError):m.trial(s,inc,inc,.001,fault=True)
        self.assertEqual(old,s.digest());n=m.trial(s,inc,inc,.001);self.assertEqual(old,s.digest());self.assertEqual(n.step,1)
        with self.assertRaises(ValueError):m.trial(s,inc,inc,0)
    def test_full_boundary_cross_mass_and_ledger(self):
        m=self.m;K=m.r.K
        def potential(q):return dict(U=.5*float(q.ravel()@K@q.ravel()),force=(K@q.ravel()).reshape(q.shape),min_detF=1.)
        solver=DryMidpoint(m,potential);s=m.rest();target=np.zeros(len(solver.fixed));target[::3]=1e-7
        before=s.digest();n,r=solver.step(s,.001,target)
        np.testing.assert_allclose(n.q.ravel()[solver.fixed],target,atol=1e-15)
        self.assertLess(abs(r['energy_defect']),1e-12);self.assertEqual(s.digest(),before)
        with self.assertRaises(RuntimeError):solver.step(s,.001,target,fault=True)
        self.assertEqual(s.digest(),before)
    def test_absolute_force_and_tangent(self):
        m=self.m;rng=np.random.default_rng(35);q=rng.normal(size=m.rest().q.shape)*1e-4;d=rng.normal(size=q.shape)
        U,f,Kd=m.stabilization(q,d);eps=1e-5;Ep,fp,_=m.stabilization(q+eps*d);Em,fm,_=m.stabilization(q-eps*d)
        np.testing.assert_allclose((Ep-Em)/(2*eps),np.sum(f*d),rtol=1e-3,atol=1e-9)
        np.testing.assert_allclose((fp-fm)/(2*eps),Kd,rtol=1e-6,atol=1e-9)
    def test_bad_state_and_domain_rejected(self):
        m=self.m;s=m.rest();s.q[0,0]=np.nan
        with self.assertRaises(ValueError):m.fields(s,np.array([[.5,.5,.5]]))
        with self.assertRaises(ValueError):m.fields(m.rest(),np.array([[2.,.5,.5]]))
        s=m.rest();s.time=-1
        with self.assertRaises(ValueError):m.validate(s)

if __name__=='__main__':unittest.main()
