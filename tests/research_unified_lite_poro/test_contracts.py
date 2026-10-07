"""Physical and isolation tests of the new scoped bridge."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.linalg import eigvalsh
from engine.aniso_phase1.research_unified_lite_poro.space import Space
from engine.aniso_phase1.research_unified_lite_poro.model import Bridge,FrozenOperator,unload
from engine.aniso_phase1.research_unified_lite_poro import checkpoint
from engine.aniso_phase1.material_snapshot import response

class Contracts(unittest.TestCase):
    def setUp(self):
        self.b=Bridge(Space());self.op,_,_,_=self.b.prepare()

    def test_mass_cross_terms_and_boundary(self):
        self.assertGreater(eigvalsh(self.op.M)[0],0)
        self.assertGreater(np.linalg.norm(self.op.M[:-2,-2:]),0)
        X=np.array([[0,.02,.04],[0,.23,.2]])
        N,D=self.b.space.basis(X);self.assertLess(np.max(abs(N)),1e-14)

    def test_particle_budgets_and_hermite_rank(self):
        counts=[]
        for ppc in (2,3,4):
            b=Bridge(Space(),ppc);op,q,v,fit=b.prepare()
            counts.append(len(op.points));self.assertEqual(fit['rank'],len(q))
            self.assertAlmostEqual(op.weights.sum(),.0625)
        self.assertEqual(counts,[32,32,32])

    def test_material_energy_derivative_and_tangent_symmetry(self):
        rng=np.random.default_rng(12);q=rng.normal(size=self.b.state.q.shape)*1e-4
        a=rng.normal(size=q.shape);a/=np.linalg.norm(a)
        b=rng.normal(size=q.shape);b/=np.linalg.norm(b)
        h=1e-6;E,g,_=self.op.material(q)
        fd=(self.op.material(q+h*a)[0]-self.op.material(q-h*a)[0])/(2*h)
        self.assertAlmostEqual(fd,float(np.sum(g*a)),delta=1e-7)
        Ka=(self.op.material(q+h*a)[1]-self.op.material(q-h*a)[1])/(2*h)
        Kb=(self.op.material(q+h*b)[1]-self.op.material(q-h*b)[1])/(2*h)
        self.assertAlmostEqual(float(np.sum(b*Ka)),float(np.sum(a*Kb)),delta=1e-5)

    def test_volume_chain_and_darcy_spd(self):
        rng=np.random.default_rng(5);q=rng.normal(size=self.b.state.q.shape)*2e-4;q0=q*.2
        V,G,H,_=self.op.geometry(q);V0=self.op.geometry(q0,False)[0]
        chain=V-V0-np.einsum('cni,ni->c',self.op.discrete_G(q0,q),q-q0)
        self.assertLess(np.max(abs(chain)),1e-13)
        self.assertGreater(eigvalsh(H)[0],0)
        self.assertLess(np.max(abs(self.op.top.B[:,self.op.top.internal].sum(axis=0))),1e-14)

    def test_objectivity_of_inherited_material(self):
        F=np.tile(np.eye(3),(len(self.op.points),1,1));F[:,0,0]=1.02
        a=.3;R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
        E,P,_=response(F,self.op.A2,self.op.A4,self.op.params)
        Er,Pr,_=response(R@F,self.op.A2,self.op.A4,self.op.params)
        self.assertLess(np.max(abs(E-Er)),1e-11)
        self.assertLess(np.max(abs(Pr-R@P)),1e-10)

    def test_real_particle_feedback_and_cross_reference_location(self):
        s=self.b.space;p=self.b.state.particles.copy()
        N,D=s.basis(p.X);q=np.zeros_like(self.b.state.q)
        q[:-2,1]=s.nodes[s.free[:-2],0]*.3
        p.x+=N@q;p.F+=np.einsum('qnd,ni->qid',D,q)
        fitted,_,_=unload(s,p)
        self.assertLess(np.max(abs(fitted-q)),1e-12)
        self.assertTrue(np.any(p.x[:,1]>s.lengths[1]))
        self.b.state.particles=p
        with self.assertRaisesRegex(ValueError,'incompatible'):self.b.prepare()
        # This exercises material-coordinate motion only, not Eulerian cell transfer.

    def test_owned_frozen_package_ignores_later_particle_changes(self):
        q=self.b.state.q.copy();q[0,0]=.0001
        E,g,_=self.op.material(q)
        self.b.state.particles.F[:]=np.nan
        E1,g1,_=self.op.material(q)
        self.assertEqual(E,E1);np.testing.assert_array_equal(g,g1)
        with self.assertRaises(ValueError):self.op.weights[0]=0
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'frozen.npz';self.op.save(path)
            loaded=FrozenOperator.load_package(path)
            np.testing.assert_allclose(loaded.material(q)[1],g,atol=1e-13)

    def test_whole_step_rollback_retry_and_checkpoint(self):
        before=self.b.state.digest()
        with self.assertRaisesRegex(RuntimeError,'injected'):self.b.step(fault=True)
        self.assertEqual(before,self.b.state.digest())
        control=Bridge(Space());self.b.step();control.step()
        self.assertEqual(self.b.state.digest(),control.state.digest())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.npz';checkpoint.save(self.b,path)
            recovered=checkpoint.load(path,checkpoint.config(self.b))
            self.assertEqual(recovered.state.digest(),self.b.state.digest())
            with self.assertRaisesRegex(ValueError,'identity'):checkpoint.load(path,{'foreign':True})

    def test_invalid_step_preserves_state(self):
        digest=self.b.state.digest()
        with self.assertRaises(ValueError):self.b.step(h=-1)
        self.assertEqual(digest,self.b.state.digest())


class FixedPositiveContracts(unittest.TestCase):
    def test_fixed_budget_positive_and_order_invariant(self):
        for ppc in (2,3,4):
            b=Bridge(Space(),ppc,rule='fixed-positive');op,_,_,_=b.prepare()
            self.assertEqual(len(op.points),128)
            self.assertGreaterEqual(np.linalg.eigvalsh(op.A4).min(),-1e-12)
            self.assertLess(np.max(abs(np.einsum('qij,j->qi',op.A4,np.eye(3).ravel()).reshape(-1,3,3)-op.A2)),1e-12)
            p=b.state.particles.copy();ix=np.arange(len(p.X))[::-1]
            for key in p.__dataclass_fields__:setattr(p,key,getattr(p,key)[ix].copy())
            other=FrozenOperator.fixed_positive(b.space,p)
            np.testing.assert_allclose(other.A4,op.A4,atol=1e-14)

    def test_reject_sparse_template_and_bad_package(self):
        b=Bridge(Space(),rule='fixed-positive');p=b.state.particles.copy()
        for key in p.__dataclass_fields__:setattr(p,key,getattr(p,key)[:-1])
        with self.assertRaises(ValueError):FrozenOperator.fixed_positive(b.space,p)
        op,_,_,_=b.prepare()
        with self.assertRaises(ValueError):FrozenOperator(b.space,op.points,-op.weights,op.A2,op.A4)

    def test_selected_checkpoint_roundtrip(self):
        b=Bridge(Space(),rule='fixed-positive')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'selected.npz';checkpoint.save(b,path);new=checkpoint.load(path)
            self.assertEqual(new.rule,'fixed-positive')
            self.assertEqual(new.prepare()[0].identity,b.prepare()[0].identity)

if __name__=="__main__":unittest.main()
