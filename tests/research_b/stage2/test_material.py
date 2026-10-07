import dataclasses
from pathlib import Path
import tempfile
import unittest
import numpy as np
from tests.research_d.test_common_space import small_package
from engine.aniso_phase1.research_d.common_space import CommonSpace
from engine.aniso_phase1.research_d.common_state import CommonState, StateTransaction
from engine.aniso_phase1.research_b.tensor import TensorRule, TensorMaterialOperator
from engine.aniso_phase1.consistent_transfer import material_response
from engine.aniso_phase1.history_increment import material_tangent
from engine.aniso_phase1.types import AnisotropicMaterialParams
from engine.aniso_phase1.research_b.stage2.material import MaterialSource, ConstitutiveState
from engine.aniso_phase1.research_b.stage2.operator import CommonMaterialOperator, FixedRule
from engine.aniso_phase1.research_b.stage2.transaction import MaterialSession, validate_material_child


class MaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        small_package(Path(cls.tmp.name))
        cls.space = CommonSpace(cls.tmp.name)
        cls.source = MaterialSource(space_sha256=cls.space.signature)
        cls.op = CommonMaterialOperator(cls.space, FixedRule.uniform(cls.space, 6), cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_original_operator_and_stabilization(self):
        s = self.space; q = s.expand(s.q0); d = s.direction_coefficients(s.test_vectors['direction'])
        old = TensorMaterialOperator(s, TensorRule.uniform(s.edges, 6)).evaluate(q, d)
        new = self.op.evaluate_full(q, directions={'d': d})
        for k, j in [('energy_J','U'), ('material_energy_J','material_U'), ('stabilization_energy_J','stabilization_U'), ('full_force','force'), ('material_force','material_force'), ('weak_moments','weak_moments')]:
            np.testing.assert_allclose(new[k], old[j], rtol=2e-9, atol=1e-10)
        np.testing.assert_allclose(new['tangents']['d']['full'], old['tangent_action'], rtol=2e-9, atol=1e-9)
        np.testing.assert_array_equal(new['stabilization_force'][:s.n], s.Ks @ (s.reference[:s.n]+q[:s.n]))

    def test_explicit_lift_derivative_and_adjoint(self):
        s = self.space; q = s.q0.copy(); d = s.test_vectors['direction']; lift = s.lift*.7
        base = self.op.evaluate_free(q, lift=lift, directions={'d': d})
        errors = []
        for h in (3e-5, 1e-5, 3e-6):
            a = self.op.evaluate_free(q+h*d, lift=lift); b = self.op.evaluate_free(q-h*d, lift=lift)
            ge = abs((a['energy_J']-b['energy_J'])/(2*h)-np.sum(base['free_force']*d))
            he = np.linalg.norm((a['free_force']-b['free_force'])/(2*h)-base['tangents']['d']['free'])
            errors.append((ge, he/max(np.linalg.norm(base['tangents']['d']['free']), 1e-8)))
        self.assertLess(max(e[0] for e in errors[-2:]), 1e-6)
        self.assertLess(max(e[1] for e in errors[-2:]), 1e-5)
        full = s.expand(q, lift); v = np.random.default_rng(1).normal(size=(np.prod(s.shape),3))
        np.testing.assert_allclose(np.sum(s.nodes(full)*v), np.sum(full*s.adjoint(v)), atol=1e-10)
        self.assertGreater(np.linalg.norm(base['full_force'][s.fixed_scalar_ids]), 0)
        with self.assertRaises(ValueError): self.op.evaluate_free(q, lift=None)
        with self.assertRaises(ValueError): self.op.evaluate_full(q)

    def test_mixed_constitutive_independent_weighted_families(self):
        rng = np.random.default_rng(4); F = np.eye(3)+rng.normal(scale=.12,size=(11,3,3)); X=rng.uniform(.2,.7,(11,3)); d=rng.normal(size=F.shape)
        for field in ('F45','partition','turning','two_family_3d'):
            source=dataclasses.replace(self.source,field=field)
            actual=ConstitutiveState(F,X,source,tangent=True)
            energy=np.zeros(len(F)); P=np.zeros_like(F); H=np.zeros_like(F)
            for w,a in source.fibers(X):
                params=AnisotropicMaterialParams(source.mu,source.lam,source.k_f,[1.,0.,0.])
                A=a[:,:,None]*a[:,None,:]
                e,p=material_response(F,A,params)
                energy+=w*e;P+=w*p;H+=w*material_tangent(F,A,d,params)
                np.testing.assert_allclose(np.linalg.norm(a,axis=1),1.,atol=1e-14)
            np.testing.assert_allclose(actual.energy,energy,atol=1e-11,rtol=1e-9)
            np.testing.assert_allclose(actual.P,P,atol=1e-10,rtol=1e-9)
            np.testing.assert_allclose(actual.action(d),H,atol=1e-9,rtol=1e-9)

    def test_physical_partition_split_and_volume(self):
        source=dataclasses.replace(self.source,field='partition',split_x=.43)
        op=CommonMaterialOperator(self.space,FixedRule.uniform(self.space,4),source)
        volume=sum(float(V.sum()) for _,_,V,_ in op._slabs)
        self.assertAlmostEqual(volume,np.prod([e[-1]-e[0] for e in self.space.edges]),places=13)
        for _,X,_,_ in op._slabs:
            self.assertFalse(np.any(X[:,0]==.43))
        q=np.zeros((self.space.ndof,3));result=op.evaluate_full(q)
        self.assertLess(abs(result['material_energy_J']),1e-20)
        self.assertLess(np.linalg.norm(result['material_force']),1e-11)

    def test_identity_rejection_even_at_zero_strain(self):
        for changes in ({'field':'turning'},{'angle':.7},{'mu':11.},{'split_x':.51}):
            wrong=dataclasses.replace(self.source,**changes)
            with self.assertRaises(ValueError): self.op.check_identity(source_sha256=wrong.signature)
        with self.assertRaises(ValueError): self.op.check_identity(space_sha256='0'*64)
        with self.assertRaises(ValueError): self.op.check_identity(cache_sha256='0'*64)
        with self.assertRaises(ValueError): CommonMaterialOperator(self.space,self.op.rule,MaterialSource())
        with self.assertRaises(ValueError): FixedRule((True,))

    def test_fallback_is_same_source_full_on_selected_slabs(self):
        full=FixedRule.uniform(self.space,6);low=FixedRule.uniform(self.space,4)
        all_full=low.fallback(full,range(len(full.orders)))
        self.assertEqual(all_full.signature,full.signature)
        q=self.space.expand(self.space.q0)
        a=CommonMaterialOperator(self.space,all_full,self.source).evaluate_full(q)
        b=self.op.evaluate_full(q)
        np.testing.assert_array_equal(a['full_force'],b['full_force'])
        with self.assertRaises(ValueError): low.fallback(full,[-1])

    def test_illegal_states_and_nonfinite_directions(self):
        q=np.zeros((self.space.ndof,3));q[:self.space.n,0]=-2*self.space.carrier_X[:,0]
        with self.assertRaises(ValueError): self.op.evaluate_full(q)
        q[:]=np.nan
        with self.assertRaises(ValueError): self.op.evaluate_full(q)
        with self.assertRaises(ValueError): self.op.evaluate_full(q.astype(complex))

    def test_whole_state_reject_stale_and_exactly_once_commit(self):
        s=self.space; initial=CommonState(s.expand(s.q0),np.zeros((s.ndof,3)),predictor=np.ones((s.ndof,3)),child_states={'ledger':{'work':1.},'cache':'original'})
        tx=StateTransaction(initial);session=MaterialSession(self.op,tx);before=tx.snapshot().digest()
        trial=tx.trial();proposal,_=session.prepare(trial);session.attach(trial,proposal)
        trial.state.child_states['ledger']['work']=3.;session.reject(trial)
        self.assertEqual(before,tx.snapshot().digest())
        with self.assertRaises(ValueError):session.attach(trial,proposal)
        trial=tx.trial();proposal,_=session.prepare(trial);trial.state.q[0,0]+=.01
        with self.assertRaises(ValueError):session.attach(trial,proposal)
        session.reject(trial)
        with self.assertRaises(ValueError):session.switch_rule(FixedRule.uniform(s,7))
        sibling=tx.trial();trial=tx.trial();proposal,_=session.prepare(trial);session.attach(trial,proposal)
        trial.state.step+=1;trial.state.time+=.001
        validate_material_child(trial.state,self.op);tx.commit(trial)
        self.assertEqual(tx.revision,1)
        with self.assertRaises(ValueError):tx.commit(trial)
        with self.assertRaises(ValueError):session.prepare(sibling)

    def test_postvalidation_and_cache_failure_leave_every_value(self):
        s=self.space; initial=CommonState(s.expand(s.q0),np.zeros((s.ndof,3)),child_states={'ledger':{'work':7.}})
        def reject_after_preparation(state):
            if state.step: validate_material_child(state,self.op);raise ValueError('post-validation failure')
        tx=StateTransaction(initial,validator=reject_after_preparation);session=MaterialSession(self.op,tx);before=tx.snapshot().digest()
        trial=tx.trial();proposal,_=session.prepare(trial);session.attach(trial,proposal)
        trial.state.step=1;trial.state.time=.1
        with self.assertRaises(ValueError):tx.commit(trial)
        self.assertEqual(before,tx.snapshot().digest())
        trial=tx.trial();proposal,_=session.prepare(trial)
        with self.assertRaises(ValueError):session.attach(trial,dataclasses.replace(proposal,cache_sha256='0'*64))
        self.assertEqual(before,tx.snapshot().digest())
        session.reject(trial)


if __name__=='__main__':unittest.main()
