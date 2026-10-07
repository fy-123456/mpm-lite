import unittest,tempfile
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_absolute_state_next.state import MaterialStateModel
from engine.aniso_phase1.research_formal_pressure_next.geometry import Geometry
from engine.aniso_phase1.research_formal_pressure_next.state import State,save,load
from engine.aniso_phase1.research_formal_pressure_next.solver import Solver

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=MaterialStateModel()
    def test_affine_volume_and_translation(self):
        m=self.m;s=m.space;g=Geometry(m)
        F=np.array([[.99,.004,0.],[0.,1.01,.002],[0.,0.,1.]])
        full=np.zeros((s.ndof,3));full[:s.n]=s.reference[:s.n]@(F-np.eye(3)).T+np.array([.02,-.01,.003])
        q=m.r.project(full);out=g.evaluate(q)
        np.testing.assert_allclose(out['V'],g.reference_volume*np.linalg.det(F),rtol=1e-7,atol=1e-10)
    def test_checkpoint_all_ledgers_and_identity(self):
        m=self.m;q=m.rest().q;s=State(q,q.copy(),np.array([.03,.02]),.01,4,.002,.004,-.001,7,.009)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.npz';save(s,p,{'formal':'test'});r=load(p,m,{'formal':'test'});self.assertEqual(s.digest(),r.digest())
            with self.assertRaises(ValueError):load(p,m,{'formal':'foreign'})
            with np.load(p,allow_pickle=False) as z:data={k:z[k] for k in z.files}
            data['p']=data['p']+.1;np.savez_compressed(p,**data)
            with self.assertRaises(ValueError):load(p,m,{'formal':'test'})
    def test_directional_cofactor_formula(self):
        from engine.aniso_phase1.research_formal_pressure_next.geometry import cofactor_column
        F=np.array([[1.,.1,0.],[.02,.95,.03],[.01,0.,1.02]])
        for a in range(3):np.testing.assert_allclose(cofactor_column(F,a),np.linalg.det(F)*np.linalg.inv(F).T[:,a],rtol=1e-12,atol=1e-12)
    def test_transaction_rejection_and_fault_control(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from engine.aniso_phase1.research_formal_pressure_next.transaction import Transaction
        m=self.m;old=State(m.rest().q,m.rest().q,np.zeros(2));digest=old.digest()
        tx=Transaction(m,None,primary=2,reserve=7)
        tx.primary=SimpleNamespace(order=2,evaluate=lambda q:dict(force=np.ones(3)*2,weak_moments=np.ones(3)*2,U=0.))
        tx.reserve=SimpleNamespace(order=7,evaluate=lambda q:dict(force=np.ones(3),weak_moments=np.ones(3),U=0.))
        class FakeSolver:
            def __init__(self,model,geometry,op,**kw):self.op=op
            def step(self,s,h,target):return State(s.q+self.op.order*1e-7,s.velocity.copy(),s.p.copy(),h,1,rule=self.op.order),{}
        with patch('engine.aniso_phase1.research_formal_pressure_next.transaction.Solver',FakeSolver):
            trial,row=tx.step(old,.001,np.zeros(1));self.assertTrue(row['fallback']);self.assertEqual(trial.rule,7)
            np.testing.assert_array_equal(trial.q,old.q+7e-7)
            with self.assertRaisesRegex(RuntimeError,'injected'):tx.step(old,.001,np.zeros(1),fault=True)
        self.assertEqual(old.digest(),digest)

    def test_mixed_work_and_rollback_in_affine_geometry(self):
        # Independent linear-volume/quadratic-energy model: exact discrete ledger.
        m=self.m;shape=m.rest().q.shape;rng=np.random.default_rng(14)
        class LinearGeometry:
            order=1;reference_volume=np.array([.0234375]*2)
            G=rng.normal(size=(2,*shape))*1e-3
            def evaluate(self,q,direction=None):
                out=dict(V=self.reference_volume+np.einsum('cni,ni->c',self.G,q),G=self.G,min_boundary_J=1.)
                if direction is not None:out['dG']=self.G*0
                return out
            def chain(self,q0,q1,direction=None):
                out=dict(V0=self.evaluate(q0)['V'],V1=self.evaluate(q1)['V'],G=self.G,min_boundary_J=1.)
                if direction is not None:out['dG']=self.G*0
                return out
        class Quadratic:
            order=7;calls=0;deadline=float('inf')
            def evaluate(self,q,direction=None):
                self.calls+=1;f=(m.r.K@q.ravel()).reshape(shape);U=.5*float(np.sum(q*f))
                out=dict(U=U,force=f,min_detF=1.,material_U=U,stabilization_U=0.)
                if direction is not None:out['tangent_action']=(m.r.K@direction.ravel()).reshape(shape)
                return out
        sol=Solver(m,LinearGeometry(),Quadratic(),drained=True);old=sol.rest();target=np.full(len(sol.fixed),1e-7)
        nxt,row=sol.step(old,.0025,target);self.assertLess(abs(row['energy_defect']),1e-11)
        self.assertLess(np.linalg.norm(row['mass_defect']),1e-11);self.assertGreaterEqual(row['darcy_dissipation'],0.)
        digest=old.digest()
        with self.assertRaisesRegex(RuntimeError,'injected'):sol.step(old,.0025,target,fault=True)
        self.assertEqual(old.digest(),digest);self.assertEqual(nxt.rule_work,old.rule_work)
        with self.assertRaises(ValueError):sol.step(old,0.,target)

if __name__=='__main__':unittest.main()
