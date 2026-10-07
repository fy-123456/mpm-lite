import unittest,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from engine.aniso_phase1.research_absolute_state_next.state import MaterialStateModel
from engine.aniso_phase1.research_formal_pressure_next.state import State,save,load
from engine.aniso_phase1.research_formal_pressure_next.solver import Solver as OldSolver
from engine.aniso_phase1.research_material_coupling_next.solver import Solver
from engine.aniso_phase1.research_material_coupling_next.budget import Budget,BudgetExceeded
from engine.aniso_phase1.research_material_coupling_next.driver import advance_series

class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=MaterialStateModel()
    def fixtures(self):
        m=self.m;shape=m.rest().q.shape;budget=Budget();rng=np.random.default_rng(11)
        class G:
            order=9;reference_volume=np.array([.0234375]*2)
            def __init__(self):self.budget=budget;self.G=rng.normal(size=(2,*shape))*1e-3
            def evaluate(self,q,direction=None):
                z=dict(V=self.reference_volume+np.einsum('cni,ni->c',self.G,q),G=self.G,min_boundary_J=1.)
                if direction is not None:z['dG']=self.G*0
                return z
            def chain(self,q0,q1,direction=None):
                z=dict(V0=self.evaluate(q0)['V'],V1=self.evaluate(q1)['V'],G=self.G,min_boundary_J=1.)
                if direction is not None:z['dG']=self.G*0
                return z
        class Q:
            order=7;calls=0;deadline=float('inf')
            def __init__(self):self.budget=budget
            def evaluate(self,q,direction=None):
                self.calls+=1;f=(m.r.K@q.ravel()).reshape(shape);U=.5*float(np.sum(q*f));z=dict(U=U,force=f,min_detF=1.,material_U=U,stabilization_U=0.)
                if direction is not None:z['tangent_action']=(m.r.K@direction.ravel()).reshape(shape)
                return z
        return G(),Q()
    def test_shared_budget_counts_and_clock(self):
        clock=[0.];b=Budget(seconds=10,material_limit=2,clock=lambda:clock[0])
        for label in ['primary','reserve']:
            t=b.begin('material',label);b.end('material',label,t,True)
        with self.assertRaises(BudgetExceeded):b.begin('material','old-rule-energy')
        t=b.begin('geometry','jv');clock[0]=11
        with self.assertRaises(BudgetExceeded):b.end('geometry','jv',t,True)
    def test_unit_resistance_matches_parent(self):
        g,op=self.fixtures();new=Solver(self.m,g,op,drained=True);old=OldSolver(self.m,g,op,drained=True)
        state=new.rest();target=np.full(len(new.fixed),1e-7)
        a,ar=new.step(state,.0025,target);b,br=old.step(state,.0025,target)
        np.testing.assert_allclose(a.q,b.q,atol=1e-12);np.testing.assert_allclose(a.p,b.p,atol=1e-12)
        self.assertAlmostEqual(ar['darcy_dissipation'],br['darcy_dissipation'])
    def test_nonunit_H_flux_mass_energy_and_jvp(self):
        g,op=self.fixtures();H=np.array([[2.,.1,0.],[.1,3.,.2],[0.,.2,4.]])
        s=Solver(self.m,g,op,drained=True,resistance=H);old=s.rest();target=np.full(len(s.fixed),1e-7)
        new,row=s.step(old,.0025,target)
        expected=np.linalg.solve(H,s.Dphysical.T@((old.p+new.p)/2));np.testing.assert_allclose(row['flux'],expected,atol=1e-12)
        self.assertLess(np.linalg.norm(row['mass_defect']),1e-10);self.assertLess(abs(row['energy_defect']),1e-10)
        self.assertAlmostEqual(row['darcy_dissipation'],.0025*float(expected@H@expected))
        w=np.random.default_rng(9).normal(size=len(s.free)+2)*1e-5;dq=np.zeros_like(new.q);dq.ravel()[s.free]=w[:-2];eps=1e-3
        exact=s.jvp(old,new.q,new.p,.0025,w);fd=(s.residual(old,new.q+eps*dq,new.p+eps*w[-2:],.0025)['residual']-s.residual(old,new.q-eps*dq,new.p-eps*w[-2:],.0025)['residual'])/(2*eps)
        np.testing.assert_allclose(exact,fd,rtol=1e-5,atol=1e-8)
    def test_pressure_relaxation_and_invalid_parameters(self):
        g,op=self.fixtures();s=Solver(self.m,g,op,drained=True,resistance=[2.,3.,4.]);old=s.rest();old.p[:]=[.2,.1];h=.001
        C=np.diag(s.C);end=np.linalg.solve(C+h*s.L/2,(C-h*s.L/2)@old.p)
        np.testing.assert_allclose(s.residual(old,old.q,end,h)['mass'],0.,atol=1e-12)
        for H in [-1.,[1.,0.,1.],np.ones((2,2))]:
            with self.assertRaises(ValueError):Solver(self.m,g,op,drained=True,resistance=H)
        with self.assertRaises(ValueError):Solver(self.m,g,op,storage_modulus=0)
    def test_four_step_driver_matches_time_layers(self):
        targets=[]
        def step(s,h,target):
            targets.append(target);return SimpleNamespace(time=s.time+h,step=s.step+1),{}
        s,rows=advance_series(step,SimpleNamespace(time=0.,step=0),.0025,4,lambda t:-.1*t*t)
        self.assertEqual(s.step,4);self.assertAlmostEqual(s.time,.01);self.assertEqual(len(rows),4)
        np.testing.assert_allclose(targets,[-.1*(i*.0025)**2 for i in range(1,5)])
    def test_controller_endpoint_retry_and_rule_ledger(self):
        from unittest.mock import patch
        from engine.aniso_phase1.research_material_coupling_next.controller import Controller
        import copy
        old=State(self.m.rest().q,np.zeros_like(self.m.rest().q),np.zeros(2),rule=5)
        old.rule_work=.125;old.external_work=.25;digest=old.digest();starts=[]
        c=Controller.__new__(Controller);c.model=self.m;c.primary=2;c.reserve=7;c.budget=Budget();c.attempts=[];c.journal=None
        class Mat:
            def __init__(self,n):self.n=n
            def evaluate(self,q):return dict(U=float(self.n))
        class Sol:
            identity={}
            def __init__(self,n):self.n=n
            def step(self,s,h,target):
                starts.append((self.n,s.digest()));z=copy.deepcopy(s);z.time+=h;z.step+=1;z.rule=self.n;z.external_work+=self.n
                return z,{}
        c.materials={n:Mat(n) for n in (2,5,7)};c.solvers={n:Sol(n) for n in (2,7)}
        with patch('engine.aniso_phase1.research_material_coupling_next.controller.compare',return_value=dict(accepted=False)):
            new,row=c.step(old,.01,None)
            self.assertEqual(starts,[(2,digest),(7,digest)])
            self.assertEqual(new.external_work,old.external_work+7)
            self.assertEqual(new.rule_work,old.rule_work+2)
            self.assertEqual(row['delta_U_rule'],2.)
            self.assertTrue(row['fallback']);self.assertEqual(old.digest(),digest)
            with self.assertRaisesRegex(RuntimeError,'injected'):c.step(old,.01,None,fault=True)
            self.assertEqual(old.digest(),digest)
            with patch.object(c.solvers[7],'step',side_effect=RuntimeError('reserve failed')):
                with self.assertRaisesRegex(RuntimeError,'reserve failed'):c.step(old,.01,None)
            self.assertEqual(old.digest(),digest)

    def test_checkpoint_rejects_changed_hydraulic_identity(self):
        g,op=self.fixtures();s=Solver(self.m,g,op,drained=True);other=Solver(self.m,g,op,drained=True,resistance=2.)
        state=s.rest();state.p[:]=[.02,.01];state.rule_work=.04
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'s.npz';save(state,p,s.identity);self.assertEqual(load(p,self.m,s.identity).digest(),state.digest())
            with self.assertRaises(ValueError):load(p,self.m,other.identity)

if __name__=='__main__':unittest.main()
