import copy
import unittest
import numpy as np
from engine.aniso_phase1.types import AnisotropicMaterialParams
from engine.aniso_phase1.research_e.flow import Grid, Darcy
from engine.aniso_phase1.research_e.poro import Skeleton, Biot
from engine.aniso_phase1.research_e.stage2.evaluation import (StressEvaluator, stress_from_gradient,
    stress_errors, stress_passed, weighted_error)
from engine.aniso_phase1.research_e.stage2.transaction import ChildAdapter


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.g = Grid((4,4))
        self.s = Skeleton(self.g,AnisotropicMaterialParams(10,20,200,[1,1,0]))
        self.e = StressEvaluator(self.s)

    def test_q2_cross_polynomial_center_mean_and_zz(self):
        x,y = self.s.nodes.T
        u = np.column_stack((x*y*y+.2*x*x*y, .3*x*x*y*y)).ravel()
        p = np.full(self.g.nc,.17)
        v = self.e.evaluate(u,p)
        def grad(X):
            x,y = X
            return np.array([[y*y+.4*x*y,2*x*y+.2*x*x],[.6*x*y*y,.6*x*x*y]])
        ref = self.e.reference(grad,lambda X:.17)
        center_exact=stress_from_gradient(np.array([grad(X) for X in self.g.centers]),p,self.s.material)[1]
        np.testing.assert_allclose(v['total_center'],center_exact,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(v['total_full_field'],ref['total_full_field'],rtol=1e-10,atol=1e-10)
        x,y = self.g.centers.T; hx,hy = self.g.h
        x2,y2 = x*x+hx*hx/12,y*y+hy*hy/12
        mean = np.empty((self.g.nc,2,2))
        mean[:,0,0]=y2+.4*x*y; mean[:,0,1]=2*x*y+.2*x2
        mean[:,1,0]=.6*x*y2; mean[:,1,1]=.6*x2*y
        exact=stress_from_gradient(mean,p,self.s.material)[1]
        np.testing.assert_allclose(v['total_cell_mean'],exact,rtol=1e-10,atol=1e-10)
        self.assertGreater(np.linalg.norm(v['total_center']-v['total_cell_mean']),.01)
        self.assertGreater(np.linalg.norm(v['total_cell_mean'][:,2,2]),.01)
        fine=StressEvaluator(self.s,6).evaluate(u,p)
        np.testing.assert_allclose(v['total_cell_mean'],fine['total_cell_mean'],atol=1e-10)

    def test_constant_strain_and_pressure_once(self):
        grad=np.array([[.01,.02],[-.03,.04]])
        v=self.e.evaluate((self.s.nodes @ grad.T).ravel(),np.full(self.g.nc,.1))
        ref=stress_from_gradient(grad,.1,self.s.material)
        np.testing.assert_allclose(v['total_cell_mean'],np.broadcast_to(ref[1],v['total_cell_mean'].shape),atol=1e-10)
        np.testing.assert_allclose(v['effective_cell_mean']-v['total_cell_mean'],np.broadcast_to(.1*np.eye(3),v['total_cell_mean'].shape),atol=1e-10)

    def test_stress_gate_rejects_bad_sign_missing_region_and_empty(self):
        zero=np.zeros(self.s.ndof); p=np.full(self.g.nc,.1)
        good=self.e.evaluate(zero,p)
        scores=stress_errors(self.e,good,good)
        self.assertTrue(stress_passed(scores))
        for key in ('effective_cell_mean','total_cell_mean','effective_full_field','total_full_field'):
            bad={k:v.copy() for k,v in good.items()}; bad[key]+=1.
            self.assertFalse(stress_passed(stress_errors(self.e,bad,good)))
        bad=self.e.evaluate(zero,-p)
        self.assertFalse(stress_passed(stress_errors(self.e,bad,good)))
        del scores['boundary']; self.assertFalse(stress_passed(scores))
        with self.assertRaises(ValueError): weighted_error(np.zeros((0,3)),np.zeros((0,3)),np.zeros(0))
        scores=stress_errors(self.e,good,good);scores['interior']['total_full_field']['error']=float('nan')
        self.assertFalse(stress_passed(scores))

    def test_near_zero_has_fixed_absolute_scale(self):
        score=weighted_error(np.full((2,1),.001),np.zeros((2,1)),np.full(2,.5))
        self.assertAlmostEqual(score['error'],.01)
        self.assertEqual(score['normalization'],'fixed_absolute_scale')


class IdentityTests(unittest.TestCase):
    def test_missing_supplement_and_changed_asset_rejected(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from benchmarks.research_e.stage2.run import check_manifest
        from engine.aniso_phase1.research_e.stage2.handoff import sha
        with TemporaryDirectory() as tmp:
            root=Path(tmp);asset=root/'protocol.json';asset.write_text('{}')
            manifest={'protocol.json':sha(asset)}
            self.assertTrue(check_manifest(root,manifest)['passed'])
            asset.write_text('{"changed":true}')
            with self.assertRaises(ValueError):check_manifest(root,manifest)
            asset.unlink()
            with self.assertRaises(ValueError):check_manifest(root,manifest)


class ChildTests(unittest.TestCase):
    def setUp(self):
        g=Grid((2,2));s=Skeleton(g,AnisotropicMaterialParams(10,20,0))
        self.b=Biot(s,Darcy(g,np.eye(2)*.01),.001)
        self.a=ChildAdapter(self.b,'parent-test')
        self.kw=dict(dt=.01,load=s.traction(0,1,[-.1,0]),boundary={(a,b):('pressure',0.) for a in range(2) for b in (0,1)})

    def test_owned_values_and_restart_parameter_rejection(self):
        old=self.a.initial(); saved=copy.deepcopy(old)
        new=self.a.prepare(old,**self.kw)
        self.assertEqual(old,saved)
        self.assertEqual(new,self.a.restore(self.a.checkpoint(new)))
        new['u'][2]+=.1
        with self.assertRaises(ValueError):self.a.validate(new)
        cp=self.a.checkpoint(old);cp['payload']['config']['storage']=2
        with self.assertRaises(ValueError):self.a.restore(cp)

    def test_nonconverged_never_produces_candidate(self):
        old=self.a.initial();saved=copy.deepcopy(old)
        with self.assertRaises(ValueError):self.a.prepare(old,method='once',**self.kw)
        self.assertEqual(old,saved)


if __name__=='__main__':unittest.main()
