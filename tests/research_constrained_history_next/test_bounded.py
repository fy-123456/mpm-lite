import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_constrained_history_next.model import BoundedBridge,SeparatedOperator,compare
from engine.aniso_phase1.research_constrained_history_next.checkpoint import save,load,config
from engine.aniso_phase1.research_common_kinematics_next.model import CommonBridge
from engine.aniso_phase1.research_common_kinematics_next.mapping import basis,push

class BoundedTests(unittest.TestCase):
    def test_material_only(self):
        b=BoundedBridge(primary_order=2);a,r,_,_=b.prepare_pair()
        self.assertEqual((len(a.points),len(r.points),len(a.gX)),(16,432,128))
        for k in ('M','M3','gX','gw','gD','gN','F0','capacity'):
            np.testing.assert_array_equal(getattr(a,k),getattr(r,k))
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'op.npz';r.save(p);rr=SeparatedOperator.load_package(p)
            self.assertEqual(r.identity,rr.identity)
            np.testing.assert_array_equal(r.material(np.zeros((10,3)))[1],rr.material(np.zeros((10,3)))[1])
    def test_real_rejection_repeats_whole_step(self):
        b=BoundedBridge(primary_order=2);direct=BoundedBridge(primary_order=6)
        m=b.step();direct.step()
        self.assertTrue(m['retried']);self.assertFalse(m['rule_check']['force']['passed'])
        self.assertEqual(b.state.active_order,6)
        for k in ('qx','qF','qv','p','content','reserve_x','reserve_F','last_increment'):
            np.testing.assert_array_equal(getattr(b.state,k),getattr(direct.state,k))
        self.assertEqual(b.state.particles.digest(),direct.state.particles.digest())
    def test_rollback_all_stages(self):
        for stage in ('prepare','retry','solve','load','commit'):
            b=BoundedBridge(primary_order=2);before=b.state.digest()
            with self.assertRaisesRegex(RuntimeError,'injected'):b.step(fault=stage)
            self.assertEqual(before,b.state.digest())
    def test_continuous_reserve_history(self):
        b=BoundedBridge();X=b.space.rule(6)[0];x=X.copy();F=np.tile(np.eye(3),(len(X),1,1))
        # Shadow points verify the MATERIAL derivative after three updated-map steps.
        eps=1e-6;shadow=[X+eps*np.eye(3)[a] for a in range(3)]
        shadowm=[X-eps*np.eye(3)[a] for a in range(3)]
        Fs=[F.copy() for _ in range(6)]
        for i in range(3):
            b.step();d=b.state.last_increment;x,F=push(b.space,X,x,F,d)
            for a in range(3):
                shadow[a],Fs[a]=push(b.space,X+eps*np.eye(3)[a],shadow[a],Fs[a],d)
                shadowm[a],Fs[a+3]=push(b.space,X-eps*np.eye(3)[a],shadowm[a],Fs[a+3],d)
        np.testing.assert_array_equal(x,b.state.reserve_x);np.testing.assert_array_equal(F,b.state.reserve_F)
        fd=np.stack([(shadow[a]-shadowm[a])/(2*eps) for a in range(3)],axis=-1)
        np.testing.assert_allclose(F,fd,rtol=2e-4,atol=2e-6)
    def test_reserve_crosses_grid_with_material_derivative(self):
        b=BoundedBridge();X=b.space.rule(6)[0];x=X.copy();F=np.tile(np.eye(3),(len(X),1,1));d=np.zeros((10,3));d[0,0]=.05
        eps=1e-6;xp=[X+eps*np.eye(3)[a] for a in range(3)];xm=[X-eps*np.eye(3)[a] for a in range(3)]
        fp=[F.copy() for _ in range(6)]
        for scale in (1.,.2):
            x,F=push(b.space,X,x,F,scale*d)
            for a in range(3):
                xp[a],fp[a]=push(b.space,X+eps*np.eye(3)[a],xp[a],fp[a],scale*d)
                xm[a],fp[a+3]=push(b.space,X-eps*np.eye(3)[a],xm[a],fp[a+3],scale*d)
        self.assertTrue(np.any(np.floor(x/b.space.h)!=np.floor(X/b.space.h)))
        fd=np.stack([(xp[a]-xm[a])/(2*eps) for a in range(3)],axis=-1)
        np.testing.assert_allclose(F,fd,rtol=2e-4,atol=2e-6)

    def test_checkpoint_and_rule_jump(self):
        b=BoundedBridge(primary_order=2);b.step()
        # Diagnostic re-arm q2 only after its history has been carried from t=0.
        b.state.active_order=2
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'state.npz';save(b,p);c=load(p,config(b))
            self.assertEqual(c.state.digest(),b.state.digest())
            m=b.step();mc=c.step();self.assertTrue(m['retried'])
            self.assertNotEqual(m['rule_energy_jump'],0.)
            self.assertEqual(c.state.digest(),b.state.digest())
            foreign=copy.deepcopy(config(b));foreign['mass_order']=6
            with self.assertRaises(ValueError):load(p,foreign)
    def test_baseline_short_window(self):
        b=BoundedBridge();old=CommonBridge()
        for i in range(4):
            m=b.step();old.step(.0025);self.assertFalse(m['retried'])
        for k in ('qx','qF','qv','p','content'):
            np.testing.assert_array_equal(getattr(b.state,k),getattr(old.state,k))
        self.assertEqual(b.state.particles.digest(),old.state.particles.digest())
    def test_invalid_history_not_silently_replaced(self):
        b=BoundedBridge();b.state.reserve_F[0]=0;before=b.state.digest()
        with self.assertRaises(ValueError):b.step()
        self.assertEqual(before,b.state.digest())
    def test_tangent_and_work_consistency(self):
        b=BoundedBridge();b.step();a,r,v,_=b.prepare_pair();rng=np.random.default_rng(7)
        d=rng.normal(size=v.shape)*1e-5;w=rng.normal(size=v.shape);eps=1e-7
        for op in (a,r):
            e,f,_=op.material(d);ep=op.material(d+eps*w)[0];em=op.material(d-eps*w)[0]
            np.testing.assert_allclose((ep-em)/(2*eps),np.sum(f*w),rtol=2e-4,atol=1e-8)
        ka=(a.material(d+eps*w)[1]-a.material(d-eps*w)[1])/(2*eps)
        kr=(r.material(d+eps*w)[1]-r.material(d-eps*w)[1])/(2*eps)
        self.assertLess(np.linalg.norm(ka-kr)/np.linalg.norm(kr),.05)
        z=np.zeros_like(d);V0=a.geometry(z,False)[0];V=a.geometry(d,False)[0]
        np.testing.assert_allclose(np.einsum('cni,ni->c',a.discrete_G(z,d),d),V-V0,rtol=1e-7,atol=1e-12)

if __name__=='__main__':unittest.main()
