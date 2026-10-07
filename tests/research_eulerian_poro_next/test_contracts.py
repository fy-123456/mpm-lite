import tempfile,unittest
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_unified_lite_poro.space import Space
from engine.aniso_phase1.research_unified_lite_poro.model import seed
from engine.aniso_phase1.research_eulerian_poro_next.materials import make_operator,validate_moments
from engine.aniso_phase1.research_eulerian_poro_next.bridge import ImprovedBridge
from engine.aniso_phase1.research_eulerian_poro_next.transfer import reference_gradient,current_gradient
from engine.aniso_phase1.research_eulerian_poro_next import checkpoint


class Contracts(unittest.TestCase):
    def test_fixed_budget_positive_common_moments_and_permutation(self):
        s=Space()
        for ppc in (2,4,8):
            p=seed(s,ppc);op,_=make_operator(s,p)
            self.assertEqual(len(op.points),128);validate_moments(op.A2,op.A4)
            bad=op.A4.copy();bad[0]*=.9
            with self.assertRaisesRegex(ValueError,'contraction'):validate_moments(op.A2,bad)
            for key in p.__dataclass_fields__:setattr(p,key,getattr(p,key)[::-1].copy())
            other,_=make_operator(s,p)
            np.testing.assert_allclose(other.A4,op.A4,atol=1e-10)

    def test_mixture_not_collapsed_and_prepare_fallback(self):
        s=Space();p=seed(s,4);p.A[:]=np.diag([.5,.5,0.])
        with self.assertRaisesRegex(ValueError,'rank-one'):make_operator(s,p)
        with self.assertRaisesRegex(ValueError,'rank-one'):make_operator(s,p,'director-or-positive')
        p.A[::2]=np.diag([1.,0.,0.]);p.A[1::2]=np.diag([0.,1.,0.])
        op,info=make_operator(s,p,'director-or-positive')
        self.assertTrue(info['fallback']);self.assertEqual(len(op.points),128)
        flat=op.A2.reshape(-1,9)
        self.assertGreater(np.linalg.norm(op.A4-flat[:,:,None]*flat[:,None,:]),.1)

    def test_scattered_rank_one_samples_and_rank_failure(self):
        s=Space();p=seed(s,4);rng=np.random.default_rng(43)
        p.X+=rng.normal(size=p.X.shape)*.001
        angle=.2+.7*p.X[:,0]+.9*p.X[:,1]
        a=np.c_[np.cos(angle),np.sin(angle),np.zeros(len(angle))];p.A=a[:,:,None]*a[:,None,:]
        op,_=make_operator(s,p)
        self.assertEqual(len(op.points),128)
        p.X[:,2]=0
        with self.assertRaises(ValueError):make_operator(s,p)

    def test_material_derivatives_and_objectivity(self):
        s=Space();op,_=make_operator(s,seed(s,4));rng=np.random.default_rng(19)
        q=rng.normal(size=(len(s.free),3))*1e-4;a=rng.normal(size=q.shape);a/=np.linalg.norm(a)
        b=rng.normal(size=q.shape);b/=np.linalg.norm(b);g=op.material(q)[1];errors=[]
        for eps in (3e-6,1e-6,3e-7):
            fd=(op.material(q+eps*a)[0]-op.material(q-eps*a)[0])/(2*eps)
            errors.append(abs(fd-np.sum(g*a)))
        self.assertLess(min(errors),1e-6)
        eps=1e-6
        Ka=(op.material(q+eps*a)[1]-op.material(q-eps*a)[1])/(2*eps)
        Kb=(op.material(q+eps*b)[1]-op.material(q-eps*b)[1])/(2*eps)
        self.assertLess(abs(np.sum(b*Ka)-np.sum(a*Kb)),1e-4)
        # Rotation at constitutive sites, including a genuinely mixed distribution.
        from engine.aniso_phase1.material_snapshot import response
        F=op.field(q);t=.4;R=np.array([[np.cos(t),-np.sin(t),0],[np.sin(t),np.cos(t),0],[0,0,1.]])
        E,P,_=response(F,op.A2,op.A4,op.params);Er,Pr,_=response(R@F,op.A2,op.A4,op.params)
        np.testing.assert_allclose(E,Er,atol=1e-8);np.testing.assert_allclose(R@P,Pr,atol=1e-7)

    def test_current_reference_increment_and_apic_distinction(self):
        F=np.array([[[1.1,.2,0],[0,.9,.1],[0,0,1.02]]]);L=np.array([[[.03,0,.1],[0,-.02,0],[0,.04,.01]]]);C=L+.1
        grad=reference_gradient(L,F)
        np.testing.assert_allclose(current_gradient(grad,F),L,atol=1e-12)
        dt=.01;np.testing.assert_allclose(F+dt*grad,(np.eye(3)+dt*L)@F)
        self.assertGreater(np.max(abs((np.eye(3)+dt*C)@F-(F+dt*grad))),1e-4)

    def test_capture_requires_matching_gradient_time_layer(self):
        from types import SimpleNamespace
        from engine.aniso_phase1.research_eulerian_poro_next.transfer import capture_lite_kinematics
        class Array:
            def __init__(self,value):self.value=np.asarray(value)
            def numpy(self):return self.value
        F0=np.diag([1.1,.9,1.])[None];L=np.diag([.1,.2,-.1])[None];F1=(np.eye(3)+.02*L)@F0
        s=SimpleNamespace(ptc_x=Array([[0,0,0]]),ptc_v=Array([[0,0,0]]),ptc_F=Array(F1),
            ptc_C=Array(L+.02),ptc_L=Array(L),ptc_A0=Array([np.diag([1.,0.,0.])]))
        self.assertNotIn('grad_X_v',capture_lite_kinematics(s))
        got=capture_lite_kinematics(s,gradient_reference_F=F0)
        np.testing.assert_array_equal(got['grad_X_v'],L@F0)
        self.assertGreater(np.max(abs(got['grad_X_v']-L@F1)),1e-5)

    def test_transaction_and_checkpoint(self):
        b=ImprovedBridge(Space());before=b.state.digest()
        with self.assertRaisesRegex(RuntimeError,'injected'):b.step(fault=True)
        self.assertEqual(before,b.state.digest());b.step()
        control=ImprovedBridge(Space());control.step();self.assertEqual(control.state.digest(),b.state.digest())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.npz';checkpoint.save(b,path);loaded=checkpoint.load(path,checkpoint.config(b))
            self.assertEqual(loaded.state.digest(),b.state.digest())
            self.assertEqual(loaded.rule,'director-log')
            with self.assertRaisesRegex(ValueError,'identity'):checkpoint.load(path,{'wrong':True})

    def test_owned_package_and_invalid_step(self):
        b=ImprovedBridge(Space());op,q,_,_=b.prepare();E=op.material(q)[0]
        b.state.particles.A[:]=np.nan
        self.assertEqual(op.material(q)[0],E)
        with self.assertRaises(ValueError):b.prepare()
        b=ImprovedBridge(Space());before=b.state.digest()
        with self.assertRaises(ValueError):b.step(-1)
        self.assertEqual(before,b.state.digest())

if __name__=='__main__':unittest.main()
