import json
import tempfile
from pathlib import Path
import unittest
from dataclasses import replace,asdict
import numpy as np
import warp as wp
from tests.research_d.test_common_space import small_package
from engine.aniso_phase1.research_d.common_space import CommonSpace
from engine.aniso_phase1.research_d.stage2.contracts import OperatorContract,MixedBlockContract,validate_handoff
from engine.aniso_phase1.research_d.stage2.solver import solve_linear,equilibrate
from engine.aniso_phase1.research_d.stage2.gpu_operator import GPUOperator,response_kernel,exact_tangent_kernel
from engine.aniso_phase1.consistent_transfer import material_response
from engine.aniso_phase1.history_increment import material_tangent
from engine.aniso_phase1.types import AnisotropicMaterialParams

class Contracts(unittest.TestCase):
    def test_wrong_identity_and_missing_capability(self):
        c=OperatorContract(*(['a'*64]*8),(219,3),(369,3),'cpu')
        with self.assertRaises(ValueError):c.validate({'mass_rule_sha256':'b'*64})
        with self.assertRaises(ValueError):replace(c,state_sha256='').validate()
        with self.assertRaises(ValueError):c.check_field(np.zeros((369,3)),'free_displacement')
        evidence={'static_operator':dict(passed=True,contract_sha256=c.signature)}
        self.assertTrue(validate_handoff(c,expected={},capabilities=['static_operator'],required=['static_operator'],evidence=evidence))
        with self.assertRaises(ValueError):validate_handoff(c,expected={},capabilities=['static_operator'],required=['cuda'],evidence=evidence)
        with self.assertRaises(ValueError):validate_handoff(c,expected={},capabilities=['static_operator'],required=['static_operator'],evidence=evidence,solver='pcg')
        mixed=replace(c,operator_kind='mixed_general',q_shape=(3,2),full_shape=(4,2))
        with self.assertRaises(ValueError):validate_handoff(mixed,expected={},capabilities=[],required=[],evidence={},solver='pcg',spd_evidence=dict(positive_definite=True,contract_sha256=mixed.signature))

    def test_serialized_shapes_and_explicit_mixed_blocks(self):
        solid=OperatorContract(*(['a'*64]*8),(219,3),(369,3),'cpu')
        restored=OperatorContract(**json.loads(json.dumps(asdict(solid))))
        restored.check_field(np.zeros((219,3)),'free_displacement')
        self.assertEqual(restored.signature,solid.signature)
        mixed=MixedBlockContract(*(['a'*64]*8),12,6,('N','m^3/s'),'cpu').validate()
        with self.assertRaises(ValueError):validate_handoff(mixed,expected={},capabilities=[],required=[],evidence={},solver='pcg',spd_evidence=dict(positive_definite=True,contract_sha256=mixed.signature))
        with self.assertRaises(ValueError):replace(mixed,pressure_size=0).validate()
        with self.assertRaises(ValueError):restored.validate({'state_sha256':'b'*64})

class Solvers(unittest.TestCase):
    def test_negative_and_singular_controls(self):
        _,r=solve_linear(np.diag([1.,-2.]),np.ones(2));self.assertEqual(r['method'],'svd_lstsq');self.assertTrue(r['converged'])
        _,r=solve_linear(np.diag([1.,0.]),np.ones(2));self.assertFalse(r['converged'])
        _,r=solve_linear(np.diag([1.,2.]),np.ones(2));self.assertEqual(r['method'],'pcg');self.assertTrue(r['converged'])
    def test_rejected_trials_rollback_and_no_false_convergence(self):
        initial=np.array([[1.]])
        def objective(q):
            if q[0,0]<0:raise ValueError('illegal state')
            return dict(U=float((q*q).sum()/2),force=q.copy(),min_detF=1.)
        q,_,r=equilibrate(objective,initial,np.array([[.1]]),max_iterations=1)
        np.testing.assert_array_equal(initial,[[1.]])
        self.assertTrue(r['rejections']);self.assertFalse(r['converged']);self.assertGreaterEqual(q[0,0],0.)

class GPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir='/root/autodl-tmp/mpm-lite-research-d/stage2/warp-cache'
        if not wp.is_cuda_available():raise unittest.SkipTest('CUDA unavailable')
        cls.tmp=tempfile.TemporaryDirectory();folder=Path(cls.tmp.name);small_package(folder)
        cls.s=CommonSpace(folder);cls.op=GPUOperator(cls.s)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def test_small_full_operator_and_mapping(self):
        s=self.s;q=s.q0;d=s.test_vectors['direction'];points=[s.test_vectors[f'points{k}'] for k in range(3)]
        for direction in (False,True):
            a=self.op.maps.evaluate(d if direction else q,points,direction)
            b=s.jvp(d,points) if direction else s.evaluate(q,points)
            for x,y in zip(a,b):np.testing.assert_allclose(x,y,atol=1e-11,rtol=1e-8)
        a=self.op.evaluate(q,d);b=s.response(q,d,order=6)
        for k in ('U','full_force','tangent_action'):np.testing.assert_allclose(a[k],b[k],atol=1e-9,rtol=1e-7)
    def test_foreign_linearization_and_memory_budget(self):
        lin=self.op.prepare(self.s.q0)
        lin.operator_sha256='0'*64
        with self.assertRaises(ValueError):self.op.action(lin,self.s.test_vectors['direction'])
        with self.assertRaises(MemoryError):GPUOperator(self.s,max_workspace_bytes=1)

    def test_material_repeated_rotation_compression_and_invalid(self):
        p=AnisotropicMaterialParams(10.,20.,200.,[1.,1.,0.]);A=p.A0
        theta=.2;rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        F=np.array([np.eye(3),1.1*np.eye(3),rotation,np.diag([.65,.8,1.]),[[1,.2,0],[0,1,.1],[.04,0,1.]]])
        d=np.random.default_rng(7).normal(size=F.shape);n=len(F);dev='cuda:0'
        f=wp.array(F,dtype=wp.mat33d,device=dev);df=wp.array(d,dtype=wp.mat33d,device=dev)
        Q=wp.empty_like(f);c=wp.empty(n,dtype=wp.vec3d,device=dev);P=wp.empty_like(f);dp=wp.empty_like(f)
        psi=wp.empty(n,dtype=wp.float64,device=dev);det=wp.empty_like(psi);bad=wp.zeros(1,dtype=wp.int32,device=dev)
        args=[f,wp.mat33d(*A.ravel()),p.mu,p.lam,p.k_f,psi,P,Q,c,det,bad]
        wp.launch(response_kernel,dim=n,inputs=args,device=dev)
        wp.launch(exact_tangent_kernel,dim=n,inputs=[f,Q,c,df,wp.mat33d(*A.ravel()),p.mu,p.lam,p.k_f,dp],device=dev)
        e,stress=material_response(F,np.broadcast_to(A,F.shape),p);tangent=material_tangent(F,np.broadcast_to(A,F.shape),d,p)
        self.assertEqual(bad.numpy()[0],0)
        np.testing.assert_allclose(psi.numpy(),e,atol=1e-10,rtol=1e-7);np.testing.assert_allclose(P.numpy(),stress,atol=1e-10,rtol=1e-7);np.testing.assert_allclose(dp.numpy(),tangent,atol=1e-9,rtol=1e-7)
        F[0,0,0]=-1;f.assign(F);wp.launch(response_kernel,dim=n,inputs=args,device=dev);self.assertEqual(bad.numpy()[0],1)

if __name__=='__main__':unittest.main()
