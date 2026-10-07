import os
import unittest
import numpy as np
import scipy.sparse as sp
import warp as wp
wp.config.kernel_cache_dir=os.environ.get('MPM_LITE_WARP_CACHE','/tmp/mpm-lite-research-d-cache')
from benchmarks.research_d.protocol import problem,CASES
from engine.aniso_phase1.research_d.gpu import TensorGPU,ResidentPCG,CSRGPU,BackendCache
from engine.aniso_phase1.research_d.tensor import FrozenTensor
from engine.aniso_phase1.research_d.cpu import pcg
from engine.aniso_phase1.research_d.precondition import Schwarz,tensor_blocks,BatchedSchwarz


@unittest.skipUnless(wp.is_cuda_available(),'CUDA unavailable; cannot certify GPU')
class GPUChecks(unittest.TestCase):
    def test_tensor_and_separable_operator_equivalence(self):
        for case in CASES[:5]:
            p=problem(case); g=TensorGPU(p); v=np.random.default_rng(6).normal(size=p.size)
            np.testing.assert_allclose(g.host_apply(v),p.assembled()@v,rtol=1e-7,atol=1e-10)
            np.testing.assert_allclose(g.host_apply(v,True),p.precondition(v),rtol=1e-7,atol=1e-10)
            g.close()

    def test_graph_and_launch_solution_reuse_true_residual(self):
        p=problem(CASES[0]); reference,r=pcg(p.apply,p.rhs,p.precondition,rtol=1e-8)
        for use_graph in (False,True):
            g=TensorGPU(p); solver=ResidentPCG(g,maxiter=600,use_graph=use_graph)
            for scale in (1.,.4,0.):
                x,info=solver.solve(scale*p.rhs,rtol=1e-8,diagnostic_true_residuals=True)
                self.assertTrue(info.converged,info.record())
                np.testing.assert_allclose(x,scale*reference,rtol=1e-5,atol=1e-9)
                self.assertLessEqual(np.linalg.norm(scale*p.rhs-p.apply(x)),info.target*1.002)
                if use_graph:self.assertTrue(info.diagnostics['graph_used'])
            solver.close(); g.close()
            with self.assertRaises(RuntimeError):solver.solve(p.rhs)

    def test_nonpositive_curvature_and_limit(self):
        A=sp.csr_matrix([[1.,2.],[2.,1.]])
        g=CSRGPU(A); s=ResidentPCG(g,maxiter=20)
        _,r=s.solve(np.array([1.,-1.])); self.assertEqual(r.status,'nonpositive_curvature')
        g2=CSRGPU(sp.diags([1.,2.])); s2=ResidentPCG(g2,maxiter=1)
        # Jacobi is exact here, hence one-iteration convergence must win over limit.
        _,r=s2.solve(np.ones(2)); self.assertTrue(r.converged)
        p=problem(CASES[0]); s3=ResidentPCG(TensorGPU(p),maxiter=1)
        _,r=s3.solve(p.rhs); self.assertEqual(r.status,'iteration_limit')

    def test_cache_dependency_changes_and_release(self):
        p=problem(CASES[0]); c=BackendCache(8); a=c.acquire(p,'one')
        self.assertIs(a,c.acquire(problem(CASES[0]),'one'))
        variants=[FrozenTensor([p.edges[0]**2,*p.edges[1:]],p.degree,p.H),
                  FrozenTensor(p.edges,p.degree,p.H*2),FrozenTensor(p.edges,p.degree+1,p.H),
                  FrozenTensor(p.edges,p.degree,p.H,faces=((True,True),(True,False),(False,False)),displacement=0.)]
        for q in variants:self.assertIsNot(a,c.acquire(q,'one'))
        self.assertIsNot(a,c.acquire(p,'two'))
        c.release('one'); self.assertTrue(a.closed)
        b=c.acquire(p,'one'); self.assertIsNot(a,b)
        c.release(); self.assertTrue(b.closed)

    def test_local_batch_and_memory_budget(self):
        p=problem(CASES[0]); M=Schwarz(p.assembled(),tensor_blocks(p.free_shape,2,1))
        g=BatchedSchwarz(M,batch_bytes=64<<10)
        r=np.random.default_rng(7).normal(size=p.size)
        a=wp.array(r,dtype=wp.float64,device='cuda:0'); out=wp.empty_like(a); g.apply(a,out)
        np.testing.assert_allclose(out.numpy(),M(r),rtol=1e-7,atol=1e-10)
        with self.assertRaises(MemoryError):BatchedSchwarz(M,total_bytes=8)

    def test_exact_material_compression_and_repeated_singular_values(self):
        from engine.aniso_phase1.research_d.material import MaterialGPU
        from engine.aniso_phase1.history_increment import material_response,material_tangent
        from engine.aniso_phase1.types import AnisotropicMaterialParams
        params=AnisotropicMaterialParams(10.,20.,200.,[1.,1.,0.])
        F=np.array([np.eye(3),np.diag([.65,.85,1.]),np.diag([1.2,1.2,1.2]),[[1.1,.2,0],[0,.9,0],[0,0,1]]])
        A=np.broadcast_to(params.A0,F.shape); dF=np.random.default_rng(8).normal(size=F.shape)
        gpu=MaterialGPU(A,params,max_batch=2); r=gpu.evaluate(F,dF); e,P=material_response(F,A,params)
        np.testing.assert_allclose(r['energy'],e,rtol=1e-7,atol=1e-10)
        np.testing.assert_allclose(r['stress'],P,rtol=1e-7,atol=1e-10)
        np.testing.assert_allclose(r['tangent'],material_tangent(F,A,dF,params),rtol=1e-5,atol=1e-7)
        with self.assertRaises(ValueError):gpu.evaluate(-F)

    def test_graph_capture_failure_keeps_guarded_launch_path(self):
        from unittest.mock import patch
        p=problem(CASES[0]);s=ResidentPCG(TensorGPU(p),maxiter=600)
        with patch('warp.ScopedCapture',side_effect=RuntimeError('injected capture failure')):
            x,r=s.solve(p.rhs)
        self.assertTrue(r.converged);self.assertFalse(r.diagnostics['graph_used'])
        self.assertIn('injected capture failure',r.diagnostics['graph_failure'])
        self.assertLessEqual(np.linalg.norm(p.rhs-p.apply(x)),r.target*1.002)
