import unittest
import numpy as np
import scipy.sparse as sp
from benchmarks.research_d.protocol import problem,CASES,GATES
from engine.aniso_phase1.research_d.cpu import pcg
from engine.aniso_phase1.research_d.tensor import FrozenTensor
from engine.aniso_phase1.research_d.precondition import Schwarz,tensor_blocks


class CPUChecks(unittest.TestCase):
    def test_tensor_oracle_multiple_orders_and_angles(self):
        rng=np.random.default_rng(4)
        for case in CASES[:5]:
            p=problem(case); A=p.assembled(); x,y=rng.normal(size=(2,p.size))
            np.testing.assert_allclose(p.apply(x),A@x,rtol=1e-7,atol=1e-10)
            self.assertLess(abs(x@p.apply(y)-y@p.apply(x))/(np.linalg.norm(x)*np.linalg.norm(p.apply(y))),1e-8)
            self.assertGreater(x@p.precondition(x),0.)

    def test_solve_residual_constraints_and_work(self):
        p=problem(CASES[0]); outputs=[]
        for tol in GATES['linear_rtol']:
            x,r=pcg(p.apply,p.rhs,p.precondition,rtol=tol)
            self.assertTrue(r.converged,r.record()); m=p.metrics(x)
            self.assertLessEqual(m['true_residual'],r.target*1.001)
            self.assertEqual(m['constraint_residual'],0.)
            self.assertLess(m['work_identity_absolute_J'],GATES['work_absolute_J']); outputs.append(m['reaction_N'])
        self.assertLess(abs(outputs[0]-outputs[1])/abs(outputs[1]),GATES['field_relative'])

    def test_indefinite_and_invalid_preconditioner(self):
        A=np.array([[1.,2.],[2.,1.]])
        _,r=pcg(lambda x:A@x,np.array([1.,-1.]))
        self.assertEqual(r.status,'nonpositive_curvature'); self.assertTrue(r.negative_curvature)
        _,r=pcg(lambda x:x,np.ones(2),lambda x:-x)
        self.assertEqual(r.status,'invalid_preconditioner')

    def test_zero_rhs_exact_start_and_iteration_limit(self):
        for b,x0 in ((np.zeros(2),None),(np.ones(2),np.ones(2))):
            _,r=pcg(lambda x:x,b,x0=x0); self.assertTrue(r.converged); self.assertEqual(r.iterations,0)
        _,r=pcg(lambda x:x,np.ones(2),maxiter=0); self.assertEqual(r.status,'iteration_limit')

    def test_nonfinite_or_invalid_contract_rejected(self):
        with self.assertRaises(ValueError): pcg(lambda x:x,np.array([np.nan]))
        with self.assertRaises(ValueError): pcg(lambda x:x,np.ones(2),rtol=-1)
        with self.assertRaises(ValueError): FrozenTensor([[0,1],[0,1],[1,0]])

    def test_local_natural_and_artificial_faces_match_principal_submatrix(self):
        from benchmarks.aniso_local_q3 import assemble
        e=[np.linspace(.25,.75,3),np.linspace(.375,.625,3),np.linspace(.375,.625,3)]
        for faces in (((True,True),(True,True),(False,False)),((True,True),(False,True),(True,False))):
            p=FrozenTensor(e,3,faces=faces,displacement=0.)
            v=np.random.default_rng(4).normal(size=p.size)
            np.testing.assert_allclose(p.apply(v),p.assembled()@v,rtol=1e-7,atol=1e-10)

    def test_schwarz_is_spd_and_original_solve_unchanged(self):
        p=problem(CASES[0]); A=p.assembled(); blocks=tensor_blocks(p.free_shape,3,1)
        Z=np.zeros((p.size,3))
        for i in range(3): Z[i*(p.size//3):(i+1)*(p.size//3),i]=1
        M=Schwarz(A,blocks,base=p.precondition,coarse=Z)
        x,y=np.random.default_rng(6).normal(size=(2,p.size))
        self.assertGreater(x@M(x),0.)
        self.assertLess(abs(x@M(y)-y@M(x))/np.linalg.norm(x)/np.linalg.norm(M(y)),1e-8)
        u,r=pcg(p.apply,p.rhs,M,rtol=1e-7); self.assertTrue(r.converged)
        self.assertLess(p.metrics(u)['work_identity_absolute_J'],1e-8)
        with self.assertRaises(np.linalg.LinAlgError): Schwarz(sp.csr_matrix([[1.,2.],[2.,1.]]),[np.array([0,1])])

    def test_input_mutation_does_not_change_frozen_problem(self):
        e=[np.linspace(.25,.75,3),np.linspace(.375,.625,3),np.linspace(.375,.625,3)]
        p=FrozenTensor(e); key=p.key; e[0][1]=.6
        self.assertEqual(p.key,key); self.assertEqual(p.edges[0][1],.5)
        with self.assertRaises(ValueError): p.H[0,0]=2


class HeterogeneousChecks(unittest.TestCase):
    def test_uniform_slabs_recover_original_tensor_and_heterogeneous_symmetry(self):
        from engine.aniso_phase1.research_d.heterogeneous import FrozenHeterogeneous
        p=problem(CASES[0]);H=np.broadcast_to(p.H,(len(p.edges[0])-1,9,9)).copy()
        q=FrozenHeterogeneous(p.edges,p.degree,H);v,w=np.random.default_rng(12).normal(size=(2,p.size))
        np.testing.assert_allclose(q.apply(v),p.apply(v),rtol=1e-7,atol=1e-10)
        H[::2]*=10;q=FrozenHeterogeneous(p.edges,p.degree,H)
        self.assertGreater(v@q.apply(v),0.)
        self.assertLess(abs(v@q.apply(w)-w@q.apply(v))/np.linalg.norm(v)/np.linalg.norm(q.apply(w)),1e-8)
        x,r=pcg(q.apply,q.rhs,q.precondition,rtol=1e-7);self.assertTrue(r.converged)
        self.assertLess(q.metrics(x)['work_identity_absolute_J'],1e-8)
