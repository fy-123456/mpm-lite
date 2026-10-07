"""Focused regression cases for the new independent model and general solver."""
import unittest
from types import SimpleNamespace
import numpy as np
from engine.aniso_phase1.research_sequential.condensation import Condensation
from engine.aniso_phase1.research_sequential.solver import solve_mixed


class CondensationTests(unittest.TestCase):
    def test_stationary_null_elimination_keeps_boundary_force_and_kinetic_energy(self):
        # Two carrier functions coincide; their original stabilization couples
        # a prescribed boundary row. Static force changes if offset is omitted.
        oldA=np.array([[1.,1.,0.],[0.,0.,1.]])
        Ks=np.array([[3.,.2,.7],[.2,2.,-.4],[.7,-.4,2.]])
        ref=np.array([[.2,0,0],[.3,0,0],[.7,0,0]])
        s=SimpleNamespace(ndof=3,n=3,oldA=oldA,Ks=Ks,reference=ref,
            free_scalar_ids=np.array([0,1]),fixed_scalar_ids=np.array([2]),signature='synthetic')
        M=oldA.T @ oldA;K=np.kron(Ks+M,np.eye(3));r=Condensation(s,M,K)
        w=np.array([[.02,.01,0],[.1,0,0]]);q=r.expand(w)
        self.assertEqual(len(r.free),1)
        np.testing.assert_allclose(r.N.T @ Ks @ (ref+q),0,atol=1e-12)
        np.testing.assert_allclose(q[2],w[1])
        d=np.array([[.03,.02,0],[.04,0,0]])
        self.assertAlmostEqual(np.sum(d*(r.M @ d)),np.sum(r.velocity(d)*(M @ r.velocity(d))))
        def energy(x):
            z=ref+r.expand(x)
            return .5*np.sum(z*(Ks @ z))
        exact=np.sum(d*(r.P.T @ Ks @ (ref+q)))
        self.assertAlmostEqual((energy(w+1e-6*d)-energy(w-1e-6*d))/2e-6,exact,places=8)

    def test_unstabilized_null_is_rejected(self):
        A=np.array([[1.,1.]])
        s=SimpleNamespace(ndof=2,n=2,oldA=A,Ks=np.zeros((2,2)),reference=np.zeros((2,3)),
            free_scalar_ids=np.array([0,1]),fixed_scalar_ids=np.array([],dtype=int),signature='bad')
        # SVD needs enough rows to represent the full coefficient kernel.
        s.oldA=np.vstack((A,A))
        with self.assertRaises(ValueError):Condensation(s,A.T @ A,np.zeros((6,6)))


class MixedTests(unittest.TestCase):
    def test_indefinite_nonsymmetric_equations_use_general_solver(self):
        A=np.array([[2.,-1.],[1.,-.2]]);b=np.array([.3,.1])
        for method in ('direct','gmres'):
            x,info=solve_mixed(A,b,method=method)
            np.testing.assert_allclose(A @ x,b,atol=1e-10)
            self.assertFalse(info['spd_assumed'])
        with self.assertRaises(ValueError):solve_mixed(A,b,method='cg')




class BackendPhysicsTests(unittest.TestCase):
    def test_general_backend_preserves_pressure_mass_and_dissipation(self):
        from engine.aniso_phase1.types import AnisotropicMaterialParams
        from engine.aniso_phase1.research_e.flow import Grid,Darcy
        from engine.aniso_phase1.research_e.poro import Skeleton,Biot
        from engine.aniso_phase1.research_sequential.solver import GeneralBiot
        grid=Grid((2,2));solid=Skeleton(grid,AnisotropicMaterialParams(10,20,200,[1,1,0]))
        flow=Darcy(grid,np.array([[.01,.003],[.003,.02]]))
        new=GeneralBiot(solid,flow,.001);old=Biot(solid,flow,.001)
        state=new.initial();boundary={(a,b):('pressure',.01*b) for a in range(2) for b in (0,1)}
        for dt in (.01,.02):
            args=(state,dt,solid.traction(0,1,[-.01,0]),boundary,.001)
            a,b=new.step(*args),old.step(*args)
            np.testing.assert_allclose(a.state.u,b.state.u,atol=1e-12)
            np.testing.assert_allclose(a.state.p,b.state.p,atol=1e-12)
            np.testing.assert_allclose(a.flux,b.flux,atol=1e-12)
            for key in ('local_mass_defect','energy_residual','physical_dissipation','momentum_balance'):
                self.assertAlmostEqual(a.metrics[key],b.metrics[key],places=11)
            self.assertTrue(a.converged);self.assertGreaterEqual(a.metrics['physical_dissipation'],0.)
            state=a.state

    def test_checkpoint_rejects_changed_state_or_model(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import json
        from engine.aniso_phase1.research_d.common_state import CommonState
        from benchmarks.research_sequential.run import save_state,load_state
        state=CommonState(np.zeros((2,3)),np.zeros((2,3)))
        model=SimpleNamespace(signature='model-A',validate=lambda state:None)
        with TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.json';save_state(path,model,state)
            self.assertEqual(load_state(path,model).digest(),state.digest())
            with self.assertRaises(ValueError):load_state(path,SimpleNamespace(signature='model-B'))
            data=json.loads(path.read_text());data['payload']['q'][0][0]=.1
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):load_state(path,model)

if __name__=='__main__':unittest.main()
