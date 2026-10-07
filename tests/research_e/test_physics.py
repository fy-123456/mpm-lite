import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from engine.aniso_phase1.types import AnisotropicMaterialParams
from engine.aniso_phase1.constitutive import energy, pk1
from engine.aniso_phase1.research_e.material import FullMaterial, tangent, linear_stress
from engine.aniso_phase1.research_e.flow import Grid, Darcy, oriented_permeability
from engine.aniso_phase1.research_e.drag import implicit_drag, exact_relative, drag_tensor
from engine.aniso_phase1.research_e.state import SaturatedState
from engine.aniso_phase1.research_e.poro import Skeleton, Biot, CouplingTransaction
from benchmarks.research_e.reference import consolidation_average


def boundary(dim, kind='pressure', value=0.):
    return {(d,s):(kind,value) for d in range(dim) for s in (0,1)}


class MaterialTests(unittest.TestCase):
    def test_full_material_derivatives_and_objectivity(self):
        params=AnisotropicMaterialParams(10,20,200,[1,1,0]); A=params.A0
        R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        for F in [np.eye(3), np.diag([1.1,1.1,.98]), np.array([[1.1,.03,0],[.01,.98,.02],[0,0,1.]])]:
            D=np.arange(9).reshape(3,3)/20-.1; eps=1e-5
            self.assertAlmostEqual(energy(R@F,A,params),energy(F,A,params),places=8)
            np.testing.assert_allclose(pk1(R@F,A,params),R@pk1(F,A,params),rtol=1e-7,atol=1e-8)
            fd=(pk1(F+eps*D,A,params)-pk1(F-eps*D,A,params))/(2*eps)
            np.testing.assert_allclose(tangent(F,A,D,params),fd,rtol=1e-5,atol=1e-5)
            de=(energy(F+eps*D,A,params)-energy(F-eps*D,A,params))/(2*eps)
            self.assertAlmostEqual(de,float(np.sum(pk1(F,A,params)*D)),places=5)

    def test_linear_control_matches_existing_law(self):
        for angle in [0,np.pi/4,np.pi/2]:
            p=AnisotropicMaterialParams(10,20,200,[np.cos(angle),np.sin(angle),0])
            E=np.array([[.1,.05],[.02,-.02]]); df=np.zeros((3,3));df[:2,:2]=E
            np.testing.assert_allclose(linear_stress(E,p),tangent(np.eye(3),p.A0,df,p)[:2,:2],atol=1e-8)

    def test_nonuniform_pairing_is_not_averaged(self):
        p=AnisotropicMaterialParams(10,20,200); m=FullMaterial([[1,0,0],[0,1,0]],[.4,.6],p)
        F=np.array([np.diag([1.2,1.,1.]),np.diag([1.,.9,1.])])
        W,P=m.evaluate(F)
        exact=sum(v*energy(f,a,p) for f,a,v in zip(F,m.A,m.volumes))
        self.assertEqual(W,exact)
        self.assertGreater(abs(W-m.evaluate(np.broadcast_to(F.mean(axis=0),F.shape))[0]),.01)
        with self.assertRaises(ValueError):m.evaluate(F[:1])
        with self.assertRaises(ValueError):FullMaterial([[1,0,0]],[0],p)


class FlowTests(unittest.TestCase):
    def test_rotated_linear_solution_and_gravity(self):
        g=Grid((5,6));K=oriented_permeability(.1,.001,np.pi/4); grad=np.array([.5,-.3])
        bc=boundary(2,value=lambda x:1+grad@x)
        f=Darcy(g,K,density=2,gravity=[.2,-.1]);r=f.solve(bc)
        np.testing.assert_allclose(r.pressure,1+g.centers@grad,atol=1e-9)
        np.testing.assert_allclose(r.cell_flux,np.broadcast_to(-K@(grad-2*np.array([.2,-.1])),r.cell_flux.shape),atol=1e-10)
        self.assertLess(np.max(abs(r.mass_residual)),1e-10)

    def test_neumann_gauge_and_compatibility(self):
        g=Grid((4,5));f=Darcy(g,np.eye(2));bc=boundary(2,'flux',0.)
        bc[0,0]=('flux',1.);bc[0,1]=('flux',-1.)
        r=f.solve(bc,mean_pressure=2.)
        np.testing.assert_allclose(r.pressure,2+g.centers[:,0]-.5,atol=1e-8)
        self.assertLess(np.max(abs(r.mass_residual)),1e-9)
        with self.assertRaisesRegex(ValueError,'incompatible'):f.solve(bc,source=1.)

    def test_discontinuous_permeability_conserves_flux(self):
        g=Grid((20,));K=np.where(g.centers[:,0]<.5,.01,1.)[:,None,None]
        f=Darcy(g,K);r=f.solve({(0,0):('pressure',1.),(0,1):('pressure',0.)})
        np.testing.assert_allclose(r.face_flux,1/(.5/.01+.5),rtol=1e-8)

    def test_invalid_tensor_and_missing_boundary_rejected(self):
        for K in [np.diag([1,-1]),np.array([[1,.2],[0,1]]),np.eye(2)*np.nan]:
            with self.assertRaises(ValueError):Darcy(Grid((2,2)),K)
        with self.assertRaises(ValueError):Darcy(Grid((2,2)),np.eye(2)).solve({})

    def test_pressure_nullspace_and_checkerboard(self):
        g=Grid((4,4));f=Darcy(g,oriented_permeability(.1,.001,.6));bc=boundary(2,'flux',0.)
        S=f.schur(bc);dense=S@np.eye(g.nc);eig=np.linalg.eigvalsh(dense)
        self.assertEqual(np.sum(eig<1e-9),1)
        checker=(-1.)**g.indices.sum(axis=1)
        self.assertGreater(checker@(dense@checker),1e-5)


class DragTests(unittest.TestCase):
    def test_actual_impulses_and_energy_at_stiff_steps(self):
        D=drag_tensor(oriented_permeability(1,.001,.8),1,.35,1)
        for dt in [1e-4,.01,1,100]:
            r=implicit_drag([0,.2],[1,-.4],2,1,D,dt)
            self.assertLess(np.linalg.norm(r.momentum_defect),1e-10)
            self.assertLess(abs(r.energy_residual),1e-9)
            self.assertGreaterEqual(r.physical_dissipation,0)
            self.assertGreaterEqual(r.numerical_loss,0)

    def test_drag_refines_to_exponential(self):
        D=np.array([[2.,.3],[.3,1.]]);errors=[]
        for n in [8,16,32]:
            vs=np.zeros(2);vf=np.array([1.,-.3])
            for _ in range(n):
                r=implicit_drag(vs,vf,2,1,D,.5/n);vs,vf=r.solid_velocity,r.fluid_velocity
            errors.append(np.linalg.norm(vf-vs-exact_relative([1,-.3],2,1,D,.5)))
        self.assertLess(errors[2],errors[1]);self.assertLess(errors[1],errors[0])

    def test_rotated_isotropic_drag_accepts_roundoff_symmetry(self):
        for angle in [0., np.pi/4, np.pi/2]:
            K=oriented_permeability(.01,.01,angle)
            D=drag_tensor(K,1.,.35,1.)
            r=implicit_drag([0,0],[1,-.2],2,1,D,.04)
            self.assertLess(abs(r.energy_residual),1e-8)
            self.assertLess(np.linalg.norm(r.momentum_defect),1e-10)

    def test_single_phase_and_invalid_drag(self):
        r=implicit_drag([.1,0],[0,0],1,0,np.zeros((2,2)),1.)
        np.testing.assert_array_equal(r.solid_velocity,[.1,0])
        with self.assertRaises(ValueError):implicit_drag([0],[1],1,0,np.ones((1,1)),1.)
        with self.assertRaises(ValueError):implicit_drag([0],[1],1,1,-np.ones((1,1)),1.)


class PoroTests(unittest.TestCase):
    def setUp(self):
        self.g=Grid((8,));self.s=Skeleton(self.g,AnisotropicMaterialParams(10,20,0))
        self.b=Biot(self.s,Darcy(self.g,[[.01]]),storage=.001)
        self.bc={(0,0):('flux',0.),(0,1):('pressure',0.)};self.load=self.s.traction(0,1,[-.1])

    def test_consolidation_matches_independent_series(self):
        old=self.b.initial(np.full(8,.1/1.04),self.load)
        for _ in range(100):
            r=self.b.step(old,.002,self.load,self.bc);old=r.state
            self.assertTrue(r.converged)
            for key in ['mass_defect','energy_residual','momentum_balance']: self.assertLess(abs(r.metrics[key]),1e-7)
        ref=consolidation_average(8,.2)
        self.assertLess(np.linalg.norm(old.p-ref)/np.linalg.norm(ref),.04)

    def test_fixed_stress_converges_and_once_does_not_commit(self):
        old=self.b.initial();direct=self.b.step(old,.01,self.load,self.bc)
        split=self.b.step(old,.01,self.load,self.bc,method='fixed_stress',tolerance=1e-10)
        self.assertTrue(split.converged);np.testing.assert_allclose(split.state.p,direct.state.p,atol=1e-6)
        tx=CouplingTransaction(self.b,old)
        with self.assertRaisesRegex(RuntimeError,'not converged'):
            tx.begin_trial(dt=.01,load=self.load,boundary=self.bc,method='once')
        with self.assertRaises(RuntimeError):tx.commit()
        np.testing.assert_array_equal(tx.committed.u,old.u)
        tx.begin_trial(dt=.01,load=self.load,boundary=self.bc);tx.rollback()
        np.testing.assert_array_equal(tx.committed.p,old.p)
        tx.begin_trial(dt=.01,load=self.load,boundary=self.bc);tx.commit()
        np.testing.assert_allclose(tx.committed.p,direct.state.p)

    def test_zero_coupling_is_same_solid(self):
        s=Skeleton(self.g,AnisotropicMaterialParams(10,20,200),alpha=0)
        b=Biot(s,Darcy(self.g,[[.01]]),storage=.001)
        r=b.step(b.initial(),.01,self.load,self.bc)
        np.testing.assert_allclose(r.state.u,s.equilibrate(np.zeros(self.g.nc),self.load),atol=1e-12)
        np.testing.assert_allclose(r.state.p,0.,atol=1e-12)

    def test_zero_storage_and_pressure_mechanical_rank(self):
        g=Grid((3,3));s=Skeleton(g,AnisotropicMaterialParams(10,20,200,[1,1,0]))
        b=Biot(s,Darcy(g,oriented_permeability(.01,1e-6,.7)),storage=0)
        eig=np.linalg.eigvalsh(b.pressure_mechanical_schur())
        self.assertGreater(eig.min(),1e-7)
        r=b.step(b.initial(),.02,s.traction(0,1,[-.1,0]),boundary(2))
        self.assertTrue(r.converged);self.assertLess(abs(r.metrics['energy_residual']),1e-8)


class StateTests(unittest.TestCase):
    def make(self):return SaturatedState([1.,.5],.35,np.diag([.1,.01,.01]),[[1,0,0],[0,1,0]])

    def test_rotation_expansion_and_recovery(self):
        s=self.make();initial=s.committed;R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        t=s.begin_trial(np.broadcast_to(R,(2,3,3)),[0,0]);s.commit()
        np.testing.assert_allclose(t['K'],R@initial['K']@R.T)
        F=np.broadcast_to(1.1*R,(2,3,3));inc=(1.1**3-1)*s.V
        t=s.begin_trial(F,inc);s.commit()
        self.assertTrue(np.all(t['n']>.35));np.testing.assert_allclose(t['solid_mass'],initial['solid_mass'])
        s.begin_trial(np.broadcast_to(np.eye(3),(2,3,3)),-inc);s.commit()
        for k in initial:np.testing.assert_allclose(s.committed[k],initial[k],atol=1e-9)

    def test_failed_trial_cannot_commit_or_mutate(self):
        s=self.make();old=s.committed;F=old['F'].copy();F[0]*=.5
        with self.assertRaises(ValueError):s.begin_trial(F,[0,0])
        with self.assertRaises(RuntimeError):s.commit()
        for k in old:np.testing.assert_array_equal(s.committed[k],old[k])
        F=old['F']*1.1
        with self.assertRaisesRegex(ValueError,'fluid mass'):s.begin_trial(F,[0,0])
        s.begin_trial(old['F'],[0,0]);s.rollback()
        with self.assertRaises(RuntimeError):s.commit()
        view=s.committed;view['F'][:]=0
        np.testing.assert_array_equal(s.committed['F'],old['F'])

    def test_checkpoint_continuation_and_bad_schema(self):
        s=self.make();F=s.committed['F']*1.02;inc=(1.02**3-1)*s.V
        s.begin_trial(F,inc);s.commit()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'restart.json';s.checkpoint(p);other=SaturatedState.restart(p)
            for obj in [s,other]: obj.begin_trial(np.tile(np.eye(3),(2,1,1)),-inc);obj.commit()
            for k in s.committed:np.testing.assert_array_equal(s.committed[k],other.committed[k])
            data=json.loads(p.read_text());data['payload']['schema_version']=99;p.write_text(json.dumps(data))
            with self.assertRaises(ValueError):SaturatedState.restart(p)


if __name__=='__main__':unittest.main()
