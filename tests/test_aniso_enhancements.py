"""Production sparse checks of optional moments and displacement stabilization."""
import gc,os,unittest
from functools import partial
import numpy as np
import warp as wp
from engine.types import mat33
from engine.aniso_phase1.solver import AnisotropicLiteImplicitSolver
from engine.aniso_phase1.operator_probe import SparseProbe
from engine.aniso_phase1.stabilization_probe import static_audit
from demos.aniso import Config,Scene


class EnhancementTests(unittest.TestCase):
    def setUp(self):
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        self.device=os.environ.get('ANISO_TEST_DEVICE','cpu')

    def tearDown(self):gc.collect()

    def probe(self,mode='supplemental'):
        p=SparseProbe(device=self.device,solver_cls=partial(AnisotropicLiteImplicitSolver,direction_model='fourth_moment',stabilization=mode))
        A=np.zeros((p.s.n_ptc,3,3));A[::2,0,0]=1;A[1::2,1,1]=1
        p.s.ptc_A0.assign(wp.array(A,dtype=mat33,device=self.device))
        p.s.step(max_iters=0,print_every=0)
        return p

    def test_mixed_material_energy_gradient_and_sparse_tangent(self):
        p=self.probe();p.set_deformation(np.array([[1.08,.03,0],[0,.98,.01],[0,0,1.]]))
        rng=np.random.default_rng(9);v=p.project(rng.normal(size=(p.n,3))*.02)
        d=p.project(rng.normal(size=v.shape));d/=np.linalg.norm(d)
        r=p.residual(v);Ad=p.tangent(d);eps=1e-5
        rp=p.residual(v+eps*d);ep=p.s.incremental_potential()
        rm=p.residual(v-eps*d);em=p.s.incremental_potential()
        self.assertLess(np.linalg.norm((rp-rm)/(2*eps)-Ad)/np.linalg.norm(Ad),1e-4)
        self.assertLess(abs((ep-em)/(2*eps)-np.sum(r*d)),1e-8)
        p.residual(v);q=p.project(rng.normal(size=v.shape));Aq=p.tangent(q)
        self.assertLess(abs(np.sum(q*Ad)-np.sum(d*Aq))/max(np.linalg.norm(Ad)*np.linalg.norm(q),1e-12),1e-7)
        projected=p.tangent(d,True);self.assertGreater(np.sum(d*projected),0.)

    def test_moment_resampling_matches_direction_distribution(self):
        p=self.probe('none');s=p.s
        M=s.enhancements.M.numpy();vol=s.center_vol[:,:int(s.bcn)].numpy().reshape(-1);active=vol>0
        A=s.aniso_A0[:,:int(s.bcn)].numpy().reshape(-1,9)[active]
        self.assertLess(np.max(np.abs(M[:,0,0]-A[:,0])),1e-7)
        self.assertLess(np.max(np.abs(M[:,4,4]-A[:,4])),1e-7)
        self.assertLess(np.max(np.abs(M[:,0,4])),1e-7)

    def test_static_modes_removed_and_actual_operator_matches_reference(self):
        results=static_audit(17,self.device)
        for r in results:
            self.assertLess(r['production_matvec_relative_error'],1e-5)
            if r['stabilization']=='none':self.assertGreater(r['soft_modes'],0)
            else:self.assertEqual(r['soft_modes'],0)

    def test_short_bent_beam_and_energy_budget(self):
        scene=Scene(Config('beam',17,.001,stabilization='supplemental',direction_model='fourth_moment',residual_atol=1e-8,cg_tol=1e-3),self.device)
        for _ in range(3):self.assertTrue(scene.step())
        row=scene.solver.energy_ledger.rows[-1]
        self.assertGreater(row['stabilization_energy'],0.)
        self.assertLess(abs(row['budget_closure']),1e-8)
        self.assertTrue(np.isfinite(scene.solver.ptc_x.numpy()).all())
        self.assertGreater(np.linalg.det(scene.solver.ptc_F.numpy()).min(),.9)


if __name__=='__main__':unittest.main()
