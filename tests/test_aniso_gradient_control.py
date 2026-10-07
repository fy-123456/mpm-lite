import gc
import itertools
import unittest
import numpy as np
import warp as wp
from demos.aniso import Config,Scene
from benchmarks.aniso_gradient_control import GradientControlLedger,sample_centers
from engine.aniso_phase1.transfer_audit import center_weights,average

wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'


class GradientControlTests(unittest.TestCase):
    def tearDown(self):gc.collect()

    def test_affine_and_nonlinear_interpolation_derivative(self):
        coords=np.array(list(itertools.product(range(5),repeat=3)));dx=.125
        x=np.array([[.2,.24,.27],[.31,.34,.38]])
        A=np.array([[.1,.03,.02],[.01,-.02,.05],[.02,.01,.04]])
        velocity=(coords+.5)*dx@A.T+np.array([.01,-.03,.02])
        v,G=sample_centers(x,coords,velocity,dx)
        np.testing.assert_allclose(G,np.broadcast_to(A,G.shape),atol=2e-15,rtol=0)
        np.testing.assert_allclose(v,x@A.T+np.array([.01,-.03,.02]),atol=2e-15,rtol=0)
        velocity[:,0]=np.sin(4*(coords[:,0]+.5)*dx)
        v,G=sample_centers(x,coords,velocity,dx);h=1e-6*dx
        fd=np.stack([(sample_centers(x+np.eye(3)[d]*h,coords,velocity,dx)[0]-sample_centers(x-np.eye(3)[d]*h,coords,velocity,dx)[0])/(2*h) for d in range(3)],axis=2)
        np.testing.assert_allclose(G,fd,atol=2e-9,rtol=0)
        with self.assertRaises(ValueError):sample_centers(x,coords[:1],velocity[:1],dx)

    def test_baseline_audit_read_only_and_decomposition(self):
        results=[]
        for audited in (False,True):
            scene=Scene(Config('tensile',9,.001,45.,smooth_loading=True,reaction_force_atol=1e-7,boundary_impulse_transfer=True),'cpu')
            if audited:scene.solver.energy_ledger=GradientControlLedger()
            for _ in range(3):
                if audited:scene.solver.energy_ledger.audit_next=True
                self.assertTrue(scene.step())
            results.append([getattr(scene.solver,'ptc_'+k).numpy().copy() for k in ('x','v','F','G')])
            if audited:
                ledger=scene.solver.energy_ledger;r=ledger.audit;snap=ledger.snapshot
                self.assertLess(r['frozen_decomposition_max_error'],1e-12)
                self.assertLess(r['pic_advection_oracle_max_error'],1e-14)
                W,_=center_weights(snap['particle_x_before'],snap['particle_volume'],snap['coords'],scene.solver.dx)
                np.testing.assert_allclose(average(W,snap['particle_F_after']),snap['frozen_F'],atol=1e-14,rtol=0)
                np.testing.assert_allclose(snap['without_gradient_gap_F']+snap['gradient_gap_increment'],snap['frozen_F'],atol=1e-12,rtol=0)
            del scene;gc.collect()
        for a,b in zip(*results):np.testing.assert_array_equal(a,b)

    def test_intervention_changes_only_history_on_same_solved_step(self):
        results=[]
        for mode in ('baseline','pic_history'):
            scene=Scene(Config('tensile',9,.001,45.,smooth_loading=True,reaction_force_atol=1e-7,boundary_impulse_transfer=True),'cpu')
            s=scene.solver;s.energy_ledger=GradientControlLedger(mode);s.energy_ledger.audit_next=True
            self.assertTrue(scene.step());ledger=s.energy_ledger
            result={k:getattr(s,'ptc_'+k).numpy().copy() for k in ('x','v','F','G')}
            result['center_F']=s.aniso_committed_F[:,:s.bcn].numpy().copy()
            result['reaction']=scene.loading_rows[-1]['right_force'];results.append(result)
            self.assertLess(scene.loading_rows[-1]['particle_momentum_balance_error_norm'],1e-7)
            if mode=='pic_history':
                np.testing.assert_array_equal(result['F'],ledger.snapshot['particle_F_pic'])
                before=result['F'].copy();t=s.sim_time
                self.assertFalse(s.step(max_iters=0,print_every=0,reaction_force_atol=1e-7))
                np.testing.assert_array_equal(s.ptc_F.numpy(),before);self.assertEqual(s.sim_time,t)
            del s,scene;gc.collect()
        for key in ('x','v','G','center_F','reaction'):np.testing.assert_array_equal(results[0][key],results[1][key])
        self.assertGreater(np.max(abs(results[0]['F']-results[1]['F'])),1e-10)


if __name__=='__main__':unittest.main()
