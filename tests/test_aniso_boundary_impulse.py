import gc
import unittest
import numpy as np
import warp as wp
from demos.aniso import Config, Scene
from engine.sp_grid import B
from engine.types import vec3
from engine.kernel.d3.kernel_lite import lite_g2c_kernel, lite_c2p_kernel
from engine.aniso_phase1.tensile import set_grip_velocity, grid_values
from benchmarks.aniso_transfer_probes import manufactured_probe

wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'


def transfer_only(beta, enabled, boundary_type=1, moving=True):
    scene=Scene(Config('tensile',9,.001,flip_ratio=beta,boundary_impulse_transfer=enabled),'cpu')
    s=scene.solver
    initial=np.array([.017,-.011,.009]);s.ptc_v.assign(np.tile(initial,(s.n_ptc,1)))
    nodes=scene.boundary_indices
    normal=np.tile([0.,1.,0.],(len(nodes),1))
    value=np.tile([.031,.023,-.019] if moving else [0.,0.,0.],(len(nodes),1))
    s.paint_boundary(nodes,np.full(len(nodes),boundary_type,dtype=np.int32),normal,value)
    # Build actual P2C/C2G state without a solve or particle commit.
    if s.step(max_iters=0,print_every=0):raise AssertionError('unexpected commit')
    ledger=s.energy_ledger
    before=np.sum(s.ptc_m.numpy()[:,None]*s.ptc_v.numpy(),axis=0)
    raw=np.sum(ledger.node_m[:,None]*ledger.raw_v,axis=0)
    projected=np.sum(s.grid_m[:s.bcn].numpy()[...,None]*s.grid_v[:s.bcn].numpy(),axis=(0,1,2,3))
    capture_error=0.
    if enabled:capture_error=float(np.max(abs(grid_values(s,s.grid_v_raw,ledger.nodes)-ledger.raw_v)))
    wp.copy(s.grid_v_new,s.grid_v)
    wp.launch(lite_g2c_kernel,dim=(s.bcn,B,B,B),inputs=[s.block_count,s.block2bid,s.block_xyz_by_id,
        s.grid_v_raw if enabled else s.grid_v,s.grid_v_new,s.center_v,s.center_dv,s.center_G,s.center_size,s.dx],device='cpu')
    wp.launch(lite_c2p_kernel,dim=s.n_ptc,inputs=[s.block2bid,s.ptc_x,s.ptc_v,s.ptc_k,s.ptc_F,
        s.ptc_G,s.ptc_dlogJ,s.center_m,s.center_v,s.center_dv,s.center_G,s.psi_params,s.center_size,s.dx,0.,beta],device='cpu')
    velocities=s.ptc_v.numpy();after=np.sum(s.ptc_m.numpy()[:,None]*velocities,axis=0)
    result=dict(before=before,raw=raw,projected=projected,after=after,velocities=velocities,capture_error=capture_error)
    del s,scene;gc.collect();return result


class BoundaryImpulseTests(unittest.TestCase):
    def tearDown(self):gc.collect()

    def test_single_impulse_all_flip_ratios(self):
        for beta in (0.,.9,1.):
            results=[]
            for enabled in (False,True):
                with self.subTest(beta=beta,enabled=enabled):
                    r=transfer_only(beta,enabled);results.append(r)
                    np.testing.assert_allclose(r['before'],r['raw'],atol=1e-17,rtol=0)
                    expected=r['projected'] if enabled else r['projected']-beta*(r['projected']-r['raw'])
                    np.testing.assert_allclose(r['after'],expected,atol=1e-17,rtol=0)
                    self.assertLess(r['capture_error'],1e-15)
                    if enabled:
                        impulse=r['projected']-r['raw']
                        self.assertLess(np.linalg.norm((r['after']-r['before'])-impulse)/np.linalg.norm(impulse),1e-12)
            if beta==0:np.testing.assert_array_equal(results[0]['velocities'],results[1]['velocities'])

    def test_stationary_and_slip_boundaries(self):
        for btype,moving in ((1,False),(2,False),(2,True)):
            with self.subTest(btype=btype,moving=moving):
                r=transfer_only(.9,True,btype,moving)
                np.testing.assert_allclose(r['after'],r['projected'],atol=1e-17,rtol=0)
                if btype==2:np.testing.assert_allclose(r['velocities'][:,[0,2]],np.tile([.017,.009],(len(r['velocities']),1)),atol=1e-15,rtol=0)

    def test_zero_projection_is_identical(self):
        a=transfer_only(.9,False,0);b=transfer_only(.9,True,0)
        np.testing.assert_array_equal(a['velocities'],b['velocities'])
        np.testing.assert_allclose(b['after'],b['before'],atol=1e-17,rtol=0)

    def test_particle_force_balance_and_failed_step(self):
        scene=Scene(Config('tensile',9,.001,45.,smooth_loading=True,reaction_force_atol=1e-7,boundary_impulse_transfer=True),'cpu')
        for _ in range(4):
            self.assertTrue(scene.step())
            row=scene.loading_rows[-1]
            self.assertLess(row['particle_momentum_balance_error_norm'],1e-7)
            self.assertLess(abs(row['momentum_balance_error']),1e-7)
            self.assertLess(scene.solver.energy_ledger.rows[-1]['particle_grid_momentum_gap_norm'],1e-15)
        s=scene.solver;before=[a.numpy().copy() for a in (s.ptc_x,s.ptc_v,s.ptc_F)]
        clock=(s.sim_steps,s.sim_time)
        set_grip_velocity(s,scene.boundary_indices,.03)
        self.assertFalse(s.step(max_iters=0,print_every=0,reaction_force_atol=1e-7))
        for a,b in zip((s.ptc_x,s.ptc_v,s.ptc_F),before):np.testing.assert_array_equal(a.numpy(),b)
        self.assertEqual((s.sim_steps,s.sim_time),clock)

    def test_saved_initial_F_is_an_independent_snapshot(self):
        import tempfile
        import json
        from pathlib import Path
        from dataclasses import asdict
        from benchmarks.aniso_mainline import run_case
        config=Config('tensile',9,.001,45.,smooth_loading=True,
                      reaction_force_atol=1e-7,boundary_impulse_transfer=True)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            self.assertTrue(run_case(out,'snapshot',asdict(config),{'duration':.002,'device':'cpu'}))
            with np.load(out/'cases/snapshot/frames.npz') as frames:
                np.testing.assert_array_equal(frames['F'][0],np.broadcast_to(np.eye(3),frames['F'][0].shape))
                self.assertGreater(np.max(abs(frames['F'][-1]-frames['F'][0])),1e-8)
                self.assertEqual(frames['time'][0],0.)

    def test_unsupported_quadrature_rejected_before_step(self):
        scene=Scene(Config('tensile',9,.001,boundary_impulse_transfer=True),'cpu');s=scene.solver
        s.quadrature_kind='particle'
        with self.assertRaisesRegex(ValueError,'center quadrature'):s.step()
        self.assertEqual(s.sim_steps,0)


class AnalyticGradientTests(unittest.TestCase):
    def test_constant_and_affine(self):
        for field in ('constant','affine'):
            for offset in (.13,.73):
                r=manufactured_probe(8,offset=offset,field=field)
                self.assertLess(r['particle_gradient_max_error'],1e-13)
                self.assertLess(r['gradient_vs_pic_derivative_rms'],1e-13)
                self.assertLess(r['pic_velocity_oracle_max_error'],1e-14)
                self.assertLess(r['pic_derivative_fd_max_error'],1e-9)

    def test_sine_refinement_and_pic_derivative(self):
        for waves in (1,2):
            for offset in (.13,.73):
                rows=[manufactured_probe(n,waves,offset) for n in (8,16,32)]
                for r in rows:
                    self.assertLess(r['pic_velocity_oracle_max_error'],1e-14)
                    self.assertLess(r['pic_derivative_fd_max_error'],1e-9)
                    self.assertGreater(r['gradient_vs_pic_derivative_rms'],1e-6)
                for a,b in zip(rows,rows[1:]):
                    self.assertLess(b['particle_gradient_rms_error'],.4*a['particle_gradient_rms_error'])
                    self.assertLess(b['center_gradient_rms_error'],.4*a['center_gradient_rms_error'])
                    self.assertLess(b['pic_derivative_rms_error'],a['pic_derivative_rms_error'])


if __name__=='__main__':unittest.main()
