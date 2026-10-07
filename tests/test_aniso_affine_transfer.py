import gc
import unittest
import numpy as np
import warp as wp
from demos.aniso import Config,Scene
from engine.sp_grid import B
from engine.types import vec3
from engine.aniso_phase1.affine_transfer import prepare_incremental,finish_transfer
from engine.kernel.d3.kernel_lite import lite_g2c_kernel,lite_c2p_kernel
from benchmarks.aniso_apic_frequency import Oracle,initial_field

wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'


def transfer_probe(beta=.9,field='sine_45',boundary=0,dt=.001):
    scene=Scene(Config('tensile',9,.001,45.,flip_ratio=beta,boundary_impulse_transfer=True,apic_transfer='incremental'),'cpu')
    s=scene.solver;x=s.ptc_x.numpy().copy();F=s.ptc_F.numpy().copy();v,C=initial_field(x,field)
    s.ptc_v.assign(v);s.ptc_C.assign(C)
    if boundary:
        nodes=scene.boundary_indices
        s.paint_boundary(nodes,np.full(len(nodes),boundary,np.int32),np.tile([0.,1.,0.],(len(nodes),1)),np.tile([.017,-.008,.004],(len(nodes),1)))
    else:s.bc_type.zero_()
    assert not s.step(max_iters=0,print_every=0)
    wp.copy(s.grid_v_new,s.grid_v)
    o=Oracle(x,s.ptc_m.numpy(),s.dx);bmap=s.block2bid.numpy();b=o.nodes//B
    bid=bmap[b[:,0],b[:,1],b[:,2]];loc=o.nodes%B;ids=bid*B**3+loc[:,0]*B**2+loc[:,1]*B+loc[:,2]
    raw=s.grid_v_raw[:s.bcn].numpy().reshape(-1,3)[ids].copy()
    new=s.grid_v_new[:s.bcn].numpy().reshape(-1,3)[ids].copy()
    expected_L=np.einsum('pc,cnj,ni->pij',o.S,o.D,new)
    raw_L=np.einsum('pc,cnj,ni->pij',o.S,o.D,raw)
    expected_C=expected_L+beta*(C-raw_L)
    expected_v=beta*(v+o.S@o.H@(new-raw))+(1-beta)*(o.S@o.H@new)
    prepare_incremental(s)
    wp.launch(lite_g2c_kernel,dim=(s.bcn,B,B,B),inputs=[s.block_count,s.block2bid,s.block_xyz_by_id,s.grid_v_raw,s.grid_v_new,s.center_v,s.center_dv,s.center_G,s.center_size,s.dx],device='cpu')
    wp.launch(lite_c2p_kernel,dim=s.n_ptc,inputs=[s.block2bid,s.ptc_x,s.ptc_v,s.ptc_k,s.ptc_F,s.ptc_G,s.ptc_dlogJ,s.center_m,s.center_v,s.center_dv,s.center_G,s.psi_params,s.center_size,s.dx,dt,beta],device='cpu')
    finish_transfer(s)
    result=dict(C=s.ptc_C.numpy().copy(),L=s.ptc_L.numpy().copy(),v=s.ptc_v.numpy().copy(),F=s.ptc_F.numpy().copy(),
        old_C=C,old_v=v,expected_C=expected_C,expected_L=expected_L,expected_v=expected_v,
        expected_F=(np.eye(3)+dt*expected_L)@F,
        momentum_error=float(np.max(abs(o.m@(s.ptc_v.numpy()-v)-o.mn@(new-raw)))))
    del s,scene;gc.collect();return result


class AffineTransferTests(unittest.TestCase):
    def test_increment_oracle_boundary_impulse_and_F_uses_L(self):
        for boundary in (0,1,2):
            for beta in (0.,.9,1.):
                with self.subTest(boundary=boundary,beta=beta):
                    r=transfer_probe(beta,boundary=boundary)
                    for key in ('C','L','v','F'):np.testing.assert_allclose(r[key],r['expected_'+key],rtol=0,atol=1e-13)
                    self.assertLess(r['momentum_error'],1e-14)
                    if beta>0:self.assertGreater(np.max(abs(r['C']-r['L'])),1e-5)

    def test_affine_and_constant_preserved(self):
        for field in ('constant','affine'):
            r=transfer_probe(field=field)
            np.testing.assert_allclose(r['C'],r['old_C'],rtol=0,atol=1e-13)
            np.testing.assert_allclose(r['L'],r['old_C'],rtol=0,atol=1e-13)
            np.testing.assert_allclose(r['v'],r['old_v'],rtol=0,atol=1e-13)

    def test_pure_flip_no_impulse_preserves_affine_state(self):
        r=transfer_probe(beta=1.)
        np.testing.assert_allclose(r['C'],r['old_C'],rtol=0,atol=1e-15)
        np.testing.assert_array_equal(r['v'],r['old_v'])
        self.assertGreater(np.max(abs(r['L']-r['C'])),1e-5)

    def test_success_and_failure_keep_C_L_separate(self):
        scene=Scene(Config('tensile',9,.001,45.,smooth_loading=True,apic_transfer='incremental',boundary_impulse_transfer=True,reaction_force_atol=1e-7),'cpu');s=scene.solver
        for _ in range(3):
            F=s.ptc_F.numpy().copy();self.assertTrue(scene.step())
            np.testing.assert_allclose(s.ptc_F.numpy(),(np.eye(3)+s.dt*s.ptc_L.numpy())@F,rtol=0,atol=1e-14)
            self.assertLess(scene.loading_rows[-1]['particle_momentum_balance_error_norm'],1e-7)
        self.assertGreater(np.max(abs(s.ptc_C.numpy()-s.ptc_L.numpy())),1e-7)
        old=[a.numpy().copy() for a in (s.ptc_x,s.ptc_v,s.ptc_F,s.ptc_C,s.ptc_L)]
        clock=s.sim_steps,s.sim_time
        self.assertFalse(s.step(max_iters=0,print_every=0))
        for a,b in zip((s.ptc_x,s.ptc_v,s.ptc_F,s.ptc_C,s.ptc_L),old):np.testing.assert_array_equal(a.numpy(),b)
        self.assertEqual(clock,(s.sim_steps,s.sim_time))
        del s,scene;gc.collect()

    def test_seed_append_preserves_L_and_C_alias(self):
        scene=Scene(Config('tensile',9,.001),'cpu');s=scene.solver
        initial=s.ptc_L.numpy().copy();A=np.diag([.01,.02,-.03]);n=s.n_ptc
        s.seed_particles(np.array([[.51,.51,.51]]),0,1.,.0001,velocity_gradient=A)
        self.assertIs(s.ptc_C,s.ptc_G);self.assertIsNot(s.ptc_L,s.ptc_C)
        np.testing.assert_array_equal(s.ptc_L.numpy()[:n],initial)
        np.testing.assert_array_equal(s.ptc_L.numpy()[-1],A)
        np.testing.assert_array_equal(s.ptc_C.numpy()[-1],A)
        del s,scene;gc.collect()

    def test_invalid_mode_or_missing_boundary_correction_rejected(self):
        for mode,boundary in [('bad',True),('incremental',False)]:
            with self.assertRaises(ValueError):Scene(Config('tensile',9,.001,apic_transfer=mode,boundary_impulse_transfer=boundary),'cpu')
        scene=Scene(Config('tensile',9,.001,apic_transfer='incremental',boundary_impulse_transfer=True),'cpu');s=scene.solver
        s.quadrature_kind='particle'
        with self.assertRaises(ValueError):s.step()
        self.assertEqual(s.sim_steps,0)
        del s,scene;gc.collect()


if __name__=='__main__':unittest.main()
