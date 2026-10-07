"""Independent projection, variational and lifecycle checks for the new map."""
import gc
import json
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from engine.aniso_phase1.projected_history import ProjectedHistoryLiteSolver, frozen_maps
from engine.aniso_phase1.operator_probe import SparseProbe
from benchmarks.aniso_apic_frequency import Oracle
from demos.aniso import Config, Scene

wp.config.kernel_cache_dir = '/tmp/mpm-lite-warp-cache'
ROOT = Path(__file__).resolve().parents[1]


class ProjectedHistoryTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def snapshot(self):
        path = ROOT/'docs/results/lite-aniso-mainline/v8/cases/incremental-fourth/audit-02000.npz'
        with np.load(path) as z:
            return {k: z[k].copy() for k in z.files}

    def test_same_state_maps_match_independent_particle_update(self):
        z = self.snapshot()
        o = Oracle(z['particle_x_before'], z['particle_mass'], .125)
        maps, F0, V, W, S, D = frozen_maps(z['particle_x_before'], z['particle_volume'],
            z['particle_F_before'], o.c, o.nodes, .125)
        lookup = {tuple(n): i for i, n in enumerate(z['grid_nodes'])}
        v = z['grid_velocity_new'][[lookup[tuple(n)] for n in o.nodes]]
        F = F0 + .000125*np.stack([m @ v for m in maps], axis=2)
        L = np.einsum('pc,cnj,ni->pij', o.S, o.D, v)
        expected_particle = (np.eye(3)+.000125*L)@z['particle_F_before']
        expected = np.einsum('cp,pij->cij', o.W, expected_particle)
        np.testing.assert_allclose(F, expected, rtol=0, atol=2e-14)
        np.testing.assert_allclose(expected_particle, z['particle_F_after'], rtol=0, atol=2e-14)
        # Translation and a full 3D affine field are reproduced, also with nonuniform F.
        A = np.array([[.02,.01,-.005],[-.004,.006,.002],[.003,-.002,-.01]])
        affine = o.xn@A.T + [.01,-.02,.03]
        result = F0 + .01*np.stack([m@affine for m in maps], axis=2)
        np.testing.assert_allclose(result, (np.eye(3)+.01*A)@F0, rtol=0, atol=2e-14)
        np.testing.assert_allclose(W.sum(axis=1), 1, rtol=0, atol=1e-14)
        self.assertAlmostEqual(V.sum(), z['particle_volume'].sum(), delta=1e-15)

    def test_potential_gradient_tangent_and_symmetry(self):
        probe = SparseProbe('cpu', 200., .005, solver_cls=ProjectedHistoryLiteSolver)
        result = probe.variational_check()
        self.assertLess(min(result['potential_fd_errors'].values()), 1e-7)
        self.assertLess(min(result['tangent_fd_errors'].values()), 1e-8)
        self.assertLess(result['exact']['symmetry_relative'], 1e-12)
        self.assertGreater(result['modified']['min_eigenvalue'], 0.)

    def test_success_history_closure_momentum_and_failure_rollback(self):
        scene = Scene(Config('tensile',9,.001,45.,smooth_loading=True,
            boundary_impulse_transfer=True,apic_transfer='incremental',reaction_force_atol=1e-7,
            history_consistency='projected_center'), 'cpu')
        s=scene.solver
        for _ in range(5):
            self.assertTrue(scene.step(),s.last_step_stats)
            expected=np.asarray(s.mapped_W@s.ptc_F.numpy().reshape(s.n_ptc,9)).reshape(-1,3,3)
            np.testing.assert_allclose(s.mapped_trial.numpy(),expected,rtol=0,atol=1e-12)
            self.assertLess(scene.loading_rows[-1]['particle_momentum_balance_error_norm'],1e-7)
            self.assertLess(abs(s.energy_ledger.rows[-1]['budget_closure']),1e-14)
        fields=('ptc_x','ptc_v','ptc_C','ptc_L','ptc_F')
        before={k:getattr(s,k).numpy().copy() for k in fields}
        clock=s.sim_steps,s.sim_time
        self.assertFalse(s.step(max_iters=0,print_every=0))
        for k in fields:np.testing.assert_array_equal(getattr(s,k).numpy(),before[k])
        self.assertEqual(clock,(s.sim_steps,s.sim_time))

    def test_nonuniform_history_and_compression_derivatives(self):
        probe=SparseProbe('cpu',200.,.005,solver_cls=ProjectedHistoryLiteSolver)
        probe.set_deformation(np.diag([.65,.85,1.]))
        s=probe.s
        F=s.ptc_F.numpy().copy();F[:,0,1]=np.linspace(-.03,.03,len(F));s.ptc_F.assign(F)
        rng=np.random.default_rng(78)
        v=probe.project(.01*rng.normal(size=(probe.n,3)))
        p=probe.project(rng.normal(size=(probe.n,3)));p/=np.linalg.norm(p)
        r=probe.residual(v);Ap=probe.tangent(p)
        slope=float(np.sum(r*p));eps=1e-5
        rp=probe.residual(v+eps*p);ep=s.incremental_potential()
        rm=probe.residual(v-eps*p);em=s.incremental_potential()
        self.assertLess(abs((ep-em)/(2*eps)-slope)/max(abs(slope),1e-20),1e-7)
        self.assertLess(np.linalg.norm((rp-rm)/(2*eps)-Ap)/np.linalg.norm(Ap),1e-8)

    def test_invalid_combinations_and_support_rejected(self):
        for opts in ({'stabilization':'quadratic'},{'history_mode':'grid_locked'},
                     {'direction_model':'fourth_moment'},{'force_discretization':'legacy_kirchhoff'}):
            with self.assertRaises(ValueError):
                ProjectedHistoryLiteSolver((9,)*3,Config().params,**opts)
        z=self.snapshot();o=Oracle(z['particle_x_before'],z['particle_mass'],.125)
        with self.assertRaisesRegex(ValueError,'grid support'):
            frozen_maps(z['particle_x_before'],z['particle_volume'],z['particle_F_before'],o.c,o.nodes[:-1],.125)
        with self.assertRaises(ValueError):
            frozen_maps(z['particle_x_before'],z['particle_volume'],z['particle_F_before'],o.c,o.nodes,0.)


if __name__=='__main__':unittest.main()
