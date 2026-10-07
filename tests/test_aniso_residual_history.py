"""Energy, history, objectivity and massless-rank gates for local-history modes."""
import gc
from functools import partial
import unittest
import numpy as np
import warp as wp
from engine.types import vec3
from engine.sp_grid import B
from engine.aniso_phase1.residual_history import ResidualHistoryLiteSolver
from engine.aniso_phase1.operator_probe import SparseProbe
from engine.aniso_phase1.stabilization_probe import node_coordinates
from engine.aniso_phase1.consistent_transfer import material_response
from engine.aniso_phase1.beam_reference import reference_hessian
from benchmarks.aniso_residual_gate import geometry, matrix, rank_gate
from demos.aniso import Config, Scene

wp.config.kernel_cache_dir = '/tmp/mpm-lite-warp-cache'


class ResidualHistoryTests(unittest.TestCase):
    def tearDown(self):
        gc.collect()

    def probe(self, stabilization='none', boundary=True):
        return SparseProbe('cpu', 200., .005, boundary=boundary,
            solver_cls=partial(ResidualHistoryLiteSolver, stabilization=stabilization))

    def test_local_energy_keeps_variance_and_history_closes(self):
        p = self.probe()
        s = p.s
        F = s.ptc_F.numpy(); F[:, 0, 0] += np.linspace(-.04, .04, len(F))
        s.ptc_F.assign(F); s._mapped_ready = False
        p.residual(np.zeros((p.n, 3)))
        expected = np.asarray(s.mapped_W @ F.reshape(len(F), 9)).reshape(-1, 3, 3)
        np.testing.assert_allclose(s.mapped_trial.numpy(), expected, atol=1e-14)
        np.testing.assert_allclose(s.local_trial.numpy(), F, atol=1e-14)
        particle_psi, _ = material_response(F, s.ptc_A0.numpy(), s.aniso_params)
        center_psi, _ = material_response(expected, s.mapped_A.numpy(), s.aniso_params)
        energy = float(s.ptc_vol0.numpy() @ particle_psi)
        self.assertGreater(energy, float(s.mapped_V.numpy() @ center_psi))
        s._potential_sum.zero_(); s._elastic_potential()
        self.assertAlmostEqual(float(s._potential_sum.numpy()[0]), energy, delta=1e-15)
        # The exact pairwise residual formula reduces to individual particle energy.
        paired = float(s.mapped_V.numpy() @ (s.mapped_W @ particle_psi))
        self.assertAlmostEqual(paired, energy, delta=1e-15)

    def test_potential_exact_tangent_and_symmetry_both_modes(self):
        rng = np.random.default_rng(903)
        for mode in ('none', 'corotated'):
            p = self.probe(mode)
            s = p.s
            F = s.ptc_F.numpy(); F[:, 0, 0] = .92; F[:, 0, 1] = np.linspace(-.02, .02, len(F))
            s.ptc_F.assign(F); s.step(max_iters=0, print_every=0)
            v = p.project(.05*rng.normal(size=(p.n, 3)))
            d = p.project(rng.normal(size=v.shape)); d /= np.linalg.norm(d)
            q = p.project(rng.normal(size=v.shape)); q /= np.linalg.norm(q)
            r = p.residual(v); H = p.tangent(d); Hq = p.tangent(q); eps = 1e-5
            rp = p.residual(v+eps*d); ep = s.incremental_potential()
            rm = p.residual(v-eps*d); em = s.incremental_potential()
            self.assertLess(abs((ep-em)/(2*eps)-np.sum(r*d))/max(abs(np.sum(r*d)), 1e-12), 1e-6)
            self.assertLess(np.linalg.norm((rp-rm)/(2*eps)-H)/np.linalg.norm(H), 1e-6)
            self.assertLess(abs(np.sum(q*H)-np.sum(d*Hq))/max(np.linalg.norm(H), 1e-20), 1e-10)
            p.residual(v)
            GN = p.tangent(d, True)
            self.assertGreater(float(np.sum(d*GN)), 0.)

    def test_finite_rigid_rotation_affine_patch_and_objectivity(self):
        for mode in ('none', 'corotated'):
            p = self.probe(mode, False); s = p.s
            x = node_coordinates(s)*s.dx
            a = .8; Q = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
            def evaluate(v):
                p.residual(v); s._potential_sum.zero_(); s._elastic_potential()
                force = s.projected_internal_force()
                if s.enhancements is not None:
                    force += s.enhancements.extra[:p.n].numpy()/s.dt
                return float(s._potential_sum.numpy()[0]), force
            rigid = ((x-.5) @ Q.T+.5+[.01, -.02, .003]-x)/s.dt
            E, force = evaluate(rigid)
            self.assertLess(abs(E), 1e-20)
            self.assertLess(np.linalg.norm(force), 1e-10)
            A = np.array([[.03, .01, 0.], [-.01, -.02, .003], [0., .005, .01]])
            affine = (x-.5) @ A.T/s.dt
            evaluate(affine)
            np.testing.assert_allclose(s.local_trial.numpy(), np.broadcast_to(np.eye(3)+A, (s.n_ptc, 3, 3)), atol=1e-13)
            if s.enhancements is not None:
                self.assertLess(abs(float(s.enhancements.hg_energy.numpy()[0])), 1e-20)
            v = .1*np.random.default_rng(87).normal(size=x.shape)
            E, f = evaluate(v)
            rotated = ((x+s.dt*v-.5) @ Q.T+.5-x)/s.dt
            Er, fr = evaluate(rotated)
            self.assertLess(abs(Er-E)/max(abs(E), 1e-20), 1e-8)
            self.assertLess(np.linalg.norm(fr-f @ Q.T)/np.linalg.norm(f), 1e-8)

    def test_massless_rank_and_actual_tangent_four_directions(self):
        rng = np.random.default_rng(31); g = geometry(9)
        for kf, angle in ((0., 0.), (200., 0.), (200., 45.), (200., 90.)):
            a = np.deg2rad(angle); H = reference_hessian(kf=kf, direction=(np.cos(a), np.sin(a), 0.))
            for mode in ('none', 'corotated'):
                name = 'residual_center' if mode == 'none' else 'residual_corotated'
                K = matrix(g, H, name); gate = rank_gate(g, K, True)
                self.assertEqual(gate['passed'], mode == 'corotated', gate)
                scene = Scene(Config('tensile', 9, .005, angle, kf=kf, history_consistency='residual_center', stabilization=mode), 'cpu')
                s = scene.solver; s.step(max_iters=0, print_every=0); self.assertTrue(s.evaluate_residual())
                n = int(s.n_active_nodes.numpy()[0]); coords = node_coordinates(s)
                lookup = {tuple(np.rint(x/s.dx).astype(int)): i for i, x in enumerate(g['nodes'])}
                perm = np.array([lookup[tuple(x)] for x in coords])
                u = rng.normal(size=(len(g['nodes']), 3)); u[g['fixed']] = 0.
                p = wp.zeros_like(s.node_residual); out = wp.zeros_like(p)
                wp.copy(p, wp.array(u[perm], dtype=vec3, device='cpu'), count=n); s.apply_tangent(p, out)
                addr = s.ndof2bijk[:n].numpy(); local = addr[:, 1]
                mass = s.grid_m[:s.bcn].numpy()[addr[:, 0], local//(B*B), (local//B) % B, local % B]
                actual = (out[:n].numpy()-mass[:, None]*u[perm])/s.dt**2
                expected = (K @ u.T.ravel()).reshape(3, -1).T[perm]; expected[g['fixed'][perm]] = 0.
                self.assertLess(np.linalg.norm(actual-expected)/np.linalg.norm(expected), 1e-10)

    def test_success_commit_energy_reaction_and_rollback(self):
        for mode in ('none', 'corotated'):
            scene = Scene(Config('tensile', 9, .001, 45., smooth_loading=True,
                history_consistency='residual_center', stabilization=mode, boundary_impulse_transfer=True,
                apic_transfer='incremental', reaction_force_atol=1e-7), 'cpu')
            s = scene.solver
            for _ in range(5):
                self.assertTrue(scene.step(), s.last_step_stats)
                np.testing.assert_allclose(s.ptc_F.numpy(), s.local_trial.numpy(), atol=1e-13)
                expected = (s.mapped_W @ s.ptc_F.numpy().reshape(s.n_ptc, 9)).reshape(-1, 3, 3)
                np.testing.assert_allclose(expected, s.mapped_trial.numpy(), atol=1e-13)
                self.assertLess(scene.loading_rows[-1]['particle_momentum_balance_error_norm'], 1e-7)
                self.assertLess(abs(s.energy_ledger.rows[-1]['budget_closure']), 1e-14)
            fields = ('ptc_x', 'ptc_v', 'ptc_C', 'ptc_L', 'ptc_F')
            old = {k: getattr(s, k).numpy().copy() for k in fields}; clock = s.sim_time, s.sim_steps
            self.assertFalse(s.step(max_iters=0, print_every=0))
            for k in fields:
                np.testing.assert_array_equal(old[k], getattr(s, k).numpy())
            self.assertEqual(clock, (s.sim_time, s.sim_steps))

    def test_invalid_combinations_rejected(self):
        for options in ({'stabilization': 'hourglass'}, {'history_mode': 'grid_locked'},
                        {'force_discretization': 'legacy_kirchhoff'}, {'direction_model': 'fourth_moment'}):
            with self.assertRaises(ValueError):
                ResidualHistoryLiteSolver((9,)*3, Config().params, **options)


if __name__ == '__main__':
    unittest.main()
