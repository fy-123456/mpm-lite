import os
import unittest

import numpy as np
import warp as wp

wp.config.kernel_cache_dir = os.environ.get("MPM_LITE_WARP_CACHE", "/tmp/mpm-lite-warp-cache")
wp.init()

from engine.aniso_phase1 import AnisotropicLiteImplicitSolver, AnisotropicMaterialParams  # noqa: E402
from engine.solver3d import MPMSolver  # noqa: E402
from engine.types import Material  # noqa: E402


from functools import partial
# Keep the original spatial-state lifecycle/legacy equivalence tests explicit.
# Default particle-carried history has separate transport and dynamics coverage.
GridLockedSolver = partial(AnisotropicLiteImplicitSolver, history_mode="grid_locked")


class TestAnisoPhase1Solver(unittest.TestCase):
    def test_kf_zero_matches_legacy_one_step(self):
        positions = np.array([[0.42, 0.42, 0.42], [0.46, 0.45, 0.44], [0.52, 0.48, 0.49], [0.58, 0.55, 0.54]], dtype=np.float64)
        E, nu = 1500.0, 0.3
        mu = E / (2.0 * (1.0 + nu))
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        boundary_ijk = np.array([[0, 0, 0]], dtype=np.int32)
        boundary = np.array([1], dtype=np.int32)

        legacy = MPMSolver((8, 8, 8), dx=1.0 / 7.0, device="cpu", gravity=-9.81, ppc=1, solver_type="lite_implicit")
        legacy.add_material(0, Material.elastic, E=E, nu=nu)
        legacy.seed_particles(positions, 0, 1000.0, 0.001)
        legacy.paint_boundary(boundary_ijk, boundary)
        legacy.set_dt(1.0e-4)
        legacy.step(max_iters=1, print_every=0, v_tol=1.0e-5)

        aniso = GridLockedSolver(
            (8, 8, 8), AnisotropicMaterialParams(mu, lam, 0.0, [1.0, 0.0, 0.0]),
            dx=1.0 / 7.0, device="cpu", gravity=-9.81, ppc=1,
        )
        aniso.seed_particles(positions, density=1000.0, vol0=0.001)
        aniso.paint_boundary(boundary_ijk, boundary)
        aniso.set_dt(1.0e-4)
        aniso.step(max_iters=1, print_every=0, v_tol=1.0e-5)

        self.assertTrue(np.allclose(legacy.ptc_x.numpy(), aniso.ptc_x.numpy(), atol=1.0e-12, rtol=0.0))
        self.assertTrue(np.allclose(legacy.ptc_v.numpy(), aniso.ptc_v.numpy(), atol=1.0e-12, rtol=0.0))
        self.assertTrue(np.allclose(legacy.ptc_F.numpy(), aniso.ptc_F.numpy(), atol=1.0e-12, rtol=0.0))

    def test_two_step_sparse_grid_cpu_smoke(self):
        solver = GridLockedSolver(
            (8, 8, 8),
            params=AnisotropicMaterialParams(500.0, 1000.0, 3000.0, [1.0, 0.0, 0.0]),
            dx=1.0 / 7.0,
            device="cpu",
            gravity=-9.81,
            ppc=1,
        )
        solver.seed_particles(np.array([[0.5, 0.5, 0.5]], dtype=np.float64), density=1000.0, vol0=0.01)
        solver.paint_boundary(np.array([[0, 0, 0]], dtype=np.int32), np.array([1], dtype=np.int32))
        solver.set_dt(1.0e-4)
        solver.step(max_iters=2, print_every=0, v_tol=1.0e-4)
        solver.step(max_iters=2, print_every=0, v_tol=1.0e-4)
        self.assertEqual(solver.sim_steps, 2)
        self.assertTrue(np.isfinite(solver.get_points()).all())
        self.assertGreater(int(solver.aniso_state_valid.numpy().sum()), 0)
        committed = solver.aniso_committed_F.numpy()
        self.assertTrue(np.isfinite(committed).all())
        self.assertTrue(np.allclose(solver.aniso_A0.numpy()[solver.aniso_state_valid.numpy() > 0][0], np.diag([1.0, 0.0, 0.0])))
        F_error = solver.center_particle_F_error()
        self.assertGreater(F_error["count"], 0)
        self.assertTrue(np.isfinite(F_error["max_frobenius"]))
        self.assertAlmostEqual(float(solver.center_m.numpy().sum()), float(solver.ptc_m.numpy().sum()), places=12)
        self.assertAlmostEqual(float(solver.center_vol.numpy().sum()), float(solver.ptc_vol0.numpy().sum()), places=12)
        stats = solver.last_step_stats
        for key in ("p2c_seconds", "cg_seconds", "material_seconds", "transfer_seconds", "step_seconds"):
            self.assertGreaterEqual(stats[key], 0.0)
        self.assertGreaterEqual(stats["matvec_calls"], 0)
        self.assertGreaterEqual(stats["cg_iterations"], 0)
        self.assertGreaterEqual(stats["newton_iterations"], 1)
        self.assertGreaterEqual(stats["last_residual_norm"], 0.0)
        self.assertGreater(stats["memory_bytes"], 0)
        self.assertEqual(solver.aniso_occupied_prev.numpy().dtype, np.uint8)
        self.assertLess(stats["memory_bytes"], 590 * 1024 * 1024)

    def test_resize_preserves_reactivation_count(self):
        solver = GridLockedSolver(
            (8, 8, 8),
            params=AnisotropicMaterialParams(10.0, 20.0, 30.0, [1.0, 0.0, 0.0]),
            dx=1.0 / 7.0,
            device="cpu",
            gravity=0.0,
            ppc=1,
        )
        solver.aniso_reactivation_count.assign(wp.array([7], dtype=wp.int32, device="cpu"))
        solver.MAX_BLOCKS = solver.aniso_committed_F.shape[1] * 2
        solver._resize_aniso_state_if_needed()
        self.assertEqual(int(solver.aniso_reactivation_count.numpy()[0]), 7)

    def test_uniform_three_dimensional_affine_field_converges(self):
        positions = np.array(
            [[0.20 + 0.12 * i, 0.20 + 0.12 * j, 0.20 + 0.12 * k]
             for i in range(4) for j in range(4) for k in range(4)],
            dtype=np.float64,
        )
        G = np.array(
            [[0.08, 0.02, 0.01], [0.00, -0.03, 0.015], [0.00, 0.00, 0.04]],
            dtype=np.float64,
        )
        solver = GridLockedSolver(
            (8, 8, 8),
            params=AnisotropicMaterialParams(10.0, 20.0, 30.0, [1.0, 0.0, 0.0]),
            dx=1.0 / 7.0,
            device="cpu",
            gravity=0.0,
            ppc=1,
        )
        solver.seed_particles(
            positions,
            density=1000.0,
            vol0=1.0e-3,
            velocity=positions @ G.T,
            velocity_gradient=G,
        )
        solver.set_dt(1.0e-3)
        expected = np.eye(3)
        for _ in range(3):
            self.assertTrue(solver.step(max_iters=5, print_every=0, v_tol=1.0e-8))
            expected = (np.eye(3) + solver.dt * G) @ expected
        valid = solver.aniso_state_valid.numpy()[0] > 0
        committed = solver.aniso_committed_F.numpy()[0][valid]
        self.assertEqual(committed.shape[0], 125)
        max_affine_error = np.max(np.linalg.norm(committed - expected, axis=(1, 2)))
        self.assertLess(max_affine_error, 5.0e-8)
        self.assertLess(solver.center_particle_F_error()["max_frobenius"], 2.0e-8)

    @unittest.skipUnless(
        any(str(device) == "cuda:1" for device in wp.get_devices()),
        "requires a visible nonzero CUDA ordinal",
    )
    def test_fixed_boundary_uses_selected_nonzero_cuda_device(self):
        solver = GridLockedSolver(
            (8, 8, 8),
            params=AnisotropicMaterialParams(10.0, 20.0, 200.0, [1.0, 0.0, 0.0]),
            dx=1.0 / 7.0,
            device="cuda:1",
            gravity=0.0,
            ppc=1,
        )
        solver.seed_particles(
            np.array([[0.45, 0.45, 0.45]], dtype=np.float64),
            density=1000.0,
            vol0=1.0e-3,
        )
        solver.paint_boundary(
            np.array([[1, 1, 1]], dtype=np.int32),
            np.array([1], dtype=np.int32),
        )
        self.assertEqual(solver.device, "cuda:1")

    def test_fixed_boundary_fiber_block_cpu_smoke(self):
        positions = np.array(
            [[0.20 + 0.12 * i, 0.20 + 0.12 * j, 0.20 + 0.12 * k]
             for i in range(4) for j in range(4) for k in range(4)],
            dtype=np.float64,
        )
        gradient = np.diag([0.08, 0.0, 0.0])
        boundary = np.array(
            [[1, j, k] for j in range(1, 7) for k in range(1, 7)],
            dtype=np.int32,
        )
        solver = GridLockedSolver(
            (8, 8, 8),
            params=AnisotropicMaterialParams(10.0, 20.0, 200.0, [1.0, 0.0, 0.0]),
            dx=1.0 / 7.0,
            device="cpu",
            gravity=0.0,
            ppc=1,
        )
        solver.seed_particles(
            positions,
            density=1000.0,
            vol0=1.0e-3,
            velocity=positions @ gradient.T,
            velocity_gradient=gradient,
        )
        solver.paint_boundary(boundary, np.ones(len(boundary), dtype=np.int32))
        solver.set_dt(1.0e-3)
        self.assertTrue(solver.step(max_iters=8, print_every=0, v_tol=1.0e-7))
        self.assertEqual(solver.last_step_stats["active_centers"], 125)
        self.assertGreater(solver.last_step_stats["matvec_calls"], 0)
        self.assertTrue(np.isfinite(solver.get_points()).all())
        valid = solver.aniso_state_valid.numpy()[0] > 0
        self.assertTrue(np.isfinite(solver.aniso_committed_F.numpy()[0][valid]).all())
        self.assertGreater(float(np.linalg.norm(solver.aniso_committed_F.numpy()[0][valid] - np.eye(3))), 0.0)

    def test_per_particle_directions_are_averaged_and_reported_as_mixed(self):
        solver = GridLockedSolver(
            (8, 8, 8),
            params=AnisotropicMaterialParams(10.0, 20.0, 30.0, [1.0, 0.0, 0.0]),
            dx=1.0 / 7.0,
            device="cpu",
            gravity=0.0,
            ppc=1,
        )
        positions = np.array([[0.46, 0.46, 0.46], [0.47, 0.47, 0.47]], dtype=np.float64)
        solver.seed_particles(
            positions,
            density=1000.0,
            vol0=1.0e-3,
            fiber_directions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )
        solver.set_dt(1.0e-4)
        self.assertTrue(solver.step(max_iters=1, print_every=0, v_tol=1.0e-7))
        mixing = solver.center_structure_tensor_mixing()
        self.assertGreater(mixing["count"], 0)
        self.assertGreater(mixing["max"], 0.1)

    def test_center_reference_direction_stays_fixed_after_activation(self):
        solver = GridLockedSolver(
            (8, 8, 8),
            params=AnisotropicMaterialParams(10.0, 20.0, 30.0, [1.0, 0.0, 0.0]),
            dx=1.0 / 7.0,
            device="cpu",
            gravity=0.0,
            ppc=1,
        )
        solver.seed_particles(
            np.array([[0.46, 0.46, 0.46], [0.47, 0.47, 0.47]], dtype=np.float64),
            density=1000.0,
            vol0=1.0e-3,
            fiber_directions=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        solver.set_dt(1.0e-4)
        self.assertTrue(solver.step(max_iters=1, print_every=0, v_tol=1.0e-7))
        before = solver.aniso_A0.numpy().copy()
        particle_A0 = solver.ptc_A0.numpy()
        particle_A0[:] = np.diag([0.0, 1.0, 0.0])
        solver.ptc_A0.assign(wp.array(particle_A0, dtype=solver.ptc_A0.dtype, device="cpu"))
        self.assertTrue(solver.step(max_iters=1, print_every=0, v_tol=1.0e-7))
        valid = solver.aniso_state_valid.numpy()[0] > 0
        np.testing.assert_allclose(solver.aniso_A0.numpy()[0][valid], before[0][valid])

    def test_reactivated_center_reinitializes_reference_direction_and_counts_event(self):
        solver = GridLockedSolver(
            (8, 8, 8),
            params=AnisotropicMaterialParams(10.0, 20.0, 30.0, [1.0, 0.0, 0.0]),
            dx=1.0 / 7.0,
            device="cpu",
            gravity=0.0,
            ppc=1,
        )
        position = np.array([[0.46, 0.46, 0.46]], dtype=np.float64)
        solver.seed_particles(position, density=1000.0, vol0=1.0e-3, fiber_directions=np.array([[1.0, 0.0, 0.0]]))
        solver.set_dt(1.0e-4)
        self.assertTrue(solver.step(max_iters=1, print_every=0, v_tol=1.0e-7))
        moved = np.array([[0.70, 0.70, 0.70]], dtype=np.float64)
        solver.ptc_x.assign(wp.array(moved, dtype=solver.ptc_x.dtype, device="cpu"))
        changed = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
        solver.ptc_A0.assign(wp.array(np.einsum("pi,pj->pij", changed, changed), dtype=solver.ptc_A0.dtype, device="cpu"))
        self.assertTrue(solver.step(max_iters=1, print_every=0, v_tol=1.0e-7))
        solver.ptc_x.assign(wp.array(position, dtype=solver.ptc_x.dtype, device="cpu"))
        self.assertTrue(solver.step(max_iters=1, print_every=0, v_tol=1.0e-7))
        self.assertGreater(solver.last_step_stats["reactivations"], 0)
        valid_A0 = solver.aniso_A0.numpy()[0][solver.aniso_state_valid.numpy()[0] > 0]
        self.assertTrue(any(np.allclose(A, np.diag([0.0, 1.0, 0.0])) for A in valid_A0))

    def test_block_renumbering_preserves_center_history(self):
        # Prefix-sum activation orders blocks by global coordinate.  Adding a
        # lower-coordinate block must not move the previous block's A0/F state
        # onto the new block.
        device = os.environ.get("ANISO_TEST_DEVICE", "cpu")
        solver = GridLockedSolver(
            (96, 96, 96),
            params=AnisotropicMaterialParams(10.0, 20.0, 30.0, [1.0, 0.0, 0.0]),
            dx=1.0 / 95.0,
            device=device,
            gravity=0.0,
            ppc=1,
        )
        positions = np.array([[0.50, 0.50, 0.50], [0.80, 0.50, 0.50]], dtype=np.float64)
        solver.seed_particles(
            positions,
            density=1.0,
            vol0=1.0e-3,
            fiber_directions=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )
        solver.set_dt(1.0e-4)
        self.assertTrue(solver.step(max_iters=1, print_every=0, v_tol=1.0e-7))

        moved = positions.copy()
        moved[1] = [0.05, 0.50, 0.50]
        particle_A0 = solver.ptc_A0.numpy()
        particle_A0[1] = np.diag([0.0, 0.0, 1.0])
        solver.ptc_A0.assign(wp.array(particle_A0, dtype=solver.ptc_A0.dtype, device=device))
        solver.ptc_x.assign(wp.array(moved, dtype=solver.ptc_x.dtype, device=device))
        self.assertTrue(solver.step(max_iters=1, print_every=0, v_tol=1.0e-7))

        block2bid = solver.block2bid.numpy()
        tensors = solver.aniso_A0.numpy()[0]
        for position, expected in ((moved[0], np.diag([1.0, 0.0, 0.0])), (moved[1], np.diag([0.0, 0.0, 1.0]))):
            center = np.floor(position / solver.dx - 0.5).astype(np.int64)
            bid = int(block2bid[tuple(center // 32)])
            local = center % 32
            actual = tensors[bid, local[0], local[1] * 32 + local[2]]
            np.testing.assert_allclose(actual, expected, atol=1.0e-12)

        # Returning to the retired block is a real reactivation event even
        # though its old prefix-sum slot no longer exists.
        moved[1] = positions[1]
        solver.ptc_x.assign(wp.array(moved, dtype=solver.ptc_x.dtype, device=device))
        self.assertTrue(solver.step(max_iters=1, print_every=0, v_tol=1.0e-7))
        self.assertGreater(solver.last_step_stats["reactivations"], 0)

    def test_zero_newton_iterations_do_not_commit_center_F(self):
        solver = GridLockedSolver(
            (8, 8, 8),
            params=AnisotropicMaterialParams(10.0, 20.0, 30.0, [0.0, 1.0, 0.0]),
            dx=1.0 / 7.0,
            device="cpu",
            gravity=0.0,
            ppc=1,
        )
        solver.seed_particles(np.array([[0.5, 0.5, 0.5]], dtype=np.float64), density=1000.0, vol0=0.01)
        solver.paint_boundary(np.array([[0, 0, 0]], dtype=np.int32), np.array([1], dtype=np.int32))
        solver.set_dt(1.0e-4)
        before = solver.get_points().copy()
        solver.step(max_iters=0, print_every=0)
        valid = solver.aniso_state_valid.numpy() > 0
        self.assertGreater(int(valid.sum()), 0)
        self.assertTrue(np.allclose(solver.aniso_committed_F.numpy()[valid], np.eye(3)))
        self.assertTrue(np.allclose(solver.aniso_trial_F.numpy()[valid], solver.aniso_committed_F.numpy()[valid]))
        self.assertEqual(solver.sim_steps, 0)
        self.assertTrue(np.allclose(solver.get_points(), before))

    def test_negative_trial_jacobian_fails_without_particle_commit(self):
        solver = GridLockedSolver(
            (8, 8, 8),
            params=AnisotropicMaterialParams(10.0, 20.0, 30.0, [1.0, 0.0, 0.0]),
            dx=1.0 / 7.0,
            device="cpu",
            gravity=0.0,
            ppc=1,
        )
        solver.seed_particles(np.array([[0.5, 0.5, 0.5]], dtype=np.float64), density=1000.0, vol0=0.01)
        solver.set_dt(1.0e-4)
        self.assertTrue(solver.step(max_iters=1, print_every=0))
        before = solver.get_points().copy()
        host_F = solver.aniso_committed_F.numpy()
        valid = solver.aniso_state_valid.numpy()[0] > 0
        bid, lci, lcjk = np.argwhere(valid)[0]
        host_F[0, bid, lci, lcjk] = np.diag([-1.0, 1.0, 1.0])
        solver.aniso_committed_F.assign(wp.array(host_F, dtype=solver.aniso_committed_F.dtype, device="cpu"))
        self.assertFalse(solver.step(max_iters=1, print_every=0))
        self.assertEqual(solver.sim_steps, 1)
        self.assertTrue(np.allclose(solver.get_points(), before))
        self.assertTrue(np.isinf(solver.last_step_stats["last_residual_norm"]))


if __name__ == "__main__":
    unittest.main()
