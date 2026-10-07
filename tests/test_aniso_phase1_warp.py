import os
import unittest

import numpy as np
import warp as wp

wp.config.kernel_cache_dir = os.environ.get("MPM_LITE_WARP_CACHE", "/tmp/mpm-lite-warp-cache")
wp.init()

from engine.aniso_phase1 import AnisotropicCenterBuffer, AnisotropicMaterialParams, center_matvec, center_residual  # noqa: E402
from engine.aniso_phase1.kernels import (  # noqa: E402
    accumulate_structure_tensor_kernel,
    center_gradient_from_nodes_kernel,
    center_matvec_kernel,
    center_residual_kernel,
    center_trial_F_kernel,
    normalize_structure_tensor_kernel,
)
from engine.types import mat33, real, vec3  # noqa: E402


class TestAnisoPhase1Warp(unittest.TestCase):
    def test_structure_tensor_accumulation_is_unoriented(self):
        particle_A0 = wp.array(
            np.array([
                np.diag([1.0, 0.0, 0.0]),
                np.diag([1.0, 0.0, 0.0]),
                np.diag([0.0, 1.0, 0.0]),
            ]),
            dtype=mat33,
            device="cpu",
        )
        particle_center = wp.array(np.array([0, 0, 1], dtype=np.int32), dtype=wp.int32, device="cpu")
        particle_weight = wp.array(np.array([1.0, 2.0, 1.0]), dtype=real, device="cpu")
        sums = wp.zeros(2, dtype=mat33, device="cpu")
        weights = wp.zeros(2, dtype=real, device="cpu")
        center_A0 = wp.zeros(2, dtype=mat33, device="cpu")
        wp.launch(accumulate_structure_tensor_kernel, 3, [particle_A0, particle_center, particle_weight, sums, weights], device="cpu")
        wp.launch(normalize_structure_tensor_kernel, 2, [sums, weights, center_A0], device="cpu")
        result = center_A0.numpy()
        self.assertTrue(np.allclose(result[0], np.diag([1.0, 0.0, 0.0])))
        self.assertTrue(np.allclose(result[1], np.diag([0.0, 1.0, 0.0])))

    def test_cpu_center_kernels_match_reference(self):
        params = AnisotropicMaterialParams(3.0, 5.0, 17.0, [1.0, 2.0, 0.5])
        # A substantial nonzero Newton iterate separates trial from history.
        dt = 0.2
        F0 = np.stack([
            np.diag([1.1, 0.95, 1.03]),
            np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.05], [0.0, 0.0, 1.0]]),
        ])
        A0 = np.stack([params.A0, params.A0])
        volume = np.array([0.7, 0.4])
        mass = np.array([2.0, 1.5, 1.0])
        velocity = np.array([[0.2, -0.1, 0.3], [-0.05, 0.4, 0.1], [0.1, 0.0, -0.2]])
        velocity_old = np.zeros_like(velocity)
        gravity = np.array([0.0, 0.0, -9.81])
        grad_weights = np.array([
            [[-0.4, -0.2, -0.1], [0.4, 0.2, 0.1], [0.0, 0.0, 0.0]],
            [[-0.3, 0.1, -0.2], [0.3, -0.1, 0.2], [0.0, 0.0, 0.0]],
        ])
        grad_v = np.einsum("ni,cnj->cij", velocity, grad_weights)

        wf = wp.array(F0, dtype=mat33, device="cpu")
        wg = wp.array(grad_v, dtype=mat33, device="cpu")
        wtrial = wp.zeros(2, dtype=mat33, device="cpu")
        wp.launch(center_trial_F_kernel, 2, [wf, wg, wtrial, dt], device="cpu")

        wv = wp.array(velocity, dtype=vec3, device="cpu")
        wvo = wp.array(velocity_old, dtype=vec3, device="cpu")
        wm = wp.array(mass, dtype=real, device="cpu")
        wvol = wp.array(volume, dtype=real, device="cpu")
        wdw = wp.array(grad_weights, dtype=real, device="cpu")
        wa = wp.array(A0, dtype=mat33, device="cpu")
        wr = wp.zeros(3, dtype=vec3, device="cpu")
        wp.launch(center_residual_kernel, 3, [wv, wvo, wm, wp.vec3d(*gravity), dt, wtrial, wa, wvol, wdw, wr, params.mu, params.lam, params.k_f], device="cpu")
        reference = center_residual(velocity, velocity_old, mass, gravity, dt, F0, A0, volume, grad_weights, params)
        self.assertTrue(np.allclose(wr.numpy(), reference, rtol=1.0e-10, atol=1.0e-10))

        direction = np.array([[0.3, 0.2, -0.1], [-0.2, 0.1, 0.25], [0.05, -0.1, 0.2]])
        wd = wp.array(direction, dtype=vec3, device="cpu")
        wgc = wp.zeros(2, dtype=mat33, device="cpu")
        wp.launch(center_gradient_from_nodes_kernel, 2, [wd, wdw, wgc], device="cpu")
        wap = wp.zeros(3, dtype=vec3, device="cpu")
        wp.launch(center_matvec_kernel, 3, [wd, wm, dt, wf, wtrial, wa, wvol, wdw, wgc, wap, params.mu, params.lam, params.k_f], device="cpu")
        expected = center_matvec(direction, mass, dt, F0, A0, volume, grad_weights, params, trial_F=wtrial.numpy())
        self.assertTrue(np.allclose(wap.numpy(), expected, rtol=1.0e-8, atol=1.0e-9))

        eps = 1.0e-5
        plus = center_residual(velocity + eps * direction, velocity_old, mass, gravity, dt, F0, A0, volume, grad_weights, params)
        minus = center_residual(velocity - eps * direction, velocity_old, mass, gravity, dt, F0, A0, volume, grad_weights, params)
        np.testing.assert_allclose(wap.numpy(), (plus - minus) / (2 * eps), rtol=1.0e-8, atol=1.0e-9)

        # The device-resident state wrapper uses the same trial/commit-owned
        # arrays and exposes the operators without particle-level callbacks.
        state = AnisotropicCenterBuffer(2, device="cpu")
        state.committed_F.assign(wf)
        state.A0.assign(wa)
        state.begin_trial(wg, dt)
        wrapped_residual = state.residual(wv, wvo, wm, gravity, dt, wvol, wdw, params.mu, params.lam, params.k_f)
        wrapped_matvec = state.matvec(wd, wm, dt, wdw, params.mu, params.lam, params.k_f, wvol)
        self.assertTrue(np.allclose(wrapped_residual.numpy(), reference, rtol=1.0e-10, atol=1.0e-10))
        self.assertTrue(np.allclose(wrapped_matvec.numpy(), expected, rtol=1.0e-8, atol=1.0e-9))
        snapshot = (state.committed_F.numpy(), state.trial_F.numpy(), state.A0.numpy())
        repeated = state.matvec(wd, wm, dt, wdw, params.mu, params.lam, params.k_f, wvol)
        np.testing.assert_array_equal(repeated.numpy(), wrapped_matvec.numpy())
        for before, after in zip(snapshot, (state.committed_F, state.trial_F, state.A0)):
            np.testing.assert_array_equal(before, after.numpy())
        state.evaluate_stress(params.mu, params.lam, params.k_f)
        state.rollback()
        self.assertTrue(np.allclose(state.trial_F.numpy(), state.committed_F.numpy()))
        self.assertTrue(np.allclose(state.trial_tau.numpy(), 0.0))


if __name__ == "__main__":
    unittest.main()
