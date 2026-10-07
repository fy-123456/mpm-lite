"""Persistent center-state storage used by the stage-one solver adapter."""

from __future__ import annotations

import numpy as np
import warp as wp

from engine.types import mat33, real, vec3


@wp.kernel
def initialize_center_state_kernel(
    committed_F: wp.array(dtype=mat33),
    trial_F: wp.array(dtype=mat33),
    A0: wp.array(dtype=mat33),
    valid: wp.array(dtype=wp.int32),
    direction: wp.vec3d,
):
    c = wp.tid()
    I = wp.identity(3, dtype=real)
    committed_F[c] = I
    trial_F[c] = I
    A0[c] = wp.outer(direction, direction)
    valid[c] = 0


@wp.kernel
def begin_center_trial_kernel(
    committed_F: wp.array(dtype=mat33),
    trial_F: wp.array(dtype=mat33),
    grad_v: wp.array(dtype=mat33),
    dt: real,
):
    c = wp.tid()
    trial_F[c] = (wp.identity(3, dtype=real) + dt * grad_v[c]) @ committed_F[c]


@wp.kernel
def rollback_center_trial_kernel(
    committed_F: wp.array(dtype=mat33),
    trial_F: wp.array(dtype=mat33),
):
    c = wp.tid()
    trial_F[c] = committed_F[c]


class AnisotropicCenterBuffer:
    """Device-resident committed/trial center state.

    The buffer intentionally has no particle inputs.  A solver adapter can
    fill ``grad_v`` from the grid, run ``begin_trial`` once per Newton iterate,
    and call ``commit`` only after convergence.
    """

    def __init__(self, count: int, device: str = "cpu") -> None:
        if count <= 0:
            raise ValueError("center count must be positive")
        self.count = int(count)
        self.device = device
        self.committed_F = wp.zeros(count, dtype=mat33, device=device)
        self.trial_F = wp.zeros(count, dtype=mat33, device=device)
        self.A0 = wp.zeros(count, dtype=mat33, device=device)
        self.valid = wp.zeros(count, dtype=wp.int32, device=device)
        self.trial_tau = wp.zeros(count, dtype=mat33, device=device)

    def initialize(self, direction: np.ndarray) -> None:
        direction = np.asarray(direction, dtype=np.float64)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError("direction must be a finite 3-vector")
        norm = np.linalg.norm(direction)
        if norm <= 0.0:
            raise ValueError("direction must have non-zero length")
        direction = direction / norm
        wp.launch(
            initialize_center_state_kernel,
            dim=self.count,
            inputs=[self.committed_F, self.trial_F, self.A0, self.valid, wp.vec3d(*direction)],
            device=self.device,
        )

    def begin_trial(self, grad_v: wp.array, dt: float) -> None:
        if grad_v.shape[0] != self.count:
            raise ValueError("grad_v length must match center count")
        wp.launch(begin_center_trial_kernel, dim=self.count, inputs=[self.committed_F, self.trial_F, grad_v, dt], device=self.device)

    def rollback(self) -> None:
        wp.launch(rollback_center_trial_kernel, dim=self.count, inputs=[self.committed_F, self.trial_F], device=self.device)
        self.trial_tau.zero_()

    def commit(self) -> None:
        self.committed_F.assign(self.trial_F)
        self.valid.fill_(1)

    def evaluate_stress(self, mu: float, lam: float, k_f: float) -> None:
        """Evaluate center Kirchhoff stress from the current trial state."""
        from .kernels import center_stress_kernel

        wp.launch(
            center_stress_kernel,
            dim=self.count,
            inputs=[self.trial_F, self.A0, self.trial_tau, mu, lam, k_f],
            device=self.device,
        )

    def residual(
        self,
        node_velocity: wp.array,
        node_velocity_old: wp.array,
        node_mass: wp.array,
        gravity: np.ndarray,
        dt: float,
        center_volume: wp.array,
        grad_weights: wp.array,
        mu: float,
        lam: float,
        k_f: float,
    ) -> wp.array:
        """Evaluate the dense-center residual on the selected device."""
        from .kernels import center_residual_kernel

        n = node_velocity.shape[0]
        residual = wp.zeros(n, dtype=vec3, device=self.device)
        g = np.asarray(gravity, dtype=np.float64)
        if g.shape != (3,):
            raise ValueError("gravity must have shape (3,)")
        wp.launch(
            center_residual_kernel,
            dim=n,
            inputs=[
                node_velocity, node_velocity_old, node_mass, wp.vec3d(*g), dt,
                self.trial_F, self.A0, center_volume, grad_weights, residual,
                mu, lam, k_f,
            ],
            device=self.device,
        )
        return residual

    def matvec(
        self,
        direction: wp.array,
        node_mass: wp.array,
        dt: float,
        grad_weights: wp.array,
        mu: float,
        lam: float,
        k_f: float,
        center_volume: wp.array,
    ) -> wp.array:
        """Apply the Jacobian at the frozen Newton trial state on device."""
        from .kernels import center_gradient_from_nodes_kernel, center_matvec_kernel

        n = direction.shape[0]
        center_grad = wp.zeros(self.count, dtype=mat33, device=self.device)
        out = wp.zeros(n, dtype=vec3, device=self.device)
        wp.launch(
            center_gradient_from_nodes_kernel,
            dim=self.count,
            inputs=[direction, grad_weights, center_grad],
            device=self.device,
        )
        wp.launch(
            center_matvec_kernel,
            dim=n,
            inputs=[
                direction, node_mass, dt, self.committed_F, self.trial_F, self.A0,
                center_volume, grad_weights, center_grad, out,
                mu, lam, k_f,
            ],
            device=self.device,
        )
        return out
