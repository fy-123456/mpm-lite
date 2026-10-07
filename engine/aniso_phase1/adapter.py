"""Independent NumPy center adapter for the first anisotropic solver path."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .constitutive import a0_to_A0, average_structure_tensors, kirchhoff
from .implicit import center_matvec, center_residual, center_velocity_gradient, center_trial_F, matrix_free_pcg
from .types import AnisotropicMaterialParams


@dataclass
class AnisotropicCenterAdapter:
    """Persistent fixed-reference center quadrature state.

    The adapter is deliberately independent of the legacy ``MPMSolver``.  It
    mirrors the state ownership required by the plan and serves as the
    executable reference for the later sparse Warp integration.
    """

    center_count: int
    params: AnisotropicMaterialParams

    def __post_init__(self) -> None:
        if self.center_count <= 0:
            raise ValueError("center_count must be positive")
        self.center_count = int(self.center_count)
        self.committed_F = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], self.center_count, axis=0)
        self.trial_F = self.committed_F.copy()
        self.A0 = np.repeat(self.params.A0[None, :, :], self.center_count, axis=0)
        self.volume = np.zeros(self.center_count, dtype=np.float64)
        self.mass = np.zeros(self.center_count, dtype=np.float64)
        self.trial_tau = np.zeros_like(self.committed_F)
        self.valid = np.zeros(self.center_count, dtype=bool)
        self._trial_active = False

    def initialize_from_particles(
        self,
        particle_directions: np.ndarray,
        weights: np.ndarray,
        center_ids: np.ndarray,
    ) -> None:
        """Initialize center ``A0`` by weighted structure-tensor averaging."""
        directions = np.asarray(particle_directions, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        center_ids = np.asarray(center_ids, dtype=np.int64)
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError("particle_directions must have shape (P, 3)")
        if weights.shape != (directions.shape[0],) or center_ids.shape != weights.shape:
            raise ValueError("weights and center_ids must match particle count")
        if np.any(center_ids < 0) or np.any(center_ids >= self.center_count):
            raise ValueError("center_ids contains an invalid center")
        particle_A0 = np.stack([a0_to_A0(a) for a in directions])
        for c in range(self.center_count):
            mask = center_ids == c
            if np.any(mask & (weights > 0.0)):
                self.A0[c] = average_structure_tensors(particle_A0[mask], weights[mask])
                self.valid[c] = True

    def set_center_measurements(self, volume: np.ndarray, mass: np.ndarray) -> None:
        volume = np.asarray(volume, dtype=np.float64)
        mass = np.asarray(mass, dtype=np.float64)
        if volume.shape != (self.center_count,) or mass.shape != volume.shape:
            raise ValueError("volume and mass must match center_count")
        if np.any(volume < 0.0) or np.any(mass < 0.0):
            raise ValueError("volume and mass must be non-negative")
        self.volume = volume.copy()
        self.mass = mass.copy()

    def begin_trial(self, node_velocity: np.ndarray, grad_weights: np.ndarray, dt: float) -> np.ndarray:
        """Create a Newton trial state without changing committed history."""
        G = center_velocity_gradient(node_velocity, grad_weights)
        self.trial_F = center_trial_F(self.committed_F, G, dt)
        self.trial_tau = np.stack([
            kirchhoff(self.trial_F[c], self.A0[c], self.params)
            for c in range(self.center_count)
        ])
        self._trial_active = True
        return self.trial_F.copy()

    def residual(
        self,
        node_velocity: np.ndarray,
        node_velocity_old: np.ndarray,
        node_mass: np.ndarray,
        gravity: np.ndarray,
        dt: float,
        grad_weights: np.ndarray,
    ) -> np.ndarray:
        """Evaluate the center residual using the current committed state."""
        return center_residual(
            node_velocity,
            node_velocity_old,
            node_mass,
            gravity,
            dt,
            self.committed_F,
            self.A0,
            self.volume,
            grad_weights,
            self.params,
        )

    def matvec(self, direction: np.ndarray, node_mass: np.ndarray, dt: float, grad_weights: np.ndarray) -> np.ndarray:
        """Apply the Jacobian at the frozen Newton trial state."""
        return center_matvec(
            direction,
            node_mass,
            dt,
            self.committed_F,
            self.A0,
            self.volume,
            grad_weights,
            self.params,
            trial_F=self.trial_F,
        )

    def commit(self, converged: bool) -> None:
        """Commit exactly once on a converged step; otherwise roll back."""
        if not self._trial_active:
            raise RuntimeError("begin_trial must be called before commit")
        if converged:
            self.committed_F = self.trial_F.copy()
            self.valid[:] = True
        else:
            self.trial_F = self.committed_F.copy()
            self.trial_tau.fill(0.0)
        self._trial_active = False

    def solve_newton(
        self,
        node_velocity: np.ndarray,
        node_velocity_old: np.ndarray,
        node_mass: np.ndarray,
        gravity: np.ndarray,
        dt: float,
        grad_weights: np.ndarray,
        max_newton: int = 8,
        max_cg: int = 200,
        velocity_tol: float = 1.0e-8,
    ) -> tuple[np.ndarray, list[dict]]:
        """Solve a small center-only implicit step with frozen-state PCG."""
        v = np.asarray(node_velocity, dtype=np.float64).copy()
        self._trial_active = False
        logs: list[dict] = []
        for newton in range(max_newton):
            self.begin_trial(v, grad_weights, dt)
            residual = self.residual(v, node_velocity_old, node_mass, gravity, dt, grad_weights)
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm <= velocity_tol:
                self.commit(True)
                logs.append({"newton": newton, "residual_norm": residual_norm, "converged": True})
                return v, logs
            rhs = -residual
            inv_mass = 1.0 / np.maximum(np.asarray(node_mass, dtype=np.float64), 1.0e-12)
            dv, cg = matrix_free_pcg(
                lambda p: self.matvec(p, node_mass, dt, grad_weights),
                rhs,
                preconditioner=lambda r: inv_mass[:, None] * r,
                maxiter=max_cg,
                tol=velocity_tol,
            )
            v += dv
            logs.append({"newton": newton, "residual_norm": residual_norm, "cg": cg, "converged": False})
            if not cg.get("converged", False):
                self.commit(False)
                return v, logs
        self.commit(False)
        return v, logs
