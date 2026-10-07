"""Host-side stage-one material and center-state types."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def _identity() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _default_A0() -> np.ndarray:
    return np.diag([1.0, 0.0, 0.0]).astype(np.float64)


@dataclass(frozen=True)
class AnisotropicMaterialParams:
    """Parameters for fixed-reference transverse isotropic elasticity.

    ``fiber_direction`` is expressed in reference coordinates.  It is
    normalized at construction and ``a`` and ``-a`` therefore represent the
    same structure tensor.
    """

    mu: float
    lam: float
    k_f: float
    fiber_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    def __post_init__(self) -> None:
        if not np.isfinite([self.mu, self.lam, self.k_f]).all():
            raise ValueError("material parameters must be finite")
        if self.mu < 0.0 or self.k_f < 0.0:
            raise ValueError("mu and k_f must be non-negative")
        direction = np.asarray(self.fiber_direction, dtype=np.float64)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError("fiber_direction must be a finite 3-vector")
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("fiber_direction must have non-zero length")
        object.__setattr__(self, "fiber_direction", direction / norm)

    @property
    def A0(self) -> np.ndarray:
        a = self.fiber_direction
        return np.outer(a, a)


@dataclass
class AnisotropicCenterState:
    """Committed and trial state for one center quadrature point.

    CG iterations only read ``trial_F``.  ``commit`` is the single operation
    that advances history, so a failed Newton step can safely call
    ``rollback`` without modifying the committed state.
    """

    committed_F: np.ndarray = field(default_factory=_identity)
    A0: np.ndarray = field(default_factory=_default_A0)
    volume: float = 0.0
    mass: float = 0.0
    valid: bool = False
    trial_F: np.ndarray = field(init=False)
    trial_tau: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.committed_F = _as_mat33(self.committed_F, "committed_F")
        self.A0 = _as_structure_tensor(self.A0)
        if self.volume < 0.0 or self.mass < 0.0:
            raise ValueError("center volume and mass must be non-negative")
        self.trial_F = self.committed_F.copy()
        self.trial_tau = np.zeros((3, 3), dtype=np.float64)

    def begin_trial(self, velocity_gradient: np.ndarray, dt: float) -> np.ndarray:
        G = _as_mat33(velocity_gradient, "velocity_gradient")
        if not np.isfinite(dt) or dt < 0.0:
            raise ValueError("dt must be finite and non-negative")
        self.trial_F = (np.eye(3) + dt * G) @ self.committed_F
        return self.trial_F.copy()

    def set_trial(self, F: np.ndarray) -> None:
        self.trial_F = _as_mat33(F, "trial_F")

    def commit(self) -> None:
        self.committed_F = self.trial_F.copy()
        self.valid = True

    def rollback(self) -> None:
        self.trial_F = self.committed_F.copy()
        self.trial_tau.fill(0.0)


def _as_mat33(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return arr.copy()


def _as_structure_tensor(value: np.ndarray) -> np.ndarray:
    arr = _as_mat33(value, "A0")
    arr = 0.5 * (arr + arr.T)
    eig = np.linalg.eigvalsh(arr)
    if eig.min() < -1.0e-10:
        raise ValueError("A0 must be positive semi-definite")
    return arr
