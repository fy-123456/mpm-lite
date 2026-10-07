"""Finite-difference check for the center matrix-free residual Jacobian."""

from __future__ import annotations

import json

import numpy as np

from engine.aniso_phase1 import AnisotropicMaterialParams, center_matvec, center_residual, center_trial_F, center_velocity_gradient


def main() -> None:
    params = AnisotropicMaterialParams(3.0, 5.0, 17.0, [1.0, 2.0, 0.5])
    committed_F = np.stack([
        np.diag([1.1, 0.95, 1.03]),
        np.array([[1.0, 0.1, 0.0], [0.0, 1.0, 0.05], [0.0, 0.0, 1.0]]),
    ])
    A0 = np.stack([params.A0, params.A0])
    mass = np.array([2.0, 1.5, 1.0])
    volume = np.array([0.7, 0.4])
    velocity_old = np.zeros((3, 3))
    gravity = np.array([0.0, 0.0, -9.81])
    dt = 2.0e-3
    grad_weights = np.array([
        [[-0.4, -0.2, -0.1], [0.4, 0.2, 0.1], [0.0, 0.0, 0.0]],
        [[-0.3, 0.1, -0.2], [0.3, -0.1, 0.2], [0.0, 0.0, 0.0]],
    ])
    velocity = np.array([[0.3, -0.2, 0.15], [-0.2, 0.1, 0.05], [0.1, 0.0, -0.2]])
    direction = np.array([[0.3, 0.2, -0.1], [-0.2, 0.1, 0.25], [0.05, -0.1, 0.2]])
    trial_F = center_trial_F(committed_F, center_velocity_gradient(velocity, grad_weights), dt)
    applied = center_matvec(direction, mass, dt, committed_F, A0, volume, grad_weights, params, trial_F=trial_F)
    errors = []
    for eps in (1.0e-4, 1.0e-5, 1.0e-6):
        plus = center_residual(
            velocity + eps * direction, velocity_old, mass, gravity, dt,
            committed_F, A0, volume, grad_weights, params,
        )
        minus = center_residual(
            velocity - eps * direction, velocity_old, mass, gravity, dt,
            committed_F, A0, volume, grad_weights, params,
        )
        numerical = (plus - minus) / (2.0 * eps)
        errors.append({
            "eps": eps,
            "relative_l2": float(np.linalg.norm(applied - numerical) / max(np.linalg.norm(numerical), 1.0e-30)),
            "absolute_l2": float(np.linalg.norm(applied - numerical)),
        })
    repeated = center_matvec(direction, mass, dt, committed_F, A0, volume, grad_weights, params, trial_F=trial_F)
    rng = np.random.default_rng(23)
    quadratic_forms = []
    for _ in range(16):
        probe = rng.normal(size=direction.shape)
        applied_probe = center_matvec(probe, mass, dt, committed_F, A0, volume, grad_weights, params, trial_F=trial_F)
        quadratic_forms.append(float(np.sum(probe * applied_probe)))
    print(json.dumps({
        "relative_errors": errors,
        "repeat_max_abs": float(np.max(np.abs(applied - repeated))),
        "spd_probe_min_quadratic_form": min(quadratic_forms),
        "spd_probe_negative_count": sum(value <= 0.0 for value in quadratic_forms),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
