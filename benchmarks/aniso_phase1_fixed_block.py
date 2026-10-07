"""Fixed-boundary fiber-enhanced block stability probe."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import warp as wp

from engine.aniso_phase1 import AnisotropicLiteImplicitSolver, AnisotropicMaterialParams, energy, query_gpu_memory, select_lowest_memory_device
from utils.resource_guard import StoragePaused, prepare_warp_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cache", default="/tmp/mpm-lite-warp-cache")
    parser.add_argument("--data-root", default=os.environ.get("MPM_LITE_DATA_ROOT"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        args.cache = prepare_warp_cache(args.cache, args.data_root)
    except StoragePaused as exc:
        raise SystemExit(f"EXPERIMENT PAUSED: {exc}") from exc
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    wp.config.kernel_cache_dir = args.cache
    wp.init()
    device = select_lowest_memory_device(args.device)
    gpu_before = query_gpu_memory().get(device, {})

    positions = np.array(
        [[0.20 + 0.12 * i, 0.20 + 0.12 * j, 0.20 + 0.12 * k]
         for i in range(4) for j in range(4) for k in range(4)],
        dtype=np.float64,
    )
    gradient = np.diag([0.08, 0.0, 0.0])
    boundary = np.array([[1, j, k] for j in range(1, 7) for k in range(1, 7)], dtype=np.int32)
    params = AnisotropicMaterialParams(10.0, 20.0, 200.0, [1.0, 0.0, 0.0])
    solver = AnisotropicLiteImplicitSolver(
        (8, 8, 8),
        params=params,
        dx=1.0 / 7.0,
        device=device,
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
    start = time.perf_counter()
    converged_steps = 0
    min_det = float("inf")
    max_speed = 0.0
    max_F_error = 0.0
    energies = []
    for _ in range(args.steps):
        if solver.step(max_iters=8, print_every=0, v_tol=1.0e-7):
            converged_steps += 1
        valid = solver.aniso_state_valid.numpy()[0] > 0
        F = solver.aniso_committed_F.numpy()[0][valid]
        min_det = min(min_det, float(np.min(np.linalg.det(F))))
        max_speed = max(max_speed, solver.max_particle_speed())
        max_F_error = max(max_F_error, solver.center_particle_F_error()["max_frobenius"])
        valid = solver.aniso_state_valid.numpy()[0] > 0
        F_valid = solver.aniso_committed_F.numpy()[0][valid]
        A_valid = solver.aniso_A0.numpy()[0][valid]
        V_valid = solver.center_vol.numpy()[0][valid]
        energies.append(float(sum(energy(F, A, params) * V for F, A, V in zip(F_valid, A_valid, V_valid))))
    gpu_after = query_gpu_memory().get(device, {})
    print(json.dumps({
        "steps": args.steps,
        "device": device,
        "gpu_used_before_mib": gpu_before.get("used_mib"),
        "gpu_used_after_mib": gpu_after.get("used_mib"),
        "gpu_total_mib": gpu_after.get("total_mib", gpu_before.get("total_mib")),
        "converged_steps": converged_steps,
        "seconds": time.perf_counter() - start,
        "active_centers": solver.last_step_stats.get("active_centers", 0),
        "active_nodes": solver.last_step_stats.get("active_nodes", 0),
        "min_det_F": min_det,
        "max_particle_speed": max_speed,
        "max_center_particle_F_error": max_F_error,
        "max_center_A0_mixing": solver.center_structure_tensor_mixing()["max"],
        "mass_error": float(abs(solver.center_m.numpy().sum() - solver.ptc_m.numpy().sum())),
        "volume_error": float(abs(solver.center_vol.numpy().sum() - solver.ptc_vol0.numpy().sum())),
        "final_internal_energy": energies[-1] if energies else 0.0,
        "max_energy_jump": max((abs(curr - prev) for prev, curr in zip(energies, energies[1:])), default=0.0),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
