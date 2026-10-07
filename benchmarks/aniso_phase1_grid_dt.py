"""Grid and time-step convergence probe for the isolated phase-one solver."""

from __future__ import annotations

import argparse
import gc
import os
import time

import numpy as np
import warp as wp

from engine.aniso_phase1 import AnisotropicLiteImplicitSolver, AnisotropicMaterialParams, query_gpu_memory, select_lowest_memory_device
from utils.resource_guard import StoragePaused, prepare_warp_cache


def run_case(
    grid_n: int,
    dt: float,
    steps: int,
    particle_count: int = 64,
    device: str = "cpu",
    cg_tol: float = 1.0e-4,
    cg_atol: float = 1.0e-12,
) -> dict[str, float | int | bool]:
    if particle_count < 0:
        raise ValueError("particle_count must be non-negative")
    if particle_count == 0:
        # A grid-filling lattice keeps the physical support comparable while
        # refining the grid: every interior cell center carries one particle.
        dx = 1.0 / (grid_n - 1)
        positions = np.array(
            [[(i + 0.5) * dx, (j + 0.5) * dx, (k + 0.5) * dx]
             for i in range(1, grid_n - 1)
             for j in range(1, grid_n - 1)
             for k in range(1, grid_n - 1)],
            dtype=np.float64,
        )
    else:
        side = int(np.ceil(particle_count ** (1.0 / 3.0)))
        extent = 0.6
        spacing = extent / max(1, side - 1)
        lattice = np.array(
            [[0.2 + spacing * i, 0.2 + spacing * j, 0.2 + spacing * k]
             for i in range(side) for j in range(side) for k in range(side)],
            dtype=np.float64,
        )
        positions = lattice[:particle_count]
    gradient = np.array(
        [[0.08, 0.02, 0.01], [0.00, -0.03, 0.015], [0.00, 0.00, 0.04]],
        dtype=np.float64,
    )
    expected_step = np.eye(3) + dt * gradient
    solver = AnisotropicLiteImplicitSolver(
        (grid_n, grid_n, grid_n),
        params=AnisotropicMaterialParams(10.0, 20.0, 30.0, [1.0, 0.0, 0.0]),
        dx=1.0 / (grid_n - 1),
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
    solver.set_dt(dt)
    start = time.perf_counter()
    gpu_before = query_gpu_memory().get(device, {})
    converged = True
    min_det = float("inf")
    max_speed = 0.0
    for _ in range(steps):
        converged = bool(
            solver.step(
                max_iters=8,
                print_every=0,
                v_tol=1.0e-10,
                cg_tol=cg_tol,
                cg_atol=cg_atol,
                max_cg_iters=max(100, grid_n * grid_n * grid_n * 3),
            )
        ) and converged
        valid = solver.aniso_state_valid.numpy()[0] > 0
        if np.any(valid):
            min_det = min(min_det, float(np.min(np.linalg.det(solver.aniso_committed_F.numpy()[0][valid]))))
        max_speed = max(max_speed, solver.max_particle_speed())
    elapsed = time.perf_counter() - start
    expected = np.linalg.matrix_power(expected_step, steps)
    valid = solver.aniso_state_valid.numpy()[0] > 0
    committed = solver.aniso_committed_F.numpy()[0][valid]
    errors = np.linalg.norm(committed - expected, axis=(1, 2)) if len(committed) else np.empty(0)
    error = float(np.max(errors)) if len(errors) else float("nan")
    mean_error = float(np.mean(errors)) if len(errors) else float("nan")
    occupied_volume = solver.center_vol.numpy()[0][valid]
    # Exclude the two-cell sparse-support boundary when comparing resolutions;
    # that boundary changes with the particle stencil and is reported by the
    # separate global maximum above.
    coords = []
    block_coords = solver.block_xyz_by_id.numpy()[:int(solver.block_count.numpy()[0])]
    for bid, block in enumerate(block_coords):
        local_indices = np.argwhere(valid[bid])
        for li, lk_flat in local_indices:
            coords.append(
                np.asarray(block, dtype=np.int64) * 32
                + np.array([li, lk_flat // 32, lk_flat % 32], dtype=np.int64)
            )
    coords = np.asarray(coords, dtype=np.int64)
    interior = np.all((coords >= 2) & (coords < (grid_n - 3)), axis=1) if len(coords) else np.zeros(0, dtype=bool)
    interior_error = float(np.max(errors[interior])) if np.any(interior) else float("nan")
    # A common physical sample near the domain center is comparable across
    # grid resolutions even though the sparse active-center count changes.
    center_sample_error = float("nan")
    if len(committed):
        center_target = np.array([0.5, 0.5, 0.5]) / solver.dx - 0.5
        center_index = np.rint(center_target).astype(np.int64)
        block = center_index // 32
        local = center_index % 32
        bid = int(solver.block2bid.numpy()[tuple(block)])
        if bid >= 0:
            sampled = solver.aniso_committed_F.numpy()[0, bid, local[0], local[1] * 32 + local[2]]
            center_sample_error = float(np.linalg.norm(sampled - expected))
    stats = solver.last_step_stats
    gpu_after = query_gpu_memory().get(device, {})
    result = {
        "device": device,
        "gpu_used_before_mib": gpu_before.get("used_mib", "na"),
        "gpu_used_after_mib": gpu_after.get("used_mib", "na"),
        "gpu_total_mib": gpu_after.get("total_mib", gpu_before.get("total_mib", "na")),
        "grid": grid_n,
        "particles": int(len(positions)),
        "dt": dt,
        "steps": steps,
        "converged": bool(converged),
        "active_centers": int(stats.get("active_centers", 0)),
        "active_nodes": int(stats.get("active_nodes", 0)),
        "step_seconds": elapsed,
        "p2c_seconds": float(stats.get("p2c_seconds", 0.0)),
        "cg_seconds": float(stats.get("cg_seconds", 0.0)),
        "material_seconds": float(stats.get("material_seconds", 0.0)),
        "transfer_seconds": float(stats.get("transfer_seconds", 0.0)),
        "matvec_calls": int(stats.get("matvec_calls", 0)),
        "cg_iterations": int(stats.get("cg_iterations", 0)),
        "newton_iterations": int(stats.get("newton_iterations", 0)),
        "last_residual_norm": float(stats.get("last_residual_norm", 0.0)),
        "affine_F_max_error": error,
        "affine_F_mean_error": mean_error,
        "affine_F_interior_max_error": interior_error,
        "affine_F_center_error": center_sample_error,
        "min_det_F": min_det,
        "max_particle_speed": max_speed,
        "memory_mib": float(stats.get("memory_bytes", 0)) / (1024.0 * 1024.0),
    }
    del solver
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/tmp/mpm-lite-warp-cache")
    parser.add_argument("--data-root", default=os.environ.get("MPM_LITE_DATA_ROOT"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--grid", type=int, action="append", help="grid size; repeat to scan several sizes")
    parser.add_argument("--dt", type=float, action="append", help="time step; repeat to scan several values")
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--cg-tol", type=float, default=1.0e-4)
    parser.add_argument("--cg-atol", type=float, default=1.0e-12)
    args = parser.parse_args()
    device = select_lowest_memory_device(args.device)
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.particles < 0:
        raise SystemExit("--particles must be non-negative (0 fills the grid)")
    try:
        args.cache = prepare_warp_cache(args.cache, args.data_root)
    except StoragePaused as exc:
        raise SystemExit(f"EXPERIMENT PAUSED: {exc}") from exc
    wp.config.kernel_cache_dir = args.cache
    wp.init()
    grids = tuple(args.grid) if args.grid else (8, 12, 16)
    dts = tuple(args.dt) if args.dt else (5.0e-4, 1.0e-3, 2.0e-3)
    if any(grid_n < 2 for grid_n in grids):
        raise SystemExit("--grid values must be at least 2")
    if any(dt <= 0.0 for dt in dts):
        raise SystemExit("--dt values must be positive")
    for grid_n in grids:
        for dt in dts:
            result = run_case(grid_n, dt, args.steps, args.particles, device, args.cg_tol, args.cg_atol)
            print(" ".join(f"{key}={value:.9g}" if isinstance(value, float) else f"{key}={value}" for key, value in result.items()))


if __name__ == "__main__":
    main()
