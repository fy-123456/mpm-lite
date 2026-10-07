"""Small CPU PPC scaling probe for the isolated phase-one solver."""

import argparse
import os
import time

import numpy as np
import warp as wp

from engine.aniso_phase1 import AnisotropicLiteImplicitSolver, AnisotropicMaterialParams, query_gpu_memory, select_lowest_memory_device
from utils.resource_guard import StoragePaused, prepare_warp_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache", default="/tmp/mpm-lite-warp-cache")
    parser.add_argument("--data-root", default=os.environ.get("MPM_LITE_DATA_ROOT"))
    args = parser.parse_args()
    try:
        args.cache = prepare_warp_cache(args.cache, args.data_root)
    except StoragePaused as exc:
        raise SystemExit(f"EXPERIMENT PAUSED: {exc}") from exc
    device = select_lowest_memory_device(args.device)
    gpu_before = query_gpu_memory().get(device, {})
    wp.config.kernel_cache_dir = args.cache
    wp.init()
    rng = np.random.default_rng(7)
    params = AnisotropicMaterialParams(500.0, 1000.0, 3000.0, [1.0, 0.0, 0.0])
    for n_particles in (8, 32, 128, 512):
        # Keep every particle inside one grid cell.  The eight center stencil
        # is fixed, so these counts realize Np/Nc = 1, 4, 16, 64 while the
        # active center/node footprint stays fixed.
        positions = np.array([[0.46, 0.46, 0.46]]) + rng.random((n_particles, 3)) * 0.02
        solver = AnisotropicLiteImplicitSolver(
            (12, 12, 12), params=params, dx=1.0 / 11.0,
            device=device, gravity=0.0, ppc=1,
        )
        solver.seed_particles(positions, density=1000.0, vol0=0.02**3/n_particles,
                              velocity=(positions-.47)*.03, velocity_gradient=np.eye(3)*.03)
        solver.paint_boundary(np.array([[0, 0, 0]], dtype=np.int32), np.array([1], dtype=np.int32))
        solver.set_dt(1.0e-4)
        solver.step(max_iters=8, print_every=0, v_tol=1.0e-8)
        start = time.perf_counter()
        solver.step(max_iters=8, print_every=0, v_tol=1.0e-8)
        wp.synchronize_device(device)
        elapsed = time.perf_counter() - start
        stats = solver.last_step_stats
        print(
            f"device={device} particles={n_particles} mass={solver.ptc_m.numpy().sum():.9g} volume={solver.ptc_vol0.numpy().sum():.9g} time={solver.sim_time:.9g} "
            f"gpu_used_mib={gpu_before.get('used_mib', 'na')} "
            f"gpu_total_mib={gpu_before.get('total_mib', 'na')} "
            f"active_centers={int(solver.n_active_centers.numpy()[0])} "
            f"active_nodes={int(solver.n_active_nodes.numpy()[0])} "
            f"np_nc_ratio={n_particles / max(1, int(solver.n_active_centers.numpy()[0])):.6g} "
            f"step_seconds={elapsed:.6f} "
            f"p2c_seconds={stats['p2c_seconds']:.6f} "
            f"cg_seconds={stats['cg_seconds']:.6f} "
            f"cg_iterations={stats['cg_iterations']} "
            f"material_seconds={stats['material_seconds']:.6f} "
            f"memory_mib={stats['memory_bytes'] / (1024.0 * 1024.0):.3f}"
        )


if __name__ == "__main__":
    main()
