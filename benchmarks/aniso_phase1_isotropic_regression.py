"""Compare the isolated phase-one solver with the legacy isotropic path at k_f=0."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import warp as wp

from engine.aniso_phase1 import AnisotropicLiteImplicitSolver, AnisotropicMaterialParams
from engine.solver3d import MPMSolver
from engine.types import Material
from engine.aniso_phase1 import query_gpu_memory, select_lowest_memory_device
from utils.resource_guard import StoragePaused, prepare_warp_cache


def _make_solvers(device: str):
    positions = np.array(
        [[0.42, 0.42, 0.42], [0.46, 0.45, 0.44], [0.52, 0.48, 0.49], [0.58, 0.55, 0.54]],
        dtype=np.float64,
    )
    E, nu = 1500.0, 0.3
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    legacy = MPMSolver((8, 8, 8), dx=1.0 / 7.0, device=device, gravity=-9.81, ppc=1, solver_type="lite_implicit")
    legacy.add_material(0, Material.elastic, E=E, nu=nu)
    legacy.seed_particles(positions, 0, 1000.0, 0.001)
    aniso = AnisotropicLiteImplicitSolver(
        (8, 8, 8), AnisotropicMaterialParams(mu, lam, 0.0, [1.0, 0.0, 0.0]),
        dx=1.0 / 7.0, device=device, gravity=-9.81, ppc=1,
    )
    aniso.seed_particles(positions, density=1000.0, vol0=0.001)
    boundary_ijk = np.array([[0, 0, 0]], dtype=np.int32)
    boundary = np.array([1], dtype=np.int32)
    legacy.paint_boundary(boundary_ijk, boundary)
    aniso.paint_boundary(boundary_ijk, boundary)
    legacy.set_dt(1.0e-4)
    aniso.set_dt(1.0e-4)
    return legacy, aniso


def _max_center_error(legacy, aniso, array_name: str, mask: np.ndarray) -> float:
    left = getattr(legacy, array_name).numpy()
    right = getattr(aniso, array_name).numpy()
    if array_name in ("center_vol", "center_dlogJ"):
        return float(np.max(np.abs(left[0][mask] - right[0][mask])))
    if array_name == "center_v":
        return float(np.max(np.abs(left[mask] - right[mask])))
    flat_mask = mask.reshape(mask.shape[0], mask.shape[1], -1)
    return float(np.max(np.abs(left[0][flat_mask] - right[0][flat_mask])))


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
    legacy, aniso = _make_solvers(device)
    start = time.perf_counter()
    legacy.step(max_iters=1, print_every=0, v_tol=1.0e-5)
    legacy_seconds = time.perf_counter() - start
    start = time.perf_counter()
    aniso_success = aniso.step(max_iters=1, print_every=0, v_tol=1.0e-5)
    aniso_seconds = time.perf_counter() - start

    legacy_vol = legacy.center_vol.numpy()[0]
    aniso_vol = aniso.center_vol.numpy()[0]
    mask = ((legacy_vol > 0.0) & (aniso_vol > 0.0)).reshape(aniso.center_v.shape[:4])
    legacy_stats = getattr(legacy, "last_legacy_implicit_stats", {})
    aniso_stats = aniso.last_step_stats
    print(json.dumps({
        "device": device,
        "gpu_used_before_mib": gpu_before.get("used_mib"),
        "gpu_total_mib": gpu_before.get("total_mib"),
        "aniso_success": bool(aniso_success),
        "position_max_abs": float(np.max(np.abs(legacy.ptc_x.numpy() - aniso.ptc_x.numpy()))),
        "velocity_max_abs": float(np.max(np.abs(legacy.ptc_v.numpy() - aniso.ptc_v.numpy()))),
        "particle_F_max_abs": float(np.max(np.abs(legacy.ptc_F.numpy() - aniso.ptc_F.numpy()))),
        "center_velocity_max_abs": _max_center_error(legacy, aniso, "center_v", mask),
        "center_stress_max_abs": _max_center_error(legacy, aniso, "center_tau", mask),
        "legacy_residual_norm": legacy_stats.get("residual_norm"),
        "aniso_residual_norm": aniso_stats.get("last_residual_norm"),
        "legacy_newton_iterations": legacy_stats.get("newton_iteration"),
        "aniso_newton_iterations": aniso_stats.get("newton_iterations"),
        "legacy_cg_iterations": legacy_stats.get("cg_iterations"),
        "aniso_cg_iterations": aniso_stats.get("cg_iterations"),
        "legacy_seconds": legacy_seconds,
        "aniso_seconds": aniso_seconds,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
