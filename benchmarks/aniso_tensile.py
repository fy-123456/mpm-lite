"""Matched 0/45/90-degree dynamic loading/unloading tensile experiments."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import warp as wp
from demos.aniso import Config, Scene, DATA_ROOT
from engine.aniso_phase1 import select_lowest_memory_device
from utils.resource_guard import prepare_warp_cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='auto')
    p.add_argument('--grid', type=int, default=9)
    p.add_argument('--dt', type=float, default=.002)
    p.add_argument('--loading-time', type=float, default=.2)
    p.add_argument('--speed', type=float, default=.01)
    p.add_argument('--out', type=Path, default=Path('output/tensile'))
    args = p.parse_args()
    wp.config.kernel_cache_dir = prepare_warp_cache('/tmp/mpm-lite-warp-cache', DATA_ROOT)
    wp.init()
    device = select_lowest_memory_device(args.device)
    steps = round(2*args.loading_time/args.dt)
    if steps < 2 or not np.isclose(steps*args.dt, 2*args.loading_time, rtol=0, atol=1e-12):
        raise ValueError('duration must be an integer number of steps')
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for angle in (0., 45., 90.):
        scene = Scene(Config('tensile', args.grid, args.dt, angle, 200.,
                             loading_speed=args.speed, loading_time=args.loading_time), device)
        for _ in range(steps):
            wp.config.kernel_cache_dir = prepare_warp_cache(wp.config.kernel_cache_dir, DATA_ROOT)
            if not scene.step():
                raise RuntimeError(f'angle {angle} failed at t={scene.solver.sim_time}')
        (args.out/f'angle{angle:g}.csv').write_text(scene.solver.energy_ledger.csv())
        rows = scene.loading_rows
        peak = max(rows, key=lambda r:r['displacement'])
        loading = [r for r in rows if r['loading_velocity']>0 and r['displacement']>=.2*peak['displacement']]
        slope = float(np.polyfit([r['displacement'] for r in loading], [r['right_force'] for r in loading], 1)[0])
        results.append(dict(angle=angle, device=device, grid=args.grid, dt=args.dt,
                            physical_volume=.75*.25*.25, mass=float(scene.solver.ptc_m.numpy().sum()),
                            loading_time=args.loading_time, speed=args.speed, steps=steps,
                            peak_displacement=peak['displacement'], peak_force=peak['right_force'],
                            peak_elastic_force=peak['right_elastic_force'], secant_stiffness=peak['effective_stiffness'],
                            fitted_loading_stiffness=slope, loading_work=peak['loading_work'],
                            cycle_work=rows[-1]['loading_work'], final_displacement=rows[-1]['displacement'],
                            final_measured_displacement=rows[-1]['measured_grip_displacement'],
                            max_momentum_balance_error=max(abs(r['momentum_balance_error']) for r in rows),
                            max_free_residual=max(r['free_force_residual_norm'] for r in rows)))
        (args.out/'summary.json').write_text(json.dumps(results, indent=2))
        print(json.dumps(results[-1]), flush=True)
        del scene
        gc.collect()


if __name__ == '__main__':
    main()
