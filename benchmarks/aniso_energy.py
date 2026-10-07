"""Equal-physical-time sweeps of the instrumented fixed and affine scenes."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import warp as wp
from demos.aniso import Config, Scene, DATA_ROOT
from engine.aniso_phase1 import select_lowest_memory_device, query_gpu_memory
from utils.resource_guard import prepare_warp_cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='auto')
    p.add_argument('--time', type=float, default=.02)
    p.add_argument('--out', type=Path, default=Path('output/energy'))
    p.add_argument('--data-root', default=DATA_ROOT)
    args = p.parse_args()
    wp.config.kernel_cache_dir = prepare_warp_cache('/tmp/mpm-lite-warp-cache', args.data_root)
    wp.init()
    args.out.mkdir(parents=True, exist_ok=True)
    configs = [(dt, .9, 1e-4, 1e-10) for dt in (.0005, .001, .002)]
    configs += [(.001, flip, 1e-4, 1e-10) for flip in (0., .5, 1.)]
    configs += [(.001, .9, cg, vt) for cg, vt in ((.5, 1e-5), (.01, 1e-8), (1e-6, 1e-12))]
    results = []
    device = select_lowest_memory_device(args.device)
    for kind in ('fixed', 'affine'):
        for dt, flip, cg, vt in configs:
            steps = round(args.time/dt)
            if steps < 1 or not np.isclose(steps*dt, args.time, atol=1e-12, rtol=0):
                raise ValueError('--time must be a positive multiple of every dt')
            before = query_gpu_memory().get(device, {})
            scene = Scene(Config(kind, 8, dt, 0., 200., flip, cg, vt, residual_atol=vt), device)
            name = f'{kind}-dt{dt}-flip{flip}-cg{cg}-vt{vt}'
            for _ in range(steps):
                wp.config.kernel_cache_dir = prepare_warp_cache(wp.config.kernel_cache_dir, args.data_root)
                if not scene.step():
                    raise RuntimeError(f'{name}: failed at {scene.solver.sim_time}')
            ledger = scene.solver.energy_ledger
            (args.out/f'{name}.csv').write_text(ledger.csv())
            last = ledger.rows[-1]
            sums = {k: sum(r.get(k, 0.) for r in ledger.rows[1:]) for k in last if k.endswith('_delta') and k != 'cumulative_delta'}
            results.append(dict(scene=kind, device=device, dt=dt, flip=flip, cg_tol=cg, v_tol=vt, newton_atol=vt, linear_solver=scene.solver.last_step_stats["linear_solver"],
                                final_time=last['time'], initial_energy=ledger.rows[0]['mechanical'],
                                final_energy=last['mechanical'], relative_change=last['cumulative_delta']/ledger.rows[0]['mechanical'],
                                max_budget_closure=max(abs(r.get('budget_closure', 0)) for r in ledger.rows),
                                reactivated_centers=sum(r.get('reactivated_centers', 0) for r in ledger.rows),
                                gpu_before=before, **sums))
            (args.out/'summary.json').write_text(json.dumps(results, indent=2))
            print(json.dumps(results[-1]), flush=True)
            del scene, ledger
            gc.collect()


if __name__ == '__main__':
    main()
