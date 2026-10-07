"""Production sparse-grid FD, symmetry, positive-direction and boundary scan."""
import argparse
import gc
import json
from pathlib import Path
import warp as wp
from engine.aniso_phase1 import select_lowest_memory_device
from engine.aniso_phase1.operator_probe import SparseProbe
from demos.aniso import DATA_ROOT
from utils.resource_guard import prepare_warp_cache


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='auto')
    p.add_argument('--out', type=Path, default=Path('output/sparse-operator.json'))
    args=p.parse_args()
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache', DATA_ROOT)
    wp.init()
    device=select_lowest_memory_device(args.device)
    rows=[]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for kf in (0., 200., 20000.):
        for dt in (.0005, .005, .05):
            wp.config.kernel_cache_dir=prepare_warp_cache(wp.config.kernel_cache_dir, DATA_ROOT)
            probe=SparseProbe(device, kf, dt)
            row=probe.check()
            row['device']=device
            rows.append(row)
            args.out.write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
            del probe
            gc.collect()


if __name__=='__main__': main()
