"""One explicitly bounded F45 reference extension after the v10 base study."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from benchmarks.aniso_boundary_reference import DEFAULT as SOURCE, hashes as prior_hashes, hessian, write
from engine.aniso_phase1 import boundary_reference as ref

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'docs/results/lite-aniso-mainline/v10-reference-fine'


def hashes():
    return {**prior_hashes(), 'benchmarks/aniso_reference_extend.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('action', choices=('freeze', 'run', 'analyze'))
    p.add_argument('--output', type=Path, default=DEFAULT); a = p.parse_args(); out = a.output
    out.mkdir(parents=True, exist_ok=True)
    if a.action == 'freeze':
        if (out/'protocol.json').exists(): raise RuntimeError('preserve protocol')
        s = json.loads((SOURCE/'summary.json').read_text()); assert s['completed']
        refs = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                for f in SOURCE.rglob('*') if f.is_file()}
        write(out/'protocol.json', dict(frozen_at=datetime.now(timezone.utc).isoformat(), source_sha256=hashes(),
            reference_sha256=refs, grids=[129, 257], boundaries=['hard', 'smooth'], case='F45',
            reason='F45 interior stress is unresolved at grid 129 under both boundary models',
            maximum_extra_levels=1, maximum_grid=257, unchanged_gates=dict(reaction=.01, interior=.02, global_field=.02)))
        return
    protocol = json.loads((out/'protocol.json').read_text()); assert protocol['source_sha256']==hashes()
    for name, digest in protocol['reference_sha256'].items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    if a.action == 'run':
        dest = out/'cases'; dest.mkdir(exist_ok=False); records = []
        for boundary in protocol['boundaries']:
            start = time.monotonic(); nodes, u, r = ref.solve(257, hessian('F45'), boundary)
            r.update(case='F45', seconds=time.monotonic()-start)
            name = f'F45-{boundary}-g257'; write(dest/(name+'.json'), r)
            np.savez_compressed(dest/(name+'.npz'), nodes=nodes, u=u)
            records.append(r); print(name, r, flush=True)
            if not r['passed']: raise RuntimeError('reference solve failed')
        write(out/'runs.json', dict(completed=True, records=records)); return
    records = []
    for boundary in protocol['boundaries']:
        values = []
        for grid, root in ((129, SOURCE), (257, out)):
            name = f'F45-{boundary}-g{grid}'
            with np.load(root/'cases'/(name+'.npz')) as z: u=z['u'].copy()
            values.append((grid, u, json.loads((root/'cases'/(name+'.json')).read_text())))
        r = ref.compare(*values, hessian('F45')); r.update(case='F45', boundary=boundary)
        records.append(r); print(boundary, r, flush=True)
    write(out/'summary.json', dict(completed=True, pairs=records,
        full_reference_certified=all(r['reaction_passed'] and r['global_passed'] for r in records),
        note='hard and smooth are separate physical boundary problems, not interchangeable references'))


if __name__ == '__main__': main()
