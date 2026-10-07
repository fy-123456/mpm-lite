"""Explicit archived-v20 configuration, with no substituted C components."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT/'docs/results/lite-aniso-mainline/v20/fast-cycle'


def legacy_v20():
    protocol = json.loads((ARCHIVE/'cycle-protocol.json').read_text())
    for name, expected in protocol['source_sha256'].items():
        actual = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'archived legacy source changed: {name}')
    # Exact archived execution chain; not another AVF variant and not alpha=0.
    from benchmarks.aniso_v20_fast_runs import setup
    from ..fast_integrated_avf import FastIntegratedAVF
    s, e, mass, h, meta = setup('gauss3-condensed')
    solver = FastIntegratedAVF(s, e, mass, h, moving=True, mode='driven',
                               condense=True, clamped=True, chord=True)
    return solver, protocol, meta
