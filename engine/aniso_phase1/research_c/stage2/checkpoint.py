"""Portable committed-state checkpoints bound to source, input, and algorithm."""
import json
import os
from pathlib import Path
import numpy as np
from ...research_d.common_state import CommonState
from ...research_d.identity import digest, sha


def source_manifest(repository):
    root = Path(repository)
    return {str(p.relative_to(root)): sha(p) for folder in
        ('engine/aniso_phase1/research_c/stage2', 'benchmarks/research_c/stage2',
         'tests/research_c/stage2') for p in sorted((root/folder).glob('*.py'))}


def binding(solver, dt, sources, parent):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('positive checkpoint dt required')
    return dict(parent_bundle_sha256=parent, extension_source_sha256=sources,
        model=solver.model.identity, dt=float(dt), path_order=solver.path_order,
        residual_atol=solver.atol, residual_rtol=solver.rtol, ledger_atol=solver.ledger_atol)


def save_checkpoint(path, solver, dt, sources, parent):
    state = solver.state
    data = dict(schema_version=1, binding=binding(solver, dt, sources, parent),
                state=state.to_dict())
    data['content_sha256'] = digest(data)
    path = Path(path)
    pending = path.with_name(path.name+'.pending')
    with pending.open('x') as stream:
        json.dump(data, stream, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, path)
    return data['content_sha256']


def load_checkpoint(path, model, expected_binding):
    data = json.loads(Path(path).read_text())
    actual = data.pop('content_sha256')
    if digest(data) != actual or data['schema_version'] != 1:
        raise ValueError('checkpoint bytes or schema changed')
    if data['binding'] != expected_binding or data['binding']['model'] != model.identity:
        raise ValueError('checkpoint source/input/dt/rule identity mismatch')
    d = data['state']
    state = CommonState(np.asarray(d['q']), np.asarray(d['velocity']), d['time'], d['step'],
        None if d['predictor'] is None else np.asarray(d['predictor']),
        d['transfer'], d['child_states'])
    model.validate(state, material=True)
    return state
