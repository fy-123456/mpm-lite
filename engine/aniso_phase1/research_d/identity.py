"""Content-addressed, append-only experiment identities."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASELINE = '136529c866133aac10af77126c839f12a2769073ee5ec964f46dcadeca7adee4'


def digest(value):
    h = hashlib.sha256()
    def add(v):
        if isinstance(v, np.ndarray):
            a = np.ascontiguousarray(v)
            if a.dtype.hasobject:
                raise TypeError('object arrays cannot identify a numerical problem')
            h.update(json.dumps([a.dtype.str, a.shape]).encode()); h.update(a.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v):
                add(k); add(v[k])
        elif isinstance(v, (tuple, list)):
            h.update(str(len(v)).encode())
            for x in v: add(x)
        else:
            h.update(json.dumps(v, sort_keys=True, allow_nan=False).encode())
        h.update(b'\0')
    add(value)
    return h.hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    def scalar(v):
        if isinstance(v,np.generic): return v.item()
        raise TypeError(f'unsupported JSON type: {type(v).__name__}')
    payload=json.dumps(value,indent=2,allow_nan=False,default=scalar)+'\n'
    path=Path(path);pending=path.with_name(path.name+'.pending')
    with pending.open('x') as f: f.write(payload)
    try:
        os.link(pending,path)  # atomic publication, refuses to replace existing evidence
    finally:
        pending.unlink()


def baseline_audit():
    base = ROOT/'docs/results/lite-aniso-mainline/v22'
    records = {}
    for name in ('source-delivered-sha256.json', 'artifact-sha256.json'):
        entries = json.loads((base/name).read_text())
        bad = [rel for rel, expected in entries.items()
               if not (ROOT/rel).is_file() or sha(ROOT/rel) != expected]
        records[name] = dict(entries=len(entries), mismatches=bad, manifest_sha256=sha(base/name))
    records['archive_sha256'] = sha(base/'source-delivered.zip')
    records['passed'] = (records['archive_sha256'] == BASELINE and
                         all(not r['mismatches'] for r in records.values() if isinstance(r, dict)))
    manifest = json.loads((base/'source-delivered-sha256.json').read_text())
    records['additional_python'] = sorted(str(p.relative_to(ROOT)) for folder in
        ('engine','benchmarks','tests','demos','utils') for p in (ROOT/folder).rglob('*.py')
        if str(p.relative_to(ROOT)) not in manifest)
    records['newer_version_directories'] = sorted(str(p.relative_to(ROOT)) for p in
        (ROOT/'docs/results/lite-aniso-mainline').glob('v[0-9]*')
        if p.name[1:].isdigit() and int(p.name[1:]) > 22)
    records['unarchived_shared_python'] = [p for p in records['additional_python']
        if not any(part.startswith('research_') for part in Path(p).parts)]
    records['passed'] &= not records['newer_version_directories'] and not records['unarchived_shared_python']
    return records


def own_sources():
    paths = [ROOT/'engine/aniso_phase1/research_contracts.py']
    for folder in ('engine/aniso_phase1/research_d','benchmarks/research_d','tests/research_d'):
        paths.extend((ROOT/folder).rglob('*.py'))
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths) if p.is_file()}
