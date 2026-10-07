"""Restore an isolated D checkout without modifying any parent member."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

PARENT = Path('docs/results/parallel-v22/integration/20260930T054100Z-common-inputs')
PIN = '55682a7b8e90b62c1306818cdf9c1f060b4286a174da069ce2217aa5b53b3c7c'

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def restore(source, target, *, dereference=True):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target == source or target.is_relative_to(source):
        raise ValueError('independent checkout must be outside source')
    parent = source / PARENT
    if sha(parent/'bundle.json') != PIN:
        raise ValueError('wrong parent')
    bundle = json.loads((parent/'bundle.json').read_text())
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(parent/'code-snapshot.zip') as z:
        for name in z.namelist():
            if not (target/name).resolve().is_relative_to(target):
                raise ValueError('unsafe snapshot member')
        z.extractall(target)
    dest = target/PARENT
    dest.mkdir(parents=True, exist_ok=True)
    for name in list(bundle['files']) + ['bundle.json', 'bundle-sha256.txt', 'code-snapshot.zip']:
        out = dest/name
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            shutil.copy2(parent/name, out, follow_symlinks=dereference)
        if sha(out) != sha(parent/name):
            raise ValueError('restore mismatch: '+name)
    baseline = Path('docs/results/lite-aniso-mainline/v22/source-delivered-sha256.json')
    (target/baseline).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source/baseline, target/baseline)
    sync(source, target)
    return dict(parent_bundle_sha256=PIN, source=str(source), target=str(target),
                dereferenced=dereference, sources=len(bundle['code_sha256']),
                files=len(bundle['files']))

def sync(source, target):
    for area in ('engine/aniso_phase1', 'benchmarks', 'tests'):
        rel = Path(area)/'research_d/stage2'
        for path in (Path(source)/rel).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                out = Path(target)/path.relative_to(source)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, out)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('target');p.add_argument('--sync',action='store_true');a=p.parse_args()
    if a.sync: sync(Path.cwd(), Path(a.target))
    else: print(json.dumps(restore(Path.cwd(), a.target),indent=2))
