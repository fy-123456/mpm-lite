"""Content-bound inputs, serial execution and resource records for N01--N12."""
from __future__ import annotations
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PREVIOUS = ROOT/'docs/results/sequential-v22/20260930T101746Z-practical'
COMMON = ROOT/'docs/results/parallel-v22/integration/20260930T054100Z-common-inputs'
PINS = {
 'docs/results/sequential-v22/20260930T101746Z-practical/final-source-sha256.json': 'c933e2e3d4bdb184920fa6eae4757d64fcd6b258d882be8825ed9d2281c8dcbf',
 'docs/results/sequential-v22/20260930T101746Z-practical/artifact-sha256.json': '9fa1b386330d6d22f8ff7111eec90483ac0bcf07e87e083f44a33e9c13d9425f',
 'docs/results/sequential-v22/20260930T101746Z-practical/protocol-v2.json': '677d9f73eefcf6ebd3aa0200c03f898bd4880ef5714ac98f655e8def31ad595d',
 'docs/results/parallel-v22/integration/20260930T054100Z-common-inputs/bundle.json': '55682a7b8e90b62c1306818cdf9c1f060b4286a174da069ce2217aa5b53b3c7c',
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write(path,value):
    """Publish a JSON file atomically and durably within its filesystem."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path);sync_dir(path.parent)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def check(base,entries):
    failures=[]
    for name,expected in entries.items():
        p=Path(base)/name
        if not p.is_file() or sha(p)!=expected:failures.append(name)
    if failures:raise ValueError('sealed content mismatch: '+', '.join(failures[:10]))
    return len(entries)


def source_files():
    folders=['engine/aniso_phase1/research_sequential_next','benchmarks/research_sequential_next']
    # Output-only rendering does not affect numerical continuation. All other
    # source changes require a new case or an explicit reviewed migration.
    return {str(p.relative_to(ROOT)):sha(p) for folder in folders
            for p in sorted((ROOT/folder).glob('*.py')) if p.name!='visualize.py'}


def register_study(output,relative,value):
    """Freeze a study protocol and the exact numerical source before its run."""
    output=Path(output);path=output/relative
    if path.exists():raise ValueError('study already registered: '+str(path))
    sources=source_files();value=dict(value,source_sha256=sources)
    snapshot=path.parent/(path.stem+'-source')
    for name,expected in sources.items():
        p=ROOT/name
        if sha(p)!=expected:raise ValueError('source changed during study freeze')
        q=snapshot/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    write(path,value)
    return value


def resources():
    limit=Path('/sys/fs/cgroup/memory.max')
    return dict(system_free_GiB=shutil.disk_usage(ROOT).free/2**30,
        data_free_GiB=shutil.disk_usage('/root/autodl-tmp').free/2**30,
        cgroup_memory_max=limit.read_text().strip() if limit.exists() else None)


def environment():
    versions={}
    for name in ('numpy','scipy','warp-lang','matplotlib'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version',
                        '--format=csv,noheader'],capture_output=True,text=True,check=False)
    return dict(utc=utc(),python=sys.version,executable=sys.executable,platform=platform.platform(),
        versions=versions,gpu=gpu.stdout.strip(),gpu_query_returncode=gpu.returncode,
        threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},**resources())


@contextlib.contextmanager
def serial_lock(output=None):
    # Same lock as the previous sequential runner; covers distinct output dirs.
    lock=ROOT/'docs/results/sequential-v22/.execution.lock'
    with lock.open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('another sequential research job holds the global lock') from exc
        try:
            if output is not None:
                Path(output).mkdir(parents=True,exist_ok=True)
            yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def freeze(output):
    output=Path(output)
    if (output/'baseline-lock.json').exists():raise ValueError('baseline already frozen; use a new run')
    check(ROOT,PINS)
    s01=read(PREVIOUS/'S01.json');old=s01['old_sources'];new=read(PREVIOUS/'final-source-sha256.json')
    bundle=read(COMMON/'bundle.json')
    counts=dict(old_sources=check(ROOT,old),new_sources=check(ROOT,new),
        input_identities=check(ROOT,s01['input_identities']),
        final_source_copy=check(PREVIOUS/'final-source',new),
        previous_artifacts=check(PREVIOUS,read(PREVIOUS/'artifact-sha256.json')),
        common_sources=check(ROOT,bundle['code_sha256']),common_files=check(COMMON,bundle['files']))
    all_sources={**old,**new}
    main=[str(p.relative_to(ROOT)) for folder in ('engine','benchmarks','tests','demos','utils')
          for p in (ROOT/folder).rglob('*.py') if 'research_sequential_next' not in p.parts]
    extra=sorted(set(main)-set(all_sources))
    if extra:raise ValueError('unaccounted newer implementation: '+str(extra))
    identities={**PINS,**s01['input_identities']}
    for name in ('common-model-package.json','coordinates.npz','protocol.json','protocol-revision.json','S01.json'):
        p=PREVIOUS/name;identities[str(p.relative_to(ROOT))]=sha(p)
    previous_ledger=read(PREVIOUS/'gpu-q7/ledger.json')
    scale=max(max(abs(row['total_J']) for row in previous_ledger),
              sum(abs(row['boundary_work_J'])+abs(row['external_work_J']) for row in previous_ledger),1e-12)
    value=dict(schema='sequential-next-baseline-v1',utc=utc(),
        version='v22 + two A-E rounds + common freeze + sequential-practical-v2',git_commit=None,
        counts=counts,old_source_sha256=all_sources,input_sha256=identities,
        previous_result=str(PREVIOUS),previous_physical_path=str(PREVIOUS.resolve()),
        common_physical_path=str(COMMON.resolve()),new_source_at_freeze=source_files(),
        energy_scale_J=scale,energy_scale_source='max(previous gpu-q7 peak |total_J|, sum absolute boundary/external work)',
        plan_sha256=sha(ROOT/'docs/MPM_LITE_NEXT_OPTIMIZATION_PLAN_20260930_ZH.md'))
    write(output/'baseline-lock.json',value)
    write(output/'baseline-lock-sha256.json',{'sha256':sha(output/'baseline-lock.json')})
    write(output/'environment.json',environment())
    write(output/'capabilities.json',{f'N{i:02}':dict(status='passed_scoped' if i==1 else 'not_run') for i in range(1,13)})
    write(output/'N01/result.json',dict(status='passed_scoped',counts=counts,energy_scale_J=scale,
        unchanged_historical_implementation=True,no_unaccounted_python=True))
    return value


def verify(output):
    output=Path(output)
    expected=read(output/'baseline-lock-sha256.json')['sha256']
    if sha(output/'baseline-lock.json')!=expected:raise ValueError('changed baseline lock')
    value=read(output/'baseline-lock.json')
    check(ROOT,PINS);check(ROOT,value['old_source_sha256']);check(ROOT,value['input_sha256'])
    return value


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    with serial_lock(args.output):
        result=freeze(args.output)
        print(json.dumps(dict(counts=result['counts'],energy_scale_J=result['energy_scale_J']),indent=2),flush=True)
