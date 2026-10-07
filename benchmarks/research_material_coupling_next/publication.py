"""Immutable descendant publication; old namespaces are only read and hashed."""
import argparse,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
from .common import write

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/results/formal-pressure-next/20261007T145423Z-formal-pressure-next'
BASE_SHA='2243316cce47c8d871cda69e153b4fc0c3e62877a28b619cd8e44d08c59be8ef'
FILES={str(p.relative_to(ROOT)) for folder in ('engine/aniso_phase1','benchmarks','tests') for p in (ROOT/folder/'research_material_coupling_next').glob('*.py')}
PROGRESS=ROOT/'docs/MPM_LITE_MATERIAL_RULE_AND_MECHANICAL_COUPLING_PROGRESS_20261007_ZH.md'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def sources():return {n:sha(ROOT/n) for n in sorted(FILES)}

def ancestry(run):
    lock=read(run/'input-lock.json')
    if lock['parent']!=str(BASE) or lock['parent_sha256']!=BASE_SHA or sha(BASE/'release.json')!=BASE_SHA:raise ValueError('wrong parent')
    if sha(run/'plan-frozen.md')!=lock['plan_sha256']:raise ValueError('changed plan')
    current={str(p.relative_to(ROOT)) for folder in ('engine','benchmarks','tests','demos','utils') for p in (ROOT/folder).rglob('*.py')}
    if current!=set(lock['old_source_sha256'])|FILES:raise ValueError('unregistered source inventory')
    verified={};count=0
    def check(p,h):
        key=str(p.resolve())
        if key not in verified:verified[key]=sha(p)
        if verified[key]!=h:raise ValueError('changed immutable file: '+str(p))
    for n,h in lock['old_source_sha256'].items():check(ROOT/n,h)
    for path,digest in read(run/'S0/ancestor-releases.json').items():
        root=ROOT/path;check(root/'release.json',digest);pub=read(root/'release.json')
        for item in pub.values():
            if isinstance(item,dict) and 'path' in item and 'sha256' in item:check(root/item['path'],item['sha256'])
        key='sources'
        if pub.get('schema')=='sequential-next-practical-release-v1':
            key='numerical_sources';check(root/'baseline-lock.json',pub['baseline_sha256'])
            historical=read(root/'baseline-lock.json')
            for kind in ('old_source_sha256','input_sha256'):
                for n,h in historical[kind].items():check(ROOT/n,h)
        for n,h in read(root/pub[key]['path']).items():check(ROOT/n,h);check(root/'final-source'/n,h)
        for n,h in read(root/pub['artifacts']['path']).items():check(root/n,h)
        if (root/'documentation.json').exists():
            docs=read(root/'documentation.json')
            for key in ('progress','coupling','current_plan'):
                if key+'_path' in docs:check(ROOT/docs[key+'_path'],docs[key+'_sha256'])
        count+=1
    return dict(ancestor_releases=count,frozen_sources=len(lock['old_source_sha256']),new_sources=len(FILES),unique_files_verified=len(verified))

def audit(run):
    pub=read(run/'release.json');counts=ancestry(run)
    for item in pub.values():
        if isinstance(item,dict) and 'path' in item and 'sha256' in item:
            if sha(run/item['path'])!=item['sha256']:raise ValueError('changed index')
    src=read(run/'final-source-sha256.json')
    if src!=sources():raise ValueError('changed new source')
    for n,h in src.items():
        if sha(run/'final-source'/n)!=h:raise ValueError('changed snapshot')
    artifacts=read(run/'artifact-sha256.json')
    for n,h in artifacts.items():
        if sha(run/n)!=h:raise ValueError('changed artifact: '+n)
    if sha(PROGRESS)!=read(run/'documentation.json')['progress_sha256']:raise ValueError('changed progress')
    return dict(**counts,artifacts=len(artifacts),release_sha256=sha(run/'release.json'))

def seal(run):
    if (run/'release.json').exists():raise ValueError('sealed release is read only')
    counts=ancestry(run);src=sources();tests=read(run/'S7/test-report.json')
    if not tests['successful'] or tests['engine_sources']!={n:h for n,h in src.items() if n.startswith('engine/')}:raise ValueError('untested engine sources')
    for n in FILES:
        p=run/'final-source'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,p)
    write(run/'final-source-sha256.json',src);shutil.copyfile(PROGRESS,run/'implementation-report.md')
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS)))
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ('release.json','artifact-sha256.json')}
    write(run/'artifact-sha256.json',artifacts)
    entry=lambda name:dict(path=name,sha256=sha(run/name))
    parent=read(BASE/'release.json')
    pub=dict(schema='material-coupling-next-scoped-v1',utc=datetime.now(timezone.utc).isoformat(),application_parent=str(BASE),application_parent_release_sha256=BASE_SHA,
        sources=dict(**entry('final-source-sha256.json'),count=len(src)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),
        input_lock=entry('input-lock.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),
        default_case=parent['default_case'],default_case_source=parent['default_case_source'],selected_geometry='BD',
        research_backend='CPU original formal material coordinates; budgeted q5/q7 midpoint and constant SPD hydraulic network',
        formal_space_changed=False,original_kernel_transfer_tested=False,eulerian_lite_integration=False,spatial_accuracy=False,temporal_accuracy=False,paper_complete=False,
        scope='qualification scope in capability-matrix and requirement-audit; no production or GPU replacement',audit=counts)
    write(run/'release.json',pub)
    return dict(**counts,artifacts=len(artifacts),release_sha256=sha(run/'release.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--seal',action='store_true');a=p.parse_args()
    print(json.dumps(seal(a.run) if a.seal else audit(a.run),indent=2))
