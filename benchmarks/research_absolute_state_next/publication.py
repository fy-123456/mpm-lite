"""Explicit descendant inventory; verify old bytes without weakening old audits."""
import argparse,hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
from .run import write

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/results/constrained-history-next/20261007T131646Z-constrained-history-next'
BASE_SHA='525a7d77d420347fbc182acd27fee5daba164a823db982e1485b6fc70b368448'
FILES={f'engine/aniso_phase1/research_absolute_state_next/{n}.py' for n in ('__init__','state','dynamics')}
FILES|={f'benchmarks/research_absolute_state_next/{n}.py' for n in ('__init__','run','nonlinear','dry_cycle','replay','visualize','publication')}
FILES|={f'tests/research_absolute_state_next/{n}.py' for n in ('__init__','test_contracts')}
PROGRESS=ROOT/'docs/MPM_LITE_ABSOLUTE_STATE_PROGRESS_20261007_ZH.md'

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def sources():return {name:sha(ROOT/name) for name in sorted(FILES)}

def ancestry(run):
    lock=read(run/'input-lock.json')
    if lock['parent']!=str(BASE) or lock['parent_sha256']!=BASE_SHA or sha(BASE/'release.json')!=BASE_SHA:
        raise ValueError('wrong or changed parent')
    if sha(run/'plan-frozen.md')!=lock['plan_sha256']:raise ValueError('plan changed')
    current={str(p.relative_to(ROOT)) for folder in ('engine','benchmarks','tests','demos','utils') for p in (ROOT/folder).rglob('*.py')}
    if current!=set(lock['old_source_sha256'])|FILES:raise ValueError('unregistered source inventory')
    verified={};count=0
    def check(p,digest):
        key=str(p.resolve())
        if key not in verified:verified[key]=sha(p)
        if verified[key]!=digest:raise ValueError('hash mismatch: '+str(p))
    for name,digest in lock['old_source_sha256'].items():check(ROOT/name,digest)
    for path,digest in read(run/'S0/ancestor-releases.json').items():
        root=ROOT/path;check(root/'release.json',digest);pub=read(root/'release.json')
        for item in pub.values():
            if isinstance(item,dict) and 'path' in item and 'sha256' in item:check(root/item['path'],item['sha256'])
        source_key='sources'
        if pub.get('schema')=='sequential-next-practical-release-v1':
            source_key='numerical_sources'
            check(root/'baseline-lock.json',pub['baseline_sha256'])
            historical=read(root/'baseline-lock.json')
            for key in ('old_source_sha256','input_sha256'):
                for name,digest in historical[key].items():check(ROOT/name,digest)
        for name,digest in read(root/pub[source_key]['path']).items():
            check(ROOT/name,digest);check(root/'final-source'/name,digest)
        for name,digest in read(root/pub['artifacts']['path']).items():check(root/name,digest)
        if (root/'documentation.json').exists():
            docs=read(root/'documentation.json')
            for key in ('progress','coupling','current_plan'):
                if key+'_path' in docs:check(ROOT/docs[key+'_path'],docs[key+'_sha256'])
        count+=1
    return dict(ancestor_releases=count,frozen_sources=len(lock['old_source_sha256']),
                new_sources=len(FILES),unique_files_verified=len(verified))

def audit(run):
    pub=read(run/'release.json');counts=ancestry(run)
    for item in pub.values():
        if isinstance(item,dict) and 'path' in item and 'sha256' in item:
            if sha(run/item['path'])!=item['sha256']:raise ValueError('changed index')
    src=read(run/'final-source-sha256.json')
    if src!=sources():raise ValueError('changed new source')
    for name,digest in src.items():
        if sha(run/'final-source'/name)!=digest:raise ValueError('changed source snapshot')
    artifacts=read(run/'artifact-sha256.json')
    for name,digest in artifacts.items():
        if sha(run/name)!=digest:raise ValueError('changed artifact: '+name)
    doc=read(run/'documentation.json')
    if sha(PROGRESS)!=doc['progress_sha256']:raise ValueError('changed progress document')
    return dict(**counts,artifacts=len(artifacts),release_sha256=sha(run/'release.json'))

def seal(run):
    if (run/'release.json').exists():raise ValueError('sealed release is read only')
    counts=ancestry(run);src=sources();tests=read(run/'S7/test-report.json')
    if not tests['successful'] or tests['engine_sources']!={n:v for n,v in src.items() if n.startswith('engine/')}:
        raise ValueError('untested source or failed tests')
    for name in FILES:
        p=run/'final-source'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
    write(run/'final-source-sha256.json',src)
    shutil.copyfile(PROGRESS,run/'implementation-report.md')
    write(run/'documentation.json',dict(progress_path=str(PROGRESS.relative_to(ROOT)),progress_sha256=sha(PROGRESS)))
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ('release.json','artifact-sha256.json')}
    write(run/'artifact-sha256.json',artifacts)
    entry=lambda name:dict(path=name,sha256=sha(run/name))
    parent=read(BASE/'release.json')
    pub=dict(schema='absolute-state-next-scoped-v1',utc=datetime.now(timezone.utc).isoformat(),
        application_parent=str(BASE),application_parent_release_sha256=BASE_SHA,
        sources=dict(**entry('final-source-sha256.json'),count=len(src)),
        artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),
        input_lock=entry('input-lock.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),
        default_case=parent['default_case'],default_case_source=parent['default_case_source'],selected_geometry='BD',
        research_backend='CPU original formal material coordinates with corrected SH query contract; 4-step nonlinear dry midpoint',research_rule='full q7 formal material, original constant full mass; no pressure coupling in new branch',
        original_kernel_transfer_tested=False,common_transfer_candidate=True,bounded_coupled_package=False,eulerian_lite_integration=False,formal_space_changed=False,
        gate_S_original_material_route=True,gate_M_original_material_small_window=True,formal_updated_SH_qualified=False,formal_pressure_coupling=False,
        inherited_small_model_bounded_retry=True,
        spatial_accuracy=False,temporal_accuracy=False,paper_complete=False,
        particle_decoupling_scope='fixed-size original material q/v state and dry operator; corrected SH point queries remain point-dependent; prior small-model coupled evidence only inherited',
        scope='original material absolute-state closure plus corrected SH CPU contract and four very small nonlinear dry midpoint steps; no updated C1 adoption, no formal pressure coupling or production replacement',audit=counts)
    write(run/'release.json',pub)
    return dict(**counts,artifacts=len(artifacts),release_sha256=sha(run/'release.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--seal',action='store_true');a=p.parse_args()
    print(json.dumps(seal(a.run) if a.seal else audit(a.run),indent=2))
