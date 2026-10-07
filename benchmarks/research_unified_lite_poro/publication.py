"""Explicit descendant audit; all ancestor bytes remain authenticated.

The old inventory-aware audit intentionally cannot accept new modules. This
validator registers an exact new-file set and still verifies old sources,
snapshots, input packages and all sealed ancestor artifacts without patching it.
"""
import argparse,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
from benchmarks.research_post_release.provenance import audit_parent as audit_math
from benchmarks.research_coupled_reference_next.provenance import ANCESTORS

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/results/coupled-reference/20261005T105825Z-coupled-reference'
BASE_SHA='8428612da43ed943a47e4e3417d2439ba13d615a5fa29c15f29a3d53aaa43413'
FILES={
 'engine/aniso_phase1/research_unified_lite_poro/'+name+'.py'
 for name in ('__init__','space','model','solve','checkpoint')}
FILES|={'benchmarks/research_unified_lite_poro/'+name+'.py'
 for name in ('__init__','run','replay','analyze','visualize','publication')}
FILES|={'tests/research_unified_lite_poro/'+name+'.py' for name in ('__init__','test_contracts')}

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False))
def check(root,entries):
    for name,digest in entries.items():
        if sha(Path(root)/name)!=digest:raise ValueError('hash mismatch: '+str(Path(root)/name))
    return len(entries)

def sources():return {name:sha(ROOT/name) for name in sorted(FILES)}

def ancestors(run,full=True):
    lock=read(Path(run)/'input-lock.json')
    if lock['parent']!=str(BASE) or lock['parent_sha256']!=BASE_SHA:raise ValueError('wrong application parent')
    check(ROOT,lock['old_source_sha256'])
    current={str(p.relative_to(ROOT)) for folder in ('engine','benchmarks','tests','demos','utils') for p in (ROOT/folder).rglob('*.py')}
    if current!=set(lock['old_source_sha256'])|FILES:raise ValueError('unregistered source inventory difference')
    audit_math(full);counts={}
    for root,digest in [*ANCESTORS,(BASE,BASE_SHA)]:
        if sha(root/'release.json')!=digest:raise ValueError('changed ancestor release')
        pub=read(root/'release.json')
        for entry in pub.values():
            if isinstance(entry,dict) and 'path' in entry and 'sha256' in entry:
                if sha(root/entry['path'])!=entry['sha256']:raise ValueError('changed ancestor index')
        src=read(root/pub['sources']['path']);check(ROOT,src)
        if full:
            check(root/'final-source',src);check(root,read(root/pub['artifacts']['path']))
        docs=read(root/'documentation.json')
        for key in ('progress','coupling','current_plan'):
            if key+'_path' in docs and sha(ROOT/docs[key+'_path'])!=docs[key+'_sha256']:
                raise ValueError('changed ancestor documentation')
        counts[root.name]=len(src)
    if sha(Path(run)/'plan-frozen.md')!=lock['plan_sha256']:raise ValueError('frozen plan changed')
    return dict(ancestors=len(counts),old_sources=len(lock['old_source_sha256']),new_sources=len(FILES),full=full)

def seal(run):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('sealed outputs are read only')
    counts=ancestors(run,True);src=sources()
    tests=read(run/'S7/test-report.json')
    if tests['failures'] or tests['errors'] or not tests['successful']:raise ValueError('tests not passed')
    if tests['engine_sources']!={n:v for n,v in src.items() if n.startswith('engine/')}:raise ValueError('tested engine changed')
    for n,h in src.items():
        p=run/'final-source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    write(run/'final-source-sha256.json',src)
    progress=ROOT/'docs/MPM_LITE_UNIFIED_LITE_PORO_PROGRESS_20261005_ZH.md'
    write(run/'documentation.json',dict(progress_path=str(progress.relative_to(ROOT)),progress_sha256=sha(progress)))
    (run/'implementation-report.md').write_bytes(progress.read_bytes())
    artifacts={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and p.name not in ('release.json','artifact-sha256.json')}
    write(run/'artifact-sha256.json',artifacts)
    entry=lambda n:dict(path=n,sha256=sha(run/n))
    parent=read(BASE/'release.json')
    release=dict(schema='unified-lite-poro-scoped-v1',utc=datetime.now(timezone.utc).isoformat(),
      application_parent=str(BASE),application_parent_release_sha256=BASE_SHA,
      sources=dict(**entry('final-source-sha256.json'),count=len(src)),artifacts=dict(**entry('artifact-sha256.json'),count=len(artifacts)),
      input_lock=entry('input-lock.json'),capabilities=entry('capability-matrix.json'),step_audit=entry('requirement-audit.json'),
      default_case=parent['default_case'],default_case_source=parent['default_case_source'],selected_geometry='BD',
      research_backend='CPU material-coordinate particle bridge',research_rule='fixed-positive',
      formal_space_changed=False,spatial_accuracy=False,temporal_accuracy=False,
      eulerian_lite_integration=False,paper_complete=False,particle_decoupling_scope='per-call frozen CPU operator, structured reference-particle template',
      scope='engineering research delivery; no production replacement',audit=counts)
    write(run/'release.json',release);return audit(run,True)

def audit(run,full=True):
    run=Path(run);pub=read(run/'release.json');counts=ancestors(run,full)
    if pub['application_parent_release_sha256']!=BASE_SHA:raise ValueError('foreign parent')
    for e in pub.values():
        if isinstance(e,dict) and 'path' in e and 'sha256' in e:
            if sha(run/e['path'])!=e['sha256']:raise ValueError('child index changed')
    src=read(run/'final-source-sha256.json')
    if src!=sources():raise ValueError('current child source differs')
    check(run/'final-source',src)
    if full:counts['artifacts']=check(run,read(run/'artifact-sha256.json'))
    docs=read(run/'documentation.json')
    if sha(ROOT/docs['progress_path'])!=docs['progress_sha256']:raise ValueError('child document changed')
    return counts

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--seal',action='store_true');a=p.parse_args()
    result=seal(a.run) if a.seal else audit(a.run)
    print(json.dumps(dict(run=str(a.run),release_sha256=sha(a.run/'release.json'),audit=result),indent=2))
