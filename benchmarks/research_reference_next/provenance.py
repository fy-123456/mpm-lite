"""Two-level ancestry, immutable registrations and data-disk-only output."""
from pathlib import Path
from datetime import datetime,timezone
from benchmarks.research_post_release.provenance import audit_parent as audit_math,PARENT,snapshot
from benchmarks.research_sequential_next.provenance import ROOT,read,write,sha,digest,check,utc,resources,environment,serial_lock
APP=ROOT/'docs/results/post-release/20261001T013007Z-post-release'
APP_SHA='5c23b687fc8565f9764690c84f53456694c21588a963776ac735e6a5e6de2d18'
PLAN=ROOT/'docs/MPM_LITE_NEXT_REFERENCE_FIRST_OPTIMIZATION_PLAN_20261001_ZH.md'


def all_sources():
    return {str(p.relative_to(ROOT)):sha(p) for folder in ('engine/aniso_phase1/research_reference_next','benchmarks/research_reference_next','tests/research_reference_next') for p in sorted((ROOT/folder).glob('*.py'))}


def source_files():
    return {n:h for n,h in all_sources().items() if n.startswith('engine/') or Path(n).name in ('__init__.py','config.py','provenance.py','run.py') and n.startswith('benchmarks/')}


def audit_parent(full=False):
    _,base,counts=audit_math(full)
    if sha(APP/'release.json')!=APP_SHA:raise ValueError('application release changed')
    pub=read(APP/'release.json')
    for key in ('sources','artifacts','qualification','scene'):
        x=pub[key]
        if sha(APP/x['path'])!=x['sha256']:raise ValueError('application index changed')
    sources=read(APP/pub['sources']['path']);counts['application_sources']=check(ROOT,sources)
    if full:
        counts['application_snapshot']=check(APP/'final-source',sources)
        counts['application_artifacts']=check(APP,read(APP/pub['artifacts']['path']))
    inventory={str(p.relative_to(ROOT)) for d in ('engine','benchmarks','tests','demos','utils') for p in (ROOT/d).rglob('*.py')}
    known=set(base['old_source_sha256'])|set(read(PARENT/'final-source-sha256.json'))|set(sources)|set(all_sources())
    if inventory-known:raise ValueError('unaccounted Python sources: '+str(sorted(inventory-known)))
    counts['python_inventory']=len(inventory)
    return pub,base,counts


def freeze():
    pub,base,counts=audit_parent(True)
    if resources()['system_free_GiB']<5:raise RuntimeError('migrate inactive data before starting below 5 GiB')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-reference-next'
    physical=Path('/root/autodl-tmp/mpm-lite-reference-next')/stamp;physical.mkdir(parents=True,exist_ok=False)
    run=ROOT/'docs/results/reference-next'/stamp;run.parent.mkdir(parents=True,exist_ok=True);run.symlink_to(physical,target_is_directory=True)
    lock=dict(schema='reference-next-input-v1',utc=utc(),application_parent=str(APP),application_release_sha256=APP_SHA,
       parent_release_sha256=pub['parent_release_sha256'],mathematical_parent=str(PARENT),counts=counts,
       plan_path=str(PLAN.relative_to(ROOT)),plan_sha256=sha(PLAN),energy_scale_J=base['energy_scale_J'],initial_sources=all_sources())
    write(run/'input-lock.json',lock);write(run/'input-lock-sha256.json',{'sha256':sha(run/'input-lock.json')})
    (run/'plan-frozen.md').write_bytes(PLAN.read_bytes());write(run/'environment.json',environment())
    write(run/'Q0/baseline-audit.json',dict(status='passed',**lock))
    write(run/'Q0/resources.json',dict(**resources(),migration='not_triggered',output=str(physical)))
    p=run/'Q4/inherited-qualification.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((APP/'S4/qualification.json').read_bytes())
    Path('/tmp/mpm-reference-run.txt').write_text(str(run)+'\n');return run


def verify(run):
    run=Path(run)
    if sha(run/'input-lock.json')!=read(run/'input-lock-sha256.json')['sha256']:raise ValueError('input lock changed')
    lock=read(run/'input-lock.json')
    if lock['application_release_sha256']!=APP_SHA or lock['application_parent']!=str(APP):raise ValueError('wrong application ancestor')
    audit_parent();return lock


def register(run,name,payload):
    run=Path(run);p=run/name
    if (run/'release.json').exists() or p.exists():raise ValueError('sealed run or existing registration')
    sources=all_sources();snapshot(p.parent/(p.stem+'-source'),sources)
    write(p,dict(payload,utc=utc(),application_release_sha256=APP_SHA,source_sha256=sources))


if __name__=='__main__':
    with serial_lock():print(freeze(),flush=True)
