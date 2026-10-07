"""Three pinned parent releases, explicit new sources and immutable attempts."""
from pathlib import Path
from datetime import datetime,timezone
from benchmarks.research_post_release.provenance import audit_parent as audit_math,PARENT,snapshot
from benchmarks.research_reference_next.provenance import APP as PREVIOUS_APP,APP_SHA as PREVIOUS_SHA
from benchmarks.research_sequential_next.provenance import ROOT,read,write,sha,digest,check,utc,resources,environment,serial_lock
APP=ROOT/'docs/results/reference-next/20261001T032104Z-reference-next'
APP_SHA='fd730ea63ceff7c63570d3600a2932272b645dfeb650930ac7fd882c0aced4b0'
PLAN=ROOT/'docs/MPM_LITE_NONLINEAR_SPACE_PHASE_COUPLING_PLAN_20261001_ZH.md'
# Declared names only: an unregistered new Python file is still rejected.
BENCH=['__init__','provenance','base_config','config','spaces','run','visualize','spatial_study','time_study','rule_study','coupling_study','performance_study','finalize','publication']
ENGINE=['__init__','space','multicell','performance']
TESTS=['__init__','test_runtime']


def all_sources():
    pairs=[('benchmarks/research_spatial_phase_next',BENCH),('engine/aniso_phase1/research_spatial_phase_next',ENGINE),('tests/research_spatial_phase_next',TESTS)]
    return {str(p.relative_to(ROOT)):sha(p) for folder,names in pairs for name in names if (p:=ROOT/folder/(name+'.py')).is_file()}


def source_files():
    return {n:h for n,h in all_sources().items() if n.startswith('engine/') or n.startswith('benchmarks/') and Path(n).stem in ['__init__','provenance','base_config','config','spaces','run']}


def audit_parent(full=False):
    math,base,counts=audit_math(full);known=set(base['old_source_sha256'])|set(read(PARENT/'final-source-sha256.json'))
    for label,folder,expected in [('previous_application',PREVIOUS_APP,PREVIOUS_SHA),('application',APP,APP_SHA)]:
        if sha(folder/'release.json')!=expected:raise ValueError('parent release changed: '+label)
        pub=read(folder/'release.json')
        for key in ('sources','artifacts','qualification','scene')+ (('sensitive_qualification',) if label=='application' else ()):
            e=pub[key]
            if sha(folder/e['path'])!=e['sha256']:raise ValueError('parent index changed: '+key)
        sources=read(folder/pub['sources']['path']);known|=set(sources);counts[label+'_sources']=check(ROOT,sources)
        if full:
            counts[label+'_snapshots']=check(folder/'final-source',sources);counts[label+'_artifacts']=check(folder,read(folder/pub['artifacts']['path']))
            docs=read(folder/'documentation.json')
            for pre in ('progress','coupling','current_plan'):
                if sha(ROOT/docs[pre+'_path'])!=docs[pre+'_sha256']:raise ValueError('parent documentation changed')
    inventory={str(p.relative_to(ROOT)) for d in ('engine','benchmarks','tests','demos','utils') for p in (ROOT/d).rglob('*.py')}
    if inventory-known-set(all_sources()):raise ValueError('unaccounted source: '+str(sorted(inventory-known-set(all_sources()))))
    counts['python_inventory']=len(inventory)
    return read(APP/'release.json'),base,counts


def freeze():
    pub,base,counts=audit_parent(True)
    if resources()['system_free_GiB']<5:raise RuntimeError('migrate inactive data before starting below 5 GiB')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-spatial-phase'
    data=Path('/root/autodl-tmp/mpm-lite-spatial-phase')/stamp;data.mkdir(parents=True,exist_ok=False)
    run=ROOT/'docs/results/spatial-phase'/stamp;run.parent.mkdir(parents=True,exist_ok=True);run.symlink_to(data,target_is_directory=True)
    lock=dict(schema='spatial-phase-input-v1',utc=utc(),application_parent=str(APP),application_release_sha256=APP_SHA,
        parent_release_sha256=pub['mathematical_release_sha256'],mathematical_parent=str(PARENT),previous_application=str(PREVIOUS_APP),previous_release_sha256=PREVIOUS_SHA,
        plan_sha256=sha(PLAN),plan_path=str(PLAN.relative_to(ROOT)),energy_scale_J=base['energy_scale_J'],source_sha256=all_sources(),counts=counts)
    write(run/'input-lock.json',lock);write(run/'input-lock-sha256.json',dict(sha256=sha(run/'input-lock.json')))
    (run/'plan-frozen.md').write_bytes(PLAN.read_bytes());write(run/'environment.json',environment());write(run/'N0/version-audit.json',dict(status='passed',**lock))
    write(run/'N0/resources.json',dict(**resources(),migration='not_triggered',output=str(data)))
    for target,source in [('qualification.json','qualification.json'),('qualification-peak0075.json','qualification-peak0075.json')]:
        p=run/'N3'/target;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((APP/'Q4'/source).read_bytes())
    write(run/'selected-space.json',dict(selected='original144',package=None,mass_order=5,full_order=7,status='baseline_pending_N1'))
    Path('/tmp/mpm-spatial-phase-run.txt').write_text(str(run)+'\n');return run


def verify(run):
    run=Path(run)
    if sha(run/'input-lock.json')!=read(run/'input-lock-sha256.json')['sha256']:raise ValueError('input lock changed')
    lock=read(run/'input-lock.json')
    if lock['application_parent']!=str(APP) or lock['application_release_sha256']!=APP_SHA:raise ValueError('wrong direct application')
    audit_parent();return lock


def register(run,name,payload):
    run=Path(run);p=run/name
    if (run/'release.json').exists() or p.exists():raise ValueError('sealed run or existing registration')
    sources=all_sources();snapshot(p.parent/(p.stem+'-source'),sources)
    write(p,dict(payload,utc=utc(),application_release_sha256=APP_SHA,source_sha256=sources))


if __name__=='__main__':
    with serial_lock():print(freeze(),flush=True)
