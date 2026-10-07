"""Source-complete child of the sealed pressure-startup release."""
from pathlib import Path
from datetime import datetime, timezone
from benchmarks.research_sequential_next.provenance import ROOT,read,write,sha,digest,check,utc,resources,environment,serial_lock
from benchmarks.research_post_release.provenance import audit_parent as audit_math,PARENT,snapshot
from benchmarks.research_startup_substeps_next.provenance import ANCESTORS as PREVIOUS,REFERENCE,LOCAL,PHASE_REFERENCE

APP=ROOT/'docs/results/startup-substeps/20261005T035524Z-startup-substeps'
ZERO=ROOT/'docs/results/zero-source-theta/20261004T182027Z-zero-source-theta'
PREV=ROOT/'docs/results/restoring-rt0/20261004T151306Z-restoring-rt0'
OBS=ROOT/'docs/results/candidate-observable/20261004T133432Z-candidate-observable'
SOLID=ROOT/'docs/results/pressure-window/20261004T114530Z-pressure-window'
OLD=ROOT/'docs/results/observable-boundary/20261001T195842Z-observable-boundary'
APP_SHA='3a0420805b250a81d05c757e7a6c639a621cff6734da04b41acba96efe13d1e1'
ANCESTORS=[*PREVIOUS,(APP,APP_SHA)]
PLAN=ROOT/'docs/MPM_LITE_NEXT_CONTINUOUS_GEOMETRY_AND_SPATIAL_REFERENCE_PLAN_20261005_ZH.md'
PROGRESS=ROOT/'docs/MPM_LITE_CONTINUOUS_GEOMETRY_PROGRESS_20261005_ZH.md'

def all_sources():
    folders=['benchmarks/research_continuous_geometry_next','engine/aniso_phase1/research_continuous_geometry_next','tests/research_continuous_geometry_next']
    return {str(p.relative_to(ROOT)):sha(p) for folder in folders for p in sorted((ROOT/folder).glob('*.py'))}

def source_files():
    core={'__init__','provenance','base_config','config','spaces','run','physics'}
    return {k:v for k,v in all_sources().items() if k.startswith('benchmarks/') and Path(k).stem in core}

def audit_parent(full=False):
    _,base,counts=audit_math(full);known=set(base['old_source_sha256'])|set(read(PARENT/'final-source-sha256.json'))
    for folder,expected in ANCESTORS:
        if sha(folder/'release.json')!=expected:raise ValueError('ancestor release changed '+str(folder))
        pub=read(folder/'release.json')
        for entry in pub.values():
            if isinstance(entry,dict) and 'path' in entry and 'sha256' in entry:
                if sha(folder/entry['path'])!=entry['sha256']:raise ValueError('ancestor index changed')
        sources=read(folder/pub['sources']['path']);known|=set(sources);record=dict(sources=check(ROOT,sources))
        if full:record.update(snapshots=check(folder/'final-source',sources),artifacts=check(folder,read(folder/pub['artifacts']['path'])))
        docs=read(folder/'documentation.json')
        for key in ('progress','coupling','current_plan'):
            if key+'_path' in docs and sha(ROOT/docs[key+'_path'])!=docs[key+'_sha256']:raise ValueError('ancestor document changed')
        counts[folder.name]=record
    inventory={str(p.relative_to(ROOT)) for d in ('engine','benchmarks','tests','demos','utils') for p in (ROOT/d).rglob('*.py')}
    unknown=inventory-known-set(all_sources())
    if unknown:raise ValueError('unaccounted source '+str(sorted(unknown)))
    counts['python_inventory']=len(inventory)
    return read(APP/'release.json'),base,counts

def freeze():
    pub,base,counts=audit_parent(True)
    if resources()['system_free_GiB']<5:raise RuntimeError('migrate inactive data before starting below 5 GiB')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-continuous-geometry'
    data=Path('/root/autodl-tmp/mpm-lite-continuous-geometry')/stamp;data.mkdir(parents=True,exist_ok=False)
    run=ROOT/'docs/results/continuous-geometry'/stamp;run.parent.mkdir(parents=True,exist_ok=True);run.symlink_to(data,target_is_directory=True)
    lock=dict(schema='continuous-geometry-input-v1',utc=utc(),application_parent=str(APP),application_release_sha256=APP_SHA,parent_release_sha256=sha(PARENT/'release.json'),mathematical_parent=str(PARENT),plan_path=str(PLAN.relative_to(ROOT)),plan_sha256=sha(PLAN),energy_scale_J=base['energy_scale_J'],source_sha256=all_sources(),counts=counts)
    write(run/'input-lock.json',lock);write(run/'input-lock-sha256.json',dict(sha256=sha(run/'input-lock.json')))
    (run/'plan-frozen.md').write_bytes(PLAN.read_bytes());write(run/'environment.json',environment());write(run/'S0/version-audit.json',dict(status='passed_scoped',**lock))
    for name in ('selected-space.json','baseline-space.json'):(run/name).write_bytes((APP/'selected-space.json').read_bytes())
    write(run/'S0/experiment-budget.json',dict(serial=True,max_total_attempts=70,stages={'S0':0,'S1':18,'S2':22,'S3':0,'S4':30,'S5':0,'S6':0},max_new_display_frames=6,max_rss_GiB=16))
    write(run/'S0/resource-policy.json',dict(status='registered',gpu_guard='unchanged .7 of initial free; exact decision sampling',single_live_gpu_model=True,system_free_min_GiB=5,all_new_results_on_data_disk=True,**resources()))
    write(run/'S0/storage-migration.json',dict(status='not_triggered',**resources()))
    Path('/tmp/mpm-continuous-run.txt').write_text(str(run)+'\n');return run


def verify(run):
    run=Path(run)
    if sha(run/'input-lock.json')!=read(run/'input-lock-sha256.json')['sha256']:raise ValueError('input lock changed')
    lock=read(run/'input-lock.json')
    if lock['application_parent']!=str(APP) or lock['application_release_sha256']!=APP_SHA:raise ValueError('wrong direct parent')
    if sha(run/'plan-frozen.md')!=lock['plan_sha256']:raise ValueError('plan changed')
    audit_parent();return lock

def mutable(run):
    if (Path(run)/'release.json').exists():raise ValueError('sealed run is read only')
    return verify(run)

def register(run,name,payload):
    run=Path(run);p=run/name
    if (run/'release.json').exists() or p.exists():raise ValueError('sealed or existing registration')
    sources=all_sources();snapshot(p.parent/(p.stem+'-source'),sources)
    write(p,dict(payload,utc=utc(),application_release_sha256=APP_SHA,source_sha256=sources))
