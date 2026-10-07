"""Source-complete child of the sealed candidate-observable release."""
from pathlib import Path
from datetime import datetime, timezone
from benchmarks.research_sequential_next.provenance import ROOT,read,write,sha,digest,check,utc,resources,environment,serial_lock
from benchmarks.research_post_release.provenance import audit_parent as audit_math,PARENT,snapshot
from benchmarks.research_candidate_observable_next.provenance import ANCESTORS as PREVIOUS,REFERENCE,LOCAL,PHASE_REFERENCE

APP=ROOT/'docs/results/candidate-observable/20261004T133432Z-candidate-observable'
SOLID=ROOT/'docs/results/pressure-window/20261004T114530Z-pressure-window'
OLD=ROOT/'docs/results/observable-boundary/20261001T195842Z-observable-boundary'
APP_SHA='5f5606d48fddb611a1c5c15843cac089f1b15c45e374ddfe43627b4cb6302526'
ANCESTORS=[*PREVIOUS,(APP,APP_SHA)]
PLAN=ROOT/'docs/MPM_LITE_NEXT_RESTORING_FORCE_AND_RT0_OPTIMIZATION_PLAN_20261004_ZH.md'
PROGRESS=ROOT/'docs/MPM_LITE_RESTORING_FORCE_RT0_PROGRESS_20261004_ZH.md'

def all_sources():
    folders=['benchmarks/research_restoring_rt0_next','engine/aniso_phase1/research_restoring_rt0_next','tests/research_restoring_rt0_next']
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
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-restoring-rt0'
    data=Path('/root/autodl-tmp/mpm-lite-restoring-rt0')/stamp;data.mkdir(parents=True,exist_ok=False)
    run=ROOT/'docs/results/restoring-rt0'/stamp;run.parent.mkdir(parents=True,exist_ok=True);run.symlink_to(data,target_is_directory=True)
    lock=dict(schema='restoring-rt0-input-v1',utc=utc(),application_parent=str(APP),application_release_sha256=APP_SHA,parent_release_sha256=sha(PARENT/'release.json'),mathematical_parent=str(PARENT),plan_path=str(PLAN.relative_to(ROOT)),plan_sha256=sha(PLAN),energy_scale_J=base['energy_scale_J'],source_sha256=all_sources(),counts=counts)
    write(run/'input-lock.json',lock);write(run/'input-lock-sha256.json',dict(sha256=sha(run/'input-lock.json')))
    (run/'plan-frozen.md').write_bytes(PLAN.read_bytes());write(run/'environment.json',environment());write(run/'S0/version-audit.json',dict(status='passed_scoped',**lock))
    for name in ('selected-space.json','baseline-space.json'):(run/name).write_bytes((APP/'selected-space.json').read_bytes())
    write(run/'S0/experiment-budget.json',dict(serial=True,max_total_attempts=356,stages={'S0':4,'S1':260,'S2':8,'S3':52,'S4':0,'S5':28,'S6':4},max_new_display_frames=12,max_rss_GiB=16))
    write(run/'S0/resource-policy.json',dict(status='registered',gpu_guard='unchanged .7 of initial free; exact decision sampling',single_live_gpu_model=True,system_free_min_GiB=5,all_new_results_on_data_disk=True,**resources()))
    write(run/'S0/storage-migration.json',dict(status='not_triggered',**resources()))
    Path('/tmp/mpm-restoring-rt0-run.txt').write_text(str(run)+'\n');return run


def verify(run):
    run=Path(run)
    if sha(run/'input-lock.json')!=read(run/'input-lock-sha256.json')['sha256']:raise ValueError('input lock changed')
    lock=read(run/'input-lock.json')
    if lock['application_parent']!=str(APP) or lock['application_release_sha256']!=APP_SHA:raise ValueError('wrong direct parent')
    if sha(run/'plan-frozen.md')!=lock['plan_sha256']:raise ValueError('plan changed')
    audit_parent();return lock

def register(run,name,payload):
    run=Path(run);p=run/name
    if (run/'release.json').exists() or p.exists():raise ValueError('sealed or existing registration')
    sources=all_sources();snapshot(p.parent/(p.stem+'-source'),sources)
    write(p,dict(payload,utc=utc(),application_release_sha256=APP_SHA,source_sha256=sources))
