"""Additive, source-complete child of the sealed cross-direction release."""
from pathlib import Path
from datetime import datetime, timezone
from benchmarks.research_sequential_next.provenance import ROOT,read,write,sha,digest,check,utc,resources,environment,serial_lock
from benchmarks.research_post_release.provenance import audit_parent as audit_math,PARENT,snapshot
from benchmarks.research_phase_boundary_next.provenance import ANCESTORS as PREVIOUS,REFERENCE,LOCAL,PHASE_REFERENCE

APP=ROOT/'docs/results/phase-boundary/20261001T183820Z-phase-boundary'
APP_SHA='5ac018b8e9e6cbce4df61614adc4ab28325cce1a7805f0555b44498df724cdd9'
ANCESTORS=[*PREVIOUS,(APP,APP_SHA)]
PLAN=ROOT/'docs/MPM_LITE_NEXT_OBSERVABLE_PHASE_AND_EARLY_DRAINAGE_PLAN_20261002_ZH.md'
PROGRESS=ROOT/'docs/MPM_LITE_OBSERVABLE_BOUNDARY_NEXT_PROGRESS_20261002_ZH.md'

def all_sources():
    folders=['benchmarks/research_observable_boundary_next','engine/aniso_phase1/research_observable_boundary_next','tests/research_observable_boundary_next']
    return {str(p.relative_to(ROOT)):sha(p) for folder in folders for p in sorted((ROOT/folder).glob('*.py'))}

def source_files():
    core={'__init__','provenance','base_config','config','spaces','run','physics'}
    return {k:v for k,v in all_sources().items() if k.startswith('engine/') or k.startswith('benchmarks/') and Path(k).stem in core}

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
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-observable-boundary'
    data=Path('/root/autodl-tmp/mpm-lite-observable-boundary')/stamp;data.mkdir(parents=True,exist_ok=False)
    run=ROOT/'docs/results/observable-boundary'/stamp;run.parent.mkdir(parents=True,exist_ok=True);run.symlink_to(data,target_is_directory=True)
    lock=dict(schema='observable-boundary-input-v1',utc=utc(),application_parent=str(APP),application_release_sha256=APP_SHA,parent_release_sha256=sha(PARENT/'release.json'),mathematical_parent=str(PARENT),plan_path=str(PLAN.relative_to(ROOT)),plan_sha256=sha(PLAN),energy_scale_J=base['energy_scale_J'],source_sha256=all_sources(),counts=counts)
    write(run/'input-lock.json',lock);write(run/'input-lock-sha256.json',dict(sha256=sha(run/'input-lock.json')))
    (run/'plan-frozen.md').write_bytes(PLAN.read_bytes());write(run/'environment.json',environment());write(run/'S0/version-audit.json',dict(status='passed',**lock));write(run/'S0/resources.json',resources())
    for name in ('selected-space.json','baseline-space.json'):(run/name).write_bytes((APP/'selected-space.json').read_bytes())
    write(run/'S0/protocol.json',dict(serial=True,formal_functions=144,max_time_steps=576,initial_time_steps=384,rebase_reserved_steps=192,max_new_space_candidates=0,max_candidate_dynamic_steps=16,max_pressure_grids=2,max_pressure_cells=32,max_fixed_steps=64,max_coupled_steps_per_grid=8,max_performance_candidates=1,max_performance_steps=6,max_full_cycles=1,max_q5_cycles=1,max_full_steps=364,max_frames=12,max_rss_GiB=16,soft_seconds=600,hard_seconds=1200))
    write(run/'S0/scope-map.json',dict(BASELINE_RELEASE=str(APP),SOLID_MAIN=read(run/'selected-space.json'),SPACE_RESEARCH='existing balanced-direction-snapshot6; bounded dynamic research only; no promotion',TIME_SELECTED='pending S1',PRESSURE_SELECTED='pending complete S2 gate',sensitive_q5=False,coupled_q5=False))
    write(run/'S0/storage-migration.json',dict(status='not_triggered',**resources()))
    Path('/tmp/mpm-observable-boundary-run.txt').write_text(str(run)+'\n');return run

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
