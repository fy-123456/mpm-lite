"""Source-complete child of the sealed observable-boundary release."""
from pathlib import Path
from datetime import datetime, timezone
from benchmarks.research_sequential_next.provenance import ROOT,read,write,sha,digest,check,utc,resources,environment,serial_lock
from benchmarks.research_post_release.provenance import audit_parent as audit_math,PARENT,snapshot
from benchmarks.research_pressure_window_next.provenance import ANCESTORS as PREVIOUS,REFERENCE,LOCAL,PHASE_REFERENCE

APP=ROOT/'docs/results/pressure-window/20261004T114530Z-pressure-window'
OLD=ROOT/'docs/results/observable-boundary/20261001T195842Z-observable-boundary'
APP_SHA='de3c5bf7bf94a77b7fe3fd1c7294abe8e0373379511882184f9456118897298b'
ANCESTORS=[*PREVIOUS,(APP,APP_SHA)]
PLAN=ROOT/'docs/MPM_LITE_NEXT_CANDIDATE_REFERENCE_AND_OBSERVABLE_COUPLING_PLAN_20261004_ZH.md'
PROGRESS=ROOT/'docs/MPM_LITE_CANDIDATE_OBSERVABLE_PROGRESS_20261004_ZH.md'

def all_sources():
    folders=['benchmarks/research_candidate_observable_next','engine/aniso_phase1/research_candidate_observable_next','tests/research_candidate_observable_next']
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
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-candidate-observable'
    data=Path('/root/autodl-tmp/mpm-lite-candidate-observable')/stamp;data.mkdir(parents=True,exist_ok=False)
    run=ROOT/'docs/results/candidate-observable'/stamp;run.parent.mkdir(parents=True,exist_ok=True);run.symlink_to(data,target_is_directory=True)
    lock=dict(schema='candidate-observable-input-v1',utc=utc(),application_parent=str(APP),application_release_sha256=APP_SHA,parent_release_sha256=sha(PARENT/'release.json'),mathematical_parent=str(PARENT),plan_path=str(PLAN.relative_to(ROOT)),plan_sha256=sha(PLAN),energy_scale_J=base['energy_scale_J'],source_sha256=all_sources(),counts=counts)
    write(run/'input-lock.json',lock);write(run/'input-lock-sha256.json',dict(sha256=sha(run/'input-lock.json')))
    (run/'plan-frozen.md').write_bytes(PLAN.read_bytes());write(run/'environment.json',environment());write(run/'S0/version-audit.json',dict(status='passed',**lock));write(run/'S0/resources.json',resources())
    for name in ('selected-space.json','baseline-space.json'):(run/name).write_bytes((APP/'selected-space.json').read_bytes())
    write(run/'S0/experiment-budget.json',dict(serial=True,max_total_attempts=1026,stages={'S0':4,'S1':134,'S2':0,'S3':96,'S4':256,'S5':8,'S6':528},max_pressure_fixtures=1,max_static_proposals=2,max_frames=12,max_rss_GiB=16,case_soft_seconds=600,case_hard_seconds=1200,coupling_total_hard_seconds=2400))
    write(run/'S0/scope-map.json',dict(BASELINE_RELEASE=str(APP),SOLID_MAIN=read(run/'selected-space.json'),SPACE_RESEARCH='existing balanced-direction-snapshot6; no promotion',TIME_SELECTED='retain252',coupled_q5=False,production_C_E_integration=False))
    write(run/'S0/initial-full-audit.json',dict(status='passed',counts=counts,prior_cli_stdout=Path('/tmp/mpm-candidate-observable-baseline-audit.json').read_text()))
    write(run/'S0/resource-policy.json',dict(status='registered',gpu_guard='unchanged .7 of initial free memory, exact decision sampling',single_live_gpu_model=True,hard_rss_GiB=16,system_free_min_GiB=5,migrate_only_inactive_unimportant=True,all_new_results_on_data_disk=True))
    write(run/'S0/storage-migration.json',dict(status='not_triggered',**resources()))
    Path('/tmp/mpm-candidate-observable-run.txt').write_text(str(run)+'\n');return run

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
