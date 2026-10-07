"""Strict additive provenance: all ancestors immutable, all new files accounted."""
from pathlib import Path
from datetime import datetime, timezone
from benchmarks.research_sequential_next.provenance import ROOT,read,write,sha,digest,check,utc,resources,environment,serial_lock
from benchmarks.research_post_release.provenance import audit_parent as audit_math,PARENT,snapshot

SPACE_PARENT=ROOT/'docs/results/spatial-phase/20261001T043604Z-spatial-phase'
COST_PARENT=ROOT/'docs/results/cost-phase/20261001T062026Z-cost-phase'
PHASE=ROOT/'docs/results/phase-stress/20261001T072129Z-phase-stress'
BASE=ROOT/'docs/results/basis-allocation/20261001T120146Z-basis-allocation'
BASE_SHA='8527c4c932795037791856384af82bebd7fefd04d244fe366c3a47383ba4726e'
LOCAL=ROOT/'docs/results/local-span/20261001T130726Z-local-span'
LOCAL_SHA='7211bfe7efa8c9ca66389a056bdb466f6fe8dc05563b6fcb088b3ca5bb388d2c'
APP=ROOT/'docs/results/phase-reference/20261001T143142Z-phase-reference'
APP_SHA='e405350d57edd64ccf3e5d8534ab17eb8e9439e5947ddb7aa429ded39c551632'
REFERENCE=ROOT/'docs/results/reference-next/20261001T032104Z-reference-next'
PLAN=ROOT/'docs/MPM_LITE_NEXT_OBSERVABLE_PHASE_AND_PRESSURE_SCOPE_PLAN_20261001_ZH.md'
PROGRESS=ROOT/'docs/MPM_LITE_OBSERVABLE_PRESSURE_NEXT_PROGRESS_20261001_ZH.md'
ANCESTORS=[
 (ROOT/'docs/results/post-release/20261001T013007Z-post-release','5c23b687fc8565f9764690c84f53456694c21588a963776ac735e6a5e6de2d18'),
 (REFERENCE,'fd730ea63ceff7c63570d3600a2932272b645dfeb650930ac7fd882c0aced4b0'),
 (SPACE_PARENT,'b9b458890a170998b57202fd1a926a2477d044e48f2a05909ba714d45d258cb6'),
 (COST_PARENT,'46a3dcd54a30508143185aac06a791e68fb65a2e62eba9245ad30872631fce48'),(PHASE,'018453bda56a570d48482e8f6ee84131d70a9943d524155ea2bb7260b63698a5'),(BASE,BASE_SHA),(LOCAL,LOCAL_SHA),(APP,APP_SHA)]
def all_sources():
    folders=['benchmarks/research_observable_pressure_next','engine/aniso_phase1/research_observable_pressure_next','tests/research_observable_pressure_next']
    return {str(p.relative_to(ROOT)):sha(p) for folder in folders for p in sorted((ROOT/folder).glob('*.py'))}


def source_files():
    return {k:v for k,v in all_sources().items() if k.startswith('engine/') or k.startswith('benchmarks/') and Path(k).stem in ['__init__','provenance','base_config','config','spaces','run']}

def audit_parent(full=False):
    math,base,counts=audit_math(full)
    known=set(base['old_source_sha256'])|set(read(PARENT/'final-source-sha256.json'))
    for folder,expected in ANCESTORS:
        if sha(folder/'release.json')!=expected:raise ValueError('ancestor release changed: '+str(folder))
        pub=read(folder/'release.json')
        for key,entry in pub.items():
            if isinstance(entry,dict) and 'path' in entry and 'sha256' in entry:
                if sha(folder/entry['path'])!=entry['sha256']:raise ValueError('ancestor index changed: '+key)
        sources=read(folder/pub['sources']['path']);known|=set(sources)
        record=dict(sources=check(ROOT,sources))
        if full:
            record.update(snapshots=check(folder/'final-source',sources),artifacts=check(folder,read(folder/pub['artifacts']['path'])))
        docs=read(folder/'documentation.json')
        for key in ('progress','coupling','current_plan'):
            if key+'_path' in docs and sha(ROOT/docs[key+'_path'])!=docs[key+'_sha256']:raise ValueError('ancestor document changed: '+key)
        counts[folder.name]=record
    inventory={str(p.relative_to(ROOT)) for d in ('engine','benchmarks','tests','demos','utils') for p in (ROOT/d).rglob('*.py')}
    unknown=inventory-known-set(all_sources())
    if unknown:raise ValueError('unaccounted source: '+str(sorted(unknown)))
    counts['python_inventory']=len(inventory)
    return read(APP/'release.json'),base,counts

def freeze():
    pub,base,counts=audit_parent(True)
    if resources()['system_free_GiB']<5:raise RuntimeError('migrate inactive data before starting below 5 GiB')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-observable-pressure'
    data=Path('/root/autodl-tmp/mpm-lite-observable-pressure')/stamp;data.mkdir(parents=True,exist_ok=False)
    run=ROOT/'docs/results/observable-pressure'/stamp;run.parent.mkdir(parents=True,exist_ok=True);run.symlink_to(data,target_is_directory=True)
    lock=dict(schema='observable-pressure-input-v1',utc=utc(),application_parent=str(APP),application_release_sha256=APP_SHA,
        parent_release_sha256=sha(PARENT/'release.json'),mathematical_parent=str(PARENT),plan_path=str(PLAN.relative_to(ROOT)),
        plan_sha256=sha(PLAN),energy_scale_J=base['energy_scale_J'],source_sha256=all_sources(),counts=counts)
    write(run/'input-lock.json',lock);write(run/'input-lock-sha256.json',dict(sha256=sha(run/'input-lock.json')))
    (run/'plan-frozen.md').write_bytes(PLAN.read_bytes())
    write(run/'environment.json',environment());write(run/'S0/version-audit.json',dict(status='passed',**lock))
    write(run/'S0/resources.json',dict(**resources(),migration='not_triggered',output=str(data)))
    write(run/'S0/protocol.json',dict(serial=True,formal_space='global-snapshot6',formal_space_changes=0,mass_order=7,full_order=7,
        max_new_time_schemes=1,max_time_candidate_steps=144,max_new_reference_steps=256,final_max_full_cycles=1,final_max_daily_cycles=1,
        final_end_s=1.6,final_max_steps=256,final_max_frames=12,direction_variation_degrees=60,pressure_max_steps=24,pressure_cells=[2,4],
        reference_soft_seconds=600,reference_hard_seconds=1200,max_rss_GiB=16,diagnostic_hypotheses=3,field_rtol=.05,max_performance_candidates=1))
    (run/'selected-space.json').write_bytes((APP/'selected-space.json').read_bytes())
    p=Path('/tmp/mpm-observable-migration.txt')
    write(run/'S0/storage-migration.json',read(Path(p.read_text().strip())) if p.exists() else dict(status='not_triggered',**resources()))
    Path('/tmp/mpm-observable-run.txt').write_text(str(run)+'\n')
    return run


def verify(run):
    run=Path(run)
    if sha(run/'input-lock.json')!=read(run/'input-lock-sha256.json')['sha256']:raise ValueError('input lock changed')
    lock=read(run/'input-lock.json')
    if lock['application_parent']!=str(APP) or lock['application_release_sha256']!=APP_SHA:raise ValueError('wrong direct parent')
    if sha(run/'plan-frozen.md')!=lock['plan_sha256']:raise ValueError('frozen plan changed')
    audit_parent();return lock

def register(run,name,payload):
    run=Path(run);p=run/name
    if (run/'release.json').exists() or p.exists():raise ValueError('sealed run or existing registration')
    sources=all_sources();snapshot(p.parent/(p.stem+'-source'),sources)
    write(p,dict(payload,utc=utc(),application_release_sha256=APP_SHA,source_sha256=sources))

if __name__=='__main__':
    with serial_lock():print(freeze(),flush=True)
