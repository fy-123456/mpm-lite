"""Explicit descendant inventory; sealed ancestor validators are never patched."""
from pathlib import Path
from datetime import datetime, timezone
from benchmarks.research_sequential_next.provenance import ROOT,read,write,sha,digest,check,utc,resources,environment,serial_lock
from benchmarks.research_post_release.provenance import audit_parent as audit_math,PARENT,snapshot
from benchmarks.research_transverse_reference_next.provenance import ANCESTORS as PREVIOUS,SOLID

APP=ROOT/'docs/results/transverse-reference/20261005T101215Z-transverse-reference'
APP_SHA='e948656e54181fd2cb3bfc35db518ae071c54898d638453cd4bd3922e46b83b1'
ANCESTORS=[*PREVIOUS,(APP,APP_SHA)]
PLAN=ROOT/'docs/MPM_LITE_NEXT_COUPLED_DYNAMIC_REFERENCE_AND_BD_COST_PLAN_20261005_ZH.md'
PROGRESS=ROOT/'docs/MPM_LITE_COUPLED_DYNAMIC_REFERENCE_AND_BD_COST_PROGRESS_20261005_ZH.md'

def all_sources():
    folders=('benchmarks/research_coupled_reference_next','engine/aniso_phase1/research_coupled_reference_next','tests/research_coupled_reference_next')
    return {str(p.relative_to(ROOT)):sha(p) for d in folders for p in sorted((ROOT/d).glob('*.py'))}

def numerical_sources():
    return {k:v for k,v in all_sources().items() if k.startswith('engine/') or (k.startswith('benchmarks/') and Path(k).stem in ('__init__','provenance','fixture'))}

def audit_parent(full=False):
    _,base,counts=audit_math(full);known=set(base['old_source_sha256'])|set(read(PARENT/'final-source-sha256.json'))
    for folder,expected in ANCESTORS:
        if sha(folder/'release.json')!=expected:raise ValueError('ancestor release changed '+str(folder))
        pub=read(folder/'release.json')
        for e in pub.values():
            if isinstance(e,dict) and 'path' in e and 'sha256' in e:
                if sha(folder/e['path'])!=e['sha256']:raise ValueError('ancestor index changed')
        sources=read(folder/pub['sources']['path']);known|=set(sources)
        record=dict(sources=check(ROOT,sources))
        if full:record.update(snapshots=check(folder/'final-source',sources),artifacts=check(folder,read(folder/pub['artifacts']['path'])))
        docs=read(folder/'documentation.json')
        for key in ('progress','coupling','current_plan'):
            if key+'_path' in docs and sha(ROOT/docs[key+'_path'])!=docs[key+'_sha256']:raise ValueError('ancestor document changed')
        counts[folder.name]=record
    inventory={str(p.relative_to(ROOT)) for d in ('engine','benchmarks','tests','demos','utils') for p in (ROOT/d).rglob('*.py')}
    if inventory-known-set(all_sources()):raise ValueError('unaccounted Python source '+str(sorted(inventory-known-set(all_sources()))))
    counts['python_inventory']=len(inventory)
    return read(APP/'release.json'),base,counts

def freeze():
    pub,base,counts=audit_parent(True)
    if resources()['system_free_GiB']<5:raise RuntimeError('inactive-data migration required')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-coupled-reference'
    data=Path('/root/autodl-tmp/mpm-lite-coupled-reference')/stamp;data.mkdir(parents=True)
    run=ROOT/'docs/results/coupled-reference'/stamp;run.parent.mkdir(parents=True,exist_ok=True);run.symlink_to(data,target_is_directory=True)
    lock=dict(schema='coupled-reference-input-v1',utc=utc(),application_parent=str(APP),application_release_sha256=APP_SHA,energy_scale_J=base['energy_scale_J'],plan_sha256=sha(PLAN),counts=counts,source_sha256=all_sources())
    write(run/'input-lock.json',lock);write(run/'input-lock-sha256.json',dict(sha256=sha(run/'input-lock.json')))
    (run/'plan-frozen.md').write_bytes(PLAN.read_bytes());write(run/'environment.json',environment())
    write(run/'S0/version-audit.json',dict(status='passed_scoped',**lock));write(run/'S0/resource-policy.json',dict(single_GPU=True,GPU_process_seconds=900,CPU_analysis_seconds=600,output_GiB=2,**resources()))
    write(run/'S0/storage-migration.json',dict(status='not_triggered',**resources()))
    write(run/'S0/experiment-budget.json',dict(max_total_attempts=36,stages={'S2':23,'S3':10,'repair':3},max_new_frames=2))
    for name in ('selected-space.json','S0/physical-contract.json','S0/observation-contract.json'):
        p=run/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((APP/name).read_bytes())
    write(run/'S0/checkpoint-sources.json',dict(parent_records=read(APP/'S0/checkpoint-sources.json')['records'],BD_store=str(APP/'S3/continuous'),BD_identity_sha256=sha(APP/'S3/continuous/identity.json'),policy='authenticated physical state, execution backend separately bound'))
    PROGRESS.write_text(f'# 运动骨架耦合参考与 BD 成本实施记录\n\n执行 [{PLAN.name}]({PLAN.name})。\n\n最新基线 `{APP.name}`，SHA `{APP_SHA}`。结果 `{run}`。\n\nS0：23源码/23快照/623产物及祖先全审计通过。原144空间、M7/q7、128格与200微秒身份保持；新数据/缓存独立，系统盘{resources()["system_free_GiB"]:.2f} GiB，不触发迁移。\n')
    Path('/tmp/mpm-coupled-reference-run.txt').write_text(str(run)+'\n');return run

def verify(run):
    run=Path(run);lock=read(run/'input-lock.json')
    if sha(run/'input-lock.json')!=read(run/'input-lock-sha256.json')['sha256']:raise ValueError('input lock changed')
    if lock['application_parent']!=str(APP) or lock['application_release_sha256']!=APP_SHA:raise ValueError('foreign application parent')
    if sha(run/'plan-frozen.md')!=lock['plan_sha256']:raise ValueError('frozen plan changed')
    audit_parent();return lock

def mutable(run):
    if (Path(run)/'release.json').exists():raise ValueError('sealed run is read only')
    return verify(run)

def register(run,name,payload):
    p=Path(run)/name
    if p.exists() or (Path(run)/'release.json').exists():raise ValueError('registration is immutable')
    sources=all_sources();snapshot(p.parent/(p.stem+'-source'),sources)
    write(p,dict(payload,utc=utc(),application_release_sha256=APP_SHA,source_sha256=sources))

if __name__=='__main__':
    with serial_lock(None):print(freeze(),flush=True)
