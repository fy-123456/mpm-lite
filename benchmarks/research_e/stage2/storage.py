"""Conditional, verified relocation of this project's obsolete export archive."""
from pathlib import Path
import shutil
from engine.aniso_phase1.research_e.stage2.handoff import sha,save


def guard(out):
    minimum=5*1024**3;before=shutil.disk_usage('/').free
    record=dict(system_free_before=before,threshold_bytes=minimum,data_free=shutil.disk_usage('/root/autodl-tmp').free,migrated=False)
    archive=Path('/root/migration-export/mpm-lite.tar.gz')
    if before<minimum and archive.is_file() and not archive.is_symlink():
        expected='dbfd45a0b2281832edf6743c6e7fc7dd701201d77a3c4ae07e34cc05bcecaa24'
        if sha(archive)!=expected:raise ValueError('historical export changed; do not move unverified data')
        destination=Path(out).parent/'migrated-export'/archive.name;destination.parent.mkdir(exist_ok=True)
        if destination.exists():raise ValueError('migration destination already exists')
        shutil.copy2(archive,destination)
        if sha(destination)!=expected:raise ValueError('migration copy hash mismatch')
        # Keep original path available via a symlink; only verified duplicate bytes removed.
        link=archive.with_name(archive.name+'.E-stage2-link')
        link.symlink_to(destination);link.replace(archive)
        record.update(migrated=True,source=str(archive),target=str(destination),sha256=expected,bytes=destination.stat().st_size)
    record['system_free_after']=shutil.disk_usage('/').free
    logs=Path(out)/'storage-observations.json'
    import json
    history=json.loads(logs.read_text()) if logs.exists() else []
    history.append(record);save(logs,history)
    if record['system_free_after']<minimum:
        raise RuntimeError('system disk remains below 5 GiB after eligible E migration; stop before new experiment')
    return record
