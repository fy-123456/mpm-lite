"""User-authorized low-disk migration of old export archives, never other jobs."""
import hashlib,json,shutil,time
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def guard(record):
    record=Path(record);before=shutil.disk_usage('/').free
    result=dict(system_free_before=before,threshold_bytes=5*1024**3,data_free=shutil.disk_usage('/root/autodl-tmp').free,time=time.time(),migrated=False)
    source=Path('/root/migration-export');target=Path('/root/autodl-tmp/mpm-lite-research-d/stage2/storage-migration/migration-export')
    if before<result['threshold_bytes'] and source.is_dir() and not source.is_symlink():
        files=list(source.iterdir())
        if any(not f.is_file() for f in files):raise ValueError('unexpected old export member')
        # Preserve every archive byte and original paths. No other direction's
        # outputs, environments, processes or caches are touched.
        target.mkdir(parents=True,exist_ok=True);members={}
        for f in files:
            shutil.copy2(f,target/f.name);h=sha(f)
            if sha(target/f.name)!=h:raise ValueError('migration digest mismatch')
            members[f.name]=dict(sha256=h,bytes=f.stat().st_size)
        for f in files:f.unlink()
        source.rmdir();source.symlink_to(target,target_is_directory=True)
        result.update(migrated=True,source=str(source),target=str(target),members=members)
    result['system_free_after']=shutil.disk_usage('/').free
    record.parent.mkdir(parents=True,exist_ok=True);record.write_text(json.dumps(result,indent=2)+'\n');return result
