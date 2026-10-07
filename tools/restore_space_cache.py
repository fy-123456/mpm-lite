"""Reassemble an authenticated, byte-exact cache; never silently overwrite data."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def restore():
    folder=ROOT/'data/research-space-cache';m=json.loads((folder/'parts.json').read_text());target=ROOT/m['target']
    if not target.resolve().is_relative_to(ROOT):raise ValueError('external cache target')
    if target.exists():
        if sha(target)!=m['sha256']:raise ValueError('existing cache differs; inspect it before replacing')
        print('Cache already valid:',m['sha256']);return
    target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_name(target.name+'.restore.tmp')
    h=hashlib.sha256();total=0;created=False
    try:
        with temporary.open('xb') as out:
            created=True
            for item in m['parts']:
                p=folder/item['path']
                if not p.resolve().is_relative_to(folder):raise ValueError('external part')
                if p.stat().st_size!=item['size'] or sha(p)!=item['sha256']:raise ValueError('corrupt cache part: '+item['path'])
                with p.open('rb') as inp:
                    while block:=inp.read(8<<20):out.write(block);h.update(block);total+=len(block)
            out.flush();os.fsync(out.fileno())
        if total!=m['size'] or h.hexdigest()!=m['sha256']:raise ValueError('reconstructed cache does not match original')
        os.replace(temporary,target);print('Restored exact cache:',h.hexdigest(),total,'bytes')
    except BaseException:
        # Only remove the incomplete output created by this invocation.
        if created and temporary.exists():temporary.unlink()
        raise
if __name__=='__main__':restore()
