"""Losslessly pack the storage-paused attempt after all overlap proofs pass."""
import hashlib,shutil,time,zipfile
from benchmarks.aniso_material_resume import PAUSED,OUT
from benchmarks.aniso_material_history import load,write

def main():
    names=[r['case'] for r in load(PAUSED/'batch.json') if r['exit_code']]
    while not all((OUT/'cases'/n/'resume-proof.json').exists() for n in names):time.sleep(10)
    assert all(load(OUT/'cases'/n/'resume-proof.json')['passed'] for n in names)
    files=sorted(p for p in (PAUSED/'cases').rglob('*') if p.is_file());hashes={str(p.relative_to(PAUSED)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};size=sum(p.stat().st_size for p in files)
    target=PAUSED/'paused-cases.zip'
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(PAUSED))
    with zipfile.ZipFile(target) as z:
        for p,h in hashes.items():assert hashlib.sha256(z.read(p)).hexdigest()==h,p
    write(PAUSED/'paused-case-sha256.json',hashes);write(PAUSED/'packed-attempt.json',dict(verified=True,files=len(files),original_bytes=size,zip_bytes=target.stat().st_size,restore='unzip paused-cases.zip inside v14-storage-attempt; every byte verified against paused-case-sha256.json; old v1-v13 archives untouched'))
    shutil.rmtree(PAUSED/'cases');print('verified lossless archive',size,target.stat().st_size,flush=True)

if __name__=='__main__':main()
