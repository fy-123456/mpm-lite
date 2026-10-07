"""Storage recovery with persistent originals and disposable new tail output.

Read-only source cases remain on the system disk throughout computation. Only
regenerable output is written on a separate tmpfs. Completed output is copied
back and checked before the original case is renamed to a persistent backup.
No source data or backup is deleted and no project symlink is introduced.
"""
import hashlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warp as wp
from benchmarks import aniso_material_resume as resume
from benchmarks.aniso_material_history import ROOT,BASE,OUT,load,write,hashes
EVIDENCE=BASE/'v14-recovery2';BACKUP=BASE/'v14-storage-attempt-2'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def worker(name,temporary):
    resume.PAUSED=OUT;resume.OUT=Path(temporary)/'new-results'
    wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
    return resume.worker(name)

def main():
    if len(sys.argv)>1:raise SystemExit(0 if worker(sys.argv[1],sys.argv[2]) else 2)
    old=load(OUT/'batch.json');failed=[r['case'] for r in old if r['exit_code']];assert len(failed)==2
    assert all('StoragePaused' in load(OUT/'cases'/n/'status.json')['error'] for n in failed)
    assert hashes()==load(OUT/'protocol.json')['source_sha256']
    EVIDENCE.mkdir(exist_ok=False);temporary=Path(tempfile.mkdtemp(prefix='mpm-lite-v14-new-tail-',dir='/dev/shm'));temporary.chmod(0o700);dest=temporary/'new-results';(dest/'cases').mkdir(parents=True);shutil.copyfile(OUT/'protocol.json',dest/'protocol.json')
    source_hashes={str(p.relative_to(ROOT)):digest(p) for n in failed for p in sorted((OUT/'cases'/n).rglob('*')) if p.is_file()}
    write(EVIDENCE/'protocol.json',dict(source_sha256=digest(Path(__file__)),reused_worker_sha256=digest(ROOT/'benchmarks/aniso_material_resume.py'),source_files=source_hashes,temporary_output=str(temporary),source_preserved_on_disk=True,no_deletion=True,cases=failed,storage_guard='unchanged; MPM_LITE_DATA_ROOT points to the separate filesystem holding new outputs'))
    shutil.copyfile(OUT/'batch.json',EVIDENCE/'interrupted-batch.json');shutil.copyfile('/tmp/mpm-v14-resume.log',EVIDENCE/'interrupted-driver.log');shutil.copyfile('/tmp/mpm-v14-analysis-resumed.log',EVIDENCE/'interrupted-analysis.log')
    def launch(n):
        with (EVIDENCE/(n+'.log')).open('x') as f:r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_material_resume_safe',n,str(temporary)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'MPM_LITE_DATA_ROOT':str(temporary),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        print(n,r.returncode,flush=True);return dict(case=n,exit_code=r.returncode)
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(launch,failed))
    write(EVIDENCE/'tail-batch.json',results);assert all(r['exit_code']==0 for r in results)
    for f,h in source_hashes.items():assert digest(ROOT/f)==h,f
    BACKUP.mkdir(exist_ok=False);(BACKUP/'cases').mkdir();shutil.copyfile(OUT/'protocol.json',BACKUP/'protocol.json');shutil.copyfile(OUT/'batch.json',BACKUP/'batch.json')
    proofs={n:load(OUT/'cases'/n/'resume-proof.json') for n in ('baseline-finest',*failed)}
    new_proofs={}
    for n in failed:
        src=dest/'cases'/n;dst=OUT/'cases'/(n+'-verified-copy');shutil.copytree(src,dst)
        hashes_new={str(p.relative_to(src)):digest(p) for p in src.rglob('*') if p.is_file()}
        for f,h in hashes_new.items():assert digest(dst/f)==h,f
        (OUT/'cases'/n).rename(BACKUP/'cases'/n);dst.rename(OUT/'cases'/n)
        (OUT/(n+'.log')).rename(BACKUP/(n+'.log'));shutil.copyfile(EVIDENCE/(n+'.log'),OUT/(n+'.log'))
        new_proofs[n]=load(OUT/'cases'/n/'resume-proof.json')
    all_results=[r for r in old if not r['exit_code']]+results;write(OUT/'batch.json',all_results)
    summary=dict(passed=True,stages=[dict(method='first disk recovery',source='benchmarks/aniso_material_resume.py',source_sha256=digest(ROOT/'benchmarks/aniso_material_resume.py'),cases=proofs),dict(method='disposable tail on separate filesystem; originals retained',source='benchmarks/aniso_material_resume_safe.py',source_sha256=digest(Path(__file__)),cases=new_proofs)],all_originals_preserved_on_disk=True,new_results_returned_to_project=True)
    assert all(r['passed'] and max(r['errors'].values())<1e-10 for stage in summary['stages'] for r in stage['cases'].values())
    write(OUT/'resume-summary.json',summary);write(EVIDENCE/'completion.json',dict(passed=True,source_files_verified=len(source_hashes),new_results_copied_back=True,temporary_results_retained=str(temporary)))
    print('SAFE_RECOVERY_COMPLETE',flush=True)

if __name__=='__main__':main()
