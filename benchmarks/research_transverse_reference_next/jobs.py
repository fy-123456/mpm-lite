"""Serial subprocess ledger includes failed runs and complete setup/teardown cost."""
import argparse, os, subprocess, sys, time
from .provenance import *


def run_job(run, label, command, gpu=False):
    run=Path(run);mutable(run)
    existing=list((run/'processes').glob('*.json'))
    elapsed=sum(read(p)['seconds'] for p in existing if read(p).get('gpu_related')==gpu and 'seconds' in read(p))
    limit=900 if gpu else 600
    if elapsed>=limit: raise RuntimeError('complete process budget exhausted')
    path=run/'processes'/f'{label}.json'
    if path.exists():raise ValueError('job label already recorded')
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def device():
        return subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
    rec=dict(status='running',command=command,gpu_related=gpu,utc=utc(),source_sha256=all_sources(),gpu_before=device())
    snapshot(run/'processes'/f'{label}-source',rec['source_sha256']);write(path,rec)
    tick=time.perf_counter()
    try:
        with (run/'processes'/f'{label}.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(1,limit-elapsed))
        rec.update(returncode=result.returncode,status='passed' if result.returncode==0 else 'failed')
    except Exception as e:
        rec.update(returncode=-1,status='failed',error=repr(e))
    finally:
        rec.update(seconds=time.perf_counter()-tick,gpu_after=device());write(path,rec)
    print(path,rec['status'],rec['seconds'],flush=True)
    return rec['returncode']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--gpu',action='store_true');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    with serial_lock(a.run):code=run_job(a.run,a.label,a.command,a.gpu)
    sys.exit(code)
