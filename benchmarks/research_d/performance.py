"""D9: three independent processes, alternating AB/BA, full static solve costs.

A = inherited scipy CG + TensorElastic separable preconditioner (CPU float64).
B = identical frozen tensor equation + guarded resident Warp PCG (GPU float64).
"""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np
import warp as wp
from scipy.sparse.linalg import LinearOperator,cg
from .protocol import CASES,GATES,problem,environment,command,storage_check
from engine.aniso_phase1.research_d.identity import write_json,own_sources,digest
from engine.aniso_phase1.research_d.gpu import TensorGPU,ResidentPCG


class ResourceMonitor:
    def __init__(self):
        self.samples=[];self.done=threading.Event();self.thread=threading.Thread(target=self.run,daemon=True)
    def run(self):
        while not self.done.is_set():
            self.samples.append(dict(time=time.time(),gpu=command(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']),
                processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])))
            self.done.wait(.25)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*args):self.done.set();self.thread.join()
    def exclusive(self):
        for s in self.samples:
            for line in s['processes'].splitlines():
                pid=line.split(',')[0].strip()
                if not pid.isdigit() or int(pid)!=os.getpid():return False
        return bool(self.samples)


def native_solve(p,rtol=1e-7):
    its=0
    def callback(_):
        nonlocal its
        its+=1
    A=LinearOperator((p.size,p.size),matvec=p.apply,dtype=float)
    M=LinearOperator(A.shape,matvec=p.precondition,dtype=float)
    x,info=cg(A,p.rhs,M=M,rtol=rtol,atol=1e-12,maxiter=2500,callback=callback)
    error=float(np.linalg.norm(p.rhs-p.apply(x)));target=max(1e-12,rtol*np.linalg.norm(p.rhs))
    return x,dict(iterations=its,true_residual=error,target=target,converged=bool(info==0 and error<=target),
                  backend='inherited scipy CG + v22 separable preconditioner')


def trial(case,kind):
    start=time.perf_counter();p=problem(case);common=time.perf_counter()-start
    build=time.perf_counter();gpu=None;solver=None
    if kind=='gpu':gpu=TensorGPU(p);solver=ResidentPCG(gpu,maxiter=2500,check_every=8)
    backend_build=time.perf_counter()-build
    solve_start=time.perf_counter()
    if solver:x,info=solver.solve(p.rhs,rtol=1e-7);r=info.record()
    else:x,r=native_solve(p)
    solve_seconds=time.perf_counter()-solve_start
    metrics=p.metrics(x);total=time.perf_counter()-start
    cold_field=p.field(x)
    warm=[];warm_metrics=[]
    for _ in range(3):
        t=time.perf_counter()
        if solver:x,info=solver.solve(p.rhs,rtol=1e-7);record=info.record()
        else:x,record=native_solve(p)
        m=p.metrics(x);warm.append(time.perf_counter()-t);warm_metrics.append(dict(solve=record,physics=m))
    row=dict(kind=kind,case=case['name'],free_dofs=p.size,common_preprocess_seconds=common,
        backend_build_seconds=backend_build,first_solve_seconds=solve_seconds,total_cold_seconds=total,
        warm_full_solve_seconds=warm,cold_solve=r,physics=metrics,warm_records=warm_metrics,
        physics_field_sha256=digest(cold_field),
        bytes=None if solver is None else dict(backend=gpu.memory_bytes,workspace=solver.memory_bytes),
        semantics='cold = common axis/eigenbasis and lifting + backend/workspace + solve/capture/readback + reaction/energy; warm = solve/readback + reaction/energy')
    if solver:solver.close();gpu.close()
    del p;gc.collect();return row


def worker(out,index,cpu):
    if cpu is not None:os.sched_setaffinity(0,{cpu})
    cache,storage=storage_check('/tmp/mpm-lite-research-d-cache');wp.config.kernel_cache_dir=cache;wp.init()
    # Compile/load is a separately recorded cold-process cost, excluded from steady comparisons.
    start=time.perf_counter();warm=problem(CASES[0]);g=TensorGPU(warm);s=ResidentPCG(g,maxiter=2500)
    s.solve(warm.rhs);s.close();g.close();module_warmup=time.perf_counter()-start;del warm;gc.collect()
    before=environment();rows=[]
    with ResourceMonitor() as monitor:
        # Include a cheap control to expose cold-start regressions, never cherry-pick.
        for case in [CASES[0],CASES[1],CASES[3],CASES[4],CASES[5],CASES[6]]:
            sequence=('cpu','gpu','gpu','cpu') if index%2==0 else ('gpu','cpu','cpu','gpu')
            for order,kind in enumerate(sequence):
                row=trial(case,kind);row.update(order=order,pair=order//2);rows.append(row)
    write_json(out,dict(process_index=index,pid=os.getpid(),before=before,after=environment(),storage=storage,
        module_warmup_seconds=module_warmup,rows=rows,monitor=monitor.samples,gpu_exclusive=monitor.exclusive(),
        cpu_isolation='single allowed core affinity; other directions run on shared host, system load recorded'))


def stats(values):
    return dict(count=len(values),mean=float(np.mean(values)),median=float(np.median(values)),p95=float(np.percentile(values,95)))


def summarize(paths):
    processes=[json.loads(Path(p).read_text()) for p in paths]
    records=[];all_valid=True
    for name in sorted({r['case'] for p in processes for r in p['rows']}):
        rows=[r for p in processes for r in p['rows'] if r['case']==name]
        physical_ok=all(r['cold_solve']['converged'] and r['physics']['finite'] and r['physics']['constraint_residual']==0
                        and all(x['solve']['converged'] and x['physics']['finite'] for x in r['warm_records']) for r in rows)
        reactions=[r['physics']['reaction_N'] for r in rows]
        physical_ok &= max(reactions)-min(reactions)<=GATES['reaction_absolute_N']+GATES['field_relative']*max(map(abs,reactions))
        valid=physical_ok and all(p['gpu_exclusive'] for p in processes);all_valid &= valid
        timing={kind:dict(cold=stats([r['total_cold_seconds'] for r in rows if r['kind']==kind]),
                         warm=stats([s for r in rows if r['kind']==kind for s in r['warm_full_solve_seconds']])) for kind in ('cpu','gpu')}
        cold_speedup=timing['cpu']['cold']['median']/timing['gpu']['cold']['median']
        warm_speedup=timing['cpu']['warm']['median']/timing['gpu']['warm']['median']
        pairs=[]
        for proc in processes:
            rr=[r for r in proc['rows'] if r['case']==name]
            for pair in (0,1):
                a=next(r for r in rr if r['pair']==pair and r['kind']=='cpu')
                b=next(r for r in rr if r['pair']==pair and r['kind']=='gpu')
                pairs.append(dict(process_index=proc['process_index'],pair=pair,
                    order='AB' if a['order']<b['order'] else 'BA',cold_ratio=a['total_cold_seconds']/b['total_cold_seconds'],
                    warm_ratio=float(np.median(a['warm_full_solve_seconds'])/np.median(b['warm_full_solve_seconds']))))
        # Require EVERY independent process pair to improve before a favorable claim.
        stable=all(p['cold_ratio']>1 for p in pairs)
        decision='optional_GPU_candidate' if valid and stable and cold_speedup>=1/(1-GATES['speedup_candidate']) else 'retain_CPU_for_cold_solve'
        records.append(dict(case=name,timing=timing,cold_speedup=cold_speedup,warm_speedup=warm_speedup,
            cold_regression_gt_3pct=cold_speedup<1/(1+GATES['regression_limit']),valid=valid,
            paired_ratios=pairs,decision=decision,
            failure_rate=sum(not r['cold_solve']['converged'] for r in rows)/len(rows)))
    return dict(passed=all_valid,independent_processes=len(processes),cases=records,
                default_changed=False,scope='fixed static tensor equations only; excludes A/B/C/E adaptation and moving steps',
                measurement_limit='GPU process ownership sampled at 250 ms; CPU core pinned but host is shared; no system-wide reservation')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--out',required=True);p.add_argument('--worker',type=int)
    p.add_argument('--cpu',type=int);a=p.parse_args();out=Path(a.out)
    if a.worker is not None:worker(out,a.worker,a.cpu);return
    out.mkdir(parents=True,exist_ok=False)
    cpu=a.cpu if a.cpu is not None else max(os.sched_getaffinity(0))
    write_json(out/'protocol.json',dict(source_sha256=own_sources(),environment=environment(),cpu_core=cpu,
        processes=3,order='AB/BA in every process; reversed initial order in odd process',gates=GATES,
        noise_policy='preserve every observation, GPU interference invalidates; gain must hold in all process pairs'))
    paths=[]
    for i in range(3):
        path=out/f'process-{i}.json';paths.append(path)
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (out/f'process-{i}.log').open('x') as log:
            subprocess.run([sys.executable,'-u','-m',__package__+'.performance','--out',str(path),'--worker',str(i),'--cpu',str(cpu)],
                           check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
        print('completed performance process',i,flush=True)
    result=summarize(paths);write_json(out/'summary.json',result);print(json.dumps(result),flush=True)
    raise SystemExit(0 if result['passed'] else 2)

if __name__=='__main__':main()
