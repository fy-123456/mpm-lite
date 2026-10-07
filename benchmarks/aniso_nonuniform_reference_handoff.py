"""Adopt surviving reference jobs after isolating an unhealthy GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .aniso_nonuniform_reference import OUT,FINE,CASES,catalog,guard,save


def alive(job):
    p=Path('/proc')/str(job['pid'])
    try:
        state=(p/'stat').read_text().rsplit(') ',1)[1].split()[0]
        args=(p/'cmdline').read_bytes().split(b'\0')
        return state not in ('Z','X') and b'benchmarks.aniso_nonuniform_reference' in args
    except FileNotFoundError:return False


def run(out):
    h=json.loads((out/'handoff.json').read_text());jobs=[j for j in h['jobs'] if j.get('device')!='cuda:1' and alive(j)]
    failed=[j for j in h['jobs'] if j.get('device')=='cuda:1']
    if any(alive(j) for j in failed):raise RuntimeError('faulty child must be stopped before handoff')
    children={};logs={};completed=[]
    def launch(stage,case,dt):
        name=f'{"run" if stage=="run" else "checks"}-{case}-dt{dt:.13f}.log';path=out/name
        if stage=='run' and path.exists():path.rename(out/(path.stem+'-gpu1-stalled.log'))
        log=path.open('w');cmd=[sys.executable,'-u','-m','benchmarks.aniso_nonuniform_reference','--stage',stage,'--case',case,'--dt',str(dt),'--out',str(out)]
        env=os.environ.copy()
        if stage=='run':
            # Only the missing local-Y coarse trajectory is relaunched. Physical
            # GPU 0 has capacity for two independent 4.6 GiB jobs.
            env['CUDA_VISIBLE_DEVICES']='0';cmd+=['--device','cuda:0']
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        job=dict(pid=p.pid,stage=stage,case=case,dt=dt)
        if stage=='run':job.update(device='cuda:0',physical_gpu=0)
        children[p.pid]=p;logs[p.pid]=log;jobs.append(job);print('START',job,flush=True)
    if 2*FINE not in catalog(out,'localY'):launch('run','localY',2*FINE)
    save(out/'status.json',dict(state='running',message='GPU 1 已隔离；三条正常轨迹保留，局部 Y 粗时间步已在 GPU 0 重启。'))
    while jobs or any(not (out/f'check-{c}-dt{FINE:.13f}.json').exists() for c in CASES):
        guard()
        for job in list(jobs):
            if job['pid'] in children:
                code=children[job['pid']].poll();running=code is None
            else:code=None;running=alive(job)
            if running:continue
            if job['pid'] in logs:logs.pop(job['pid']).close()
            valid=(job['dt'] in catalog(out,job['case']) if job['stage']=='run' else (out/f'check-{job["case"]}-dt{FINE:.13f}.json').exists())
            if code not in (None,0) or not valid:
                save(out/'status.json',dict(state='stopped',message='接管任务失败，其他进程保持运行：'+str(job)))
                raise RuntimeError('adopted/new child failed: '+str(job))
            jobs.remove(job);completed.append(job);print('END',job,flush=True)
        active_checks={j['case'] for j in jobs if j['stage']=='check'}
        for case in CASES:
            if case in active_checks or (out/f'check-{case}-dt{FINE:.13f}.json').exists():continue
            deps={case,'base','Y'}|({'localY'} if case=='fineY' else set())
            if len(active_checks)<2 and all(FINE in catalog(out,c) and 2*FINE in catalog(out,c) for c in deps):
                launch('check',case,FINE);active_checks.add(case)
        save(out/'handoff-live.json',dict(active=jobs,completed=completed,excluded_gpu=1))
        if jobs:time.sleep(10)
    failed=[c for c in CASES if not json.loads((out/f'check-{c}-dt{FINE:.13f}.json').read_text())['time']['passed']]
    if failed:raise RuntimeError('TIME REFINEMENT REQUIRED '+str(failed))
    for stage in ('analyze','audit','plot'):
        print('STAGE',stage,flush=True)
        with (out/f'{stage}.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_nonuniform_reference','--stage',stage,'--dt',str(FINE),'--out',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True)
    from .aniso_nonuniform_report import report
    report(out)
    save(out/'status.json',dict(state='complete',message='四条新增轨迹与独立审计完成，详见 RESULTS_ZH.md。'))
    subprocess.run([sys.executable,'-m','benchmarks.aniso_nonuniform_reference','--stage','progress','--once','--out',str(out)],check=True)
    print('COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    try:run(a.out)
    except Exception as exc:
        save(a.out/'status.json',dict(state='stopped',message='接管队列停止：'+str(exc)+'；现有结果及其他进程保留。'))
        raise


if __name__=='__main__':main()
