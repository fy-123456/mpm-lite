"""Run nonuniform Y time pairs and retain validated uniform X/Y/Z controls."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from .aniso_nonuniform_reference import OUT,CASES,FINE,catalog,guard,save,fingerprint,initial


def run_queue(out,fine,devices):
    tasks=[('fineY',fine),('localY',fine),('fineY',2*fine),('localY',2*fine)]
    tasks += [(c,dt) for c in ('base','X','Y','Z') for dt in (fine,2*fine)]
    tasks=[(c,dt) for c,dt in tasks if dt not in catalog(out,c)]
    gpu={};cpu={};available=list(devices);pending=set(CASES);completed=[];checks=[]
    def launch(stage,case,dt,device=None):
        log=(out/f'{"run" if stage=="run" else "checks"}-{case}-dt{dt:.13f}.log').open('w')
        command=[sys.executable,'-u','-m','benchmarks.aniso_nonuniform_reference','--stage',stage,
                 '--case',case,'--dt',str(dt),'--out',str(out)]
        if device is not None:command.extend(['--device',device])
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        print('START',stage,case,dt,device,flush=True)
        return dict(process=proc,log=log,case=case,dt=dt,device=device,start=time.time())
    try:
        while tasks or gpu or pending or cpu:
            guard()
            while tasks and available:
                case,dt=tasks.pop(0);device=available.pop(0);gpu[device]=launch('run',case,dt,device)
            for case in sorted(pending):
                deps={case,'base','Y'}|({'localY'} if case=='fineY' else set())
                ready=all(fine in catalog(out,c) and 2*fine in catalog(out,c) for c in deps)
                if ready and len(cpu)<3:cpu[case]=launch('check',case,fine);pending.remove(case)
            for active,records in ((gpu,completed),(cpu,checks)):
                for key,job in list(active.items()):
                    code=job['process'].poll()
                    if code is None:continue
                    job['log'].close();del active[key]
                    record=dict(case=job['case'],dt=job['dt'],device=job['device'],exit=code,elapsed=time.time()-job['start'])
                    records.append(record);print('END',record,flush=True)
                    save(out/'schedule.json',dict(trajectories=completed,checks=checks,fine_dt=fine))
                    if code:raise RuntimeError(f'child failed: {record}')
                    if active is gpu:available.append(key)
            if tasks or gpu or pending or cpu:time.sleep(10)
    finally:
        for active in (gpu,cpu):
            for job in active.values():job['process'].terminate();job['process'].wait();job['log'].close()
    failed=[c for c in CASES if not json.loads((out/f'check-{c}-dt{fine:.13f}.json').read_text())['time']['passed']]
    if failed:raise RuntimeError(f'TIME REFINEMENT REQUIRED {failed}; keep all completed results')
    for stage in ('analyze','audit','plot'):
        print('STAGE',stage,flush=True)
        with (out/f'{stage}.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_nonuniform_reference','--stage',stage,
                            '--dt',str(fine),'--out',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ.copy())
    print('COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--fine-dt',type=float,default=FINE);p.add_argument('--devices',default='cuda:2,cuda:0,cuda:3');a=p.parse_args()
    guard();a.out.mkdir(parents=True,exist_ok=True);protocol=json.loads((a.out/'protocol.json').read_text())
    if protocol['initial']!=fingerprint(initial()) or not protocol['initial_passed'] or not protocol['equal_cost_passed']:
        raise RuntimeError('successful prepare required before trajectories')
    if a.fine_dt<=0 or not math.isfinite(a.fine_dt) or abs(round(.003/a.fine_dt)*a.fine_dt-.003)>1e-14:
        raise ValueError('positive finite time step dividing duration required')
    devices=a.devices.split(',')
    if not devices or len(set(devices))!=len(devices):raise ValueError('distinct devices required')
    progress_command=[sys.executable,'-u','-m','benchmarks.aniso_nonuniform_reference','--stage','progress','--out',str(a.out)]
    # Check devices before allocating full material arrays or starting children.
    import warp as wp
    wp.init();available={str(d) for d in wp.get_cuda_devices()}
    check=dict(requested=devices,available=sorted(available),passed=all(d in available for d in devices))
    save(a.out/'gpu-preflight.json',check)
    if not check['passed']:
        save(a.out/'status.json',dict(state='blocked_cuda',message='CUDA 驱动初始化失败；完整新轨迹尚未启动。CPU 验证和旧对照结果见各自报告。',gpu=check))
        subprocess.run(progress_command+['--once'],check=False)
        raise RuntimeError('CUDA devices unavailable; no trajectory started: '+str(check))
    save(a.out/'status.json',dict(state='running',message='CUDA 检查通过，参考轨迹／诊断队列运行中。'))
    with (a.out/'progress-writer.log').open('w') as log:
        writer=subprocess.Popen(progress_command,stdout=log,stderr=subprocess.STDOUT)
        try:
            run_queue(a.out,a.fine_dt,devices)
        except BaseException as exc:
            save(a.out/'status.json',dict(state='stopped',message='队列停止：'+str(exc)+'；已完成结果保留。'))
            raise
        else:
            save(a.out/'status.json',dict(state='complete',message='新增轨迹及独立审计完成；空间结论见 results.json。'))
        finally:
            if writer.poll() is None:writer.terminate()
            writer.wait()
            subprocess.run(progress_command+['--once'],stdout=log,stderr=subprocess.STDOUT,check=False)


if __name__=='__main__':main()
