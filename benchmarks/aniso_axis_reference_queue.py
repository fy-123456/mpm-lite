"""Run a common temporal pair and overlap independent reference diagnostics."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .aniso_axis_reference import OUT,CASES,FINE,catalog,guard,save,fingerprint,initial


def run_queue(out,fine,devices):
    # Long fine trajectories first; each completed GPU slot admits one coarse
    # trajectory. A slot is freed only after its child has actually exited.
    tasks=[(c,fine) for c in ('Y','X','Z')]+[(c,2*fine) for c in ('Y','X','Z')]
    for dt in (fine,2*fine):
        if dt not in catalog(out,'base'):tasks.append(('base',dt))
    tasks=[(c,dt) for c,dt in tasks if dt not in catalog(out,c)]
    gpu={};cpu={};available=list(devices);pending=set(CASES);completed=[];checks=[]
    def launch(stage,case,dt,device=None):
        path=out/(f'{case}-{dt:.13f}.log' if stage=='run' else f'checks-{case}-dt{dt:.13f}.log')
        log=path.open('w')
        command=[sys.executable,'-u','-m','benchmarks.aniso_axis_reference','--stage',stage,
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
            for case in list(pending):
                own=catalog(out,case);base=catalog(out,'base')
                if fine in own and 2*fine in own and fine in base and 2*fine in base:
                    cpu[case]=launch('check',case,fine);pending.remove(case)
            for active,records in ((gpu,completed),(cpu,checks)):
                for key,job in list(active.items()):
                    code=job['process'].poll()
                    if code is None:continue
                    job['log'].close();del active[key]
                    record=dict(case=job['case'],dt=job['dt'],device=job['device'],exit=code,
                                elapsed=time.time()-job['start'])
                    records.append(record);print('END',record,flush=True)
                    save(out/'schedule.json',dict(trajectories=completed,checks=checks,fine_dt=fine))
                    if code:raise RuntimeError(f'child failed: {record}')
                    if active is gpu:available.append(key)
            if tasks or gpu or pending or cpu:time.sleep(10)
    finally:
        for active in (gpu,cpu):
            for job in active.values():
                job['process'].terminate();job['process'].wait();job['log'].close()
    import json
    failed=[]
    for case in CASES:
        row=json.loads((out/f'check-{case}-dt{fine:.13f}.json').read_text())
        if not row['time_passed']:failed.append(case)
    if failed:raise RuntimeError(f'TIME REFINEMENT REQUIRED {failed}; keep all completed results')
    for stage in ('analyze','audit','plot'):
        print('STAGE',stage,flush=True)
        with (out/f'{stage}.log').open('w') as log:
            subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_axis_reference','--stage',stage,'--out',str(out)],
                           stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ.copy())
    print('COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--fine-dt',type=float,default=FINE)
    p.add_argument('--devices',default='cuda:0,cuda:1,cuda:2,cuda:3');a=p.parse_args()
    guard();a.out.mkdir(parents=True,exist_ok=True)
    import json
    protocol=json.loads((a.out/'protocol.json').read_text())
    if protocol['initial']!=fingerprint(initial()) or not protocol['representation_passed']:
        raise RuntimeError('run --stage prepare successfully before starting trajectories')
    devices=a.devices.split(',')
    if not devices or len(set(devices))!=len(devices):raise ValueError('distinct devices required')
    run_queue(a.out,a.fine_dt,devices)


if __name__=='__main__':main()
