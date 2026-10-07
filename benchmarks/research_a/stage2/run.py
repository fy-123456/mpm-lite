"""Start or resume A stage-two work with bounded, sequential child processes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .protocol import freeze, run_path, write_json, sha, GIB, ROOT


def read(path):return json.loads(Path(path).read_text())


def stage(run,tag,action,name=None,test=False):
    p=read(run/'protocol.json');status=run/f'{tag}-status.json'
    if status.exists():
        old=read(status)
        if old['returncode']!=0:raise RuntimeError(f'Preserved failure in {tag}; use a new run identity for repairs')
        return old
    command=[p['python'],'-m','unittest','tests.research_a.stage2.test_cases','tests.research_a.stage2.test_adaptive','-v'] if test else [p['python'],'-m','benchmarks.research_a.stage2.worker',action,'--run',str(run)]
    if name:command+=['--name',name]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMBA_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH',None)
    start=time.monotonic();stop=None;max_rss=0
    with (run/f'{tag}.log').open('x') as log:
        process=subprocess.Popen(command,cwd=p['source_checkout'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(run/f'{tag}-started.json',dict(pid=process.pid,command=command,source=p['source_checkout'],utc=datetime.now(timezone.utc).isoformat()))
        while process.poll() is None:
            try:
                content=Path(f'/proc/{process.pid}/status').read_text()
                current=int(next(line.split()[1] for line in content.splitlines() if line.startswith('VmRSS:')))*1024
                max_rss=max(max_rss,current)
            except (FileNotFoundError,StopIteration):pass
            if max_rss>24*GIB:stop='24 GiB resident-memory limit'
            if time.monotonic()-start>3600:stop='3600 s stage time limit'
            if shutil.disk_usage('/').free<5*GIB:stop='system disk below 5 GiB'
            if shutil.disk_usage(run/'arrays').free<5*GIB:stop='data disk below 5 GiB'
            if stop:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                break
            time.sleep(2)
        code=process.wait()
    result=dict(returncode=code,stop_reason=stop,seconds=time.monotonic()-start,max_sampled_rss_bytes=max_rss,
                log_sha256=sha(run/f'{tag}.log'),exploratory_timing=True)
    write_json(status,result)
    print(tag,result,flush=True)
    if code or stop:raise RuntimeError(f'{tag} stopped; see {tag}.log')
    return result


def execute(run,phase):
    p=read(run/'protocol.json')
    if phase=='kickoff':
        stage(run,'tests','',test=True)
        stage(run,'case-construction','construction')
        stage(run,'reference-pilot','reference')
    else:
        specs=p['candidates']
        if phase=='budgets':specs=specs[:3]
        elif phase=='ablation':specs=[s for s in specs[3:] if s['case']=='F45']
        elif phase=='generalization':
            baseline=next(s for s in specs if s['case']=='F45' and s['budget']==144)
            audit=run/'candidates'/baseline['name']/'nonlinear-audit.json'
            if not audit.exists() or not read(audit)['passed']:raise RuntimeError('A8 baseline strategy operator audit must precede A9')
            specs=[s for s in specs if s['case']!='F45']
        elif phase=='audit':
            specs=[s for s in specs if (run/'candidates'/s['name']/'summary.json').exists()]
        for spec in specs:
            name=spec['name'];action='audit' if phase=='audit' else 'candidate'
            stage(run,action+'-'+name,action,name)
    progress(run)


def progress(run):
    p=read(run/'protocol.json');results=[]
    for spec in p['candidates']:
        folder=run/'candidates'/spec['name']
        item=dict(**spec,status='not_started')
        if (folder/'summary.json').exists():item.update(status='linear_solved',summary=read(folder/'summary.json'))
        elif folder.exists():item['status']='failed' if (folder/'failure.json').exists() else 'running_or_interrupted'
        if (folder/'nonlinear-audit.json').exists():item['nonlinear_audit']=read(folder/'nonlinear-audit.json')
        results.append(item)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    write_json(run/f'progress-{stamp}.json',dict(candidates=results,parent_bundle_sha256=p['parent_bundle_sha256'],
        research_complete=False,spatial_certified=False,production_defaults_changed=False,
        A10='not executed; no new certified final holdout available',A11='exploratory timing only; no three-repeat fair measurement',
        A12='no recommended spatial package until candidate operator and reload audits complete'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--phase',choices=['prepare','kickoff','budgets','ablation','audit','generalization','progress'],required=True)
    a=parser.parse_args();run=run_path(a.run_id)
    if a.phase=='prepare':print(freeze(a.run_id),flush=True)
    elif a.phase=='progress':progress(run)
    else:execute(run,a.phase)

if __name__=='__main__':main()
