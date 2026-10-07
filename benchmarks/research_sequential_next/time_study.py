"""N03: one process, bounded windows sharing actual committed initial states."""
from pathlib import Path
import argparse
import json
from .provenance import read,write,serial_lock,source_files,utc
from .checkpoint import GenerationStore
from .run import create_config,run_case
from .compare import compare_cases


def run_windows(run):
    run=Path(run);parent=run/'cases/gpu-q7-dt005'
    history=GenerationStore(parent,read(parent/'identity.json')).history()
    lookup={round(x['state'].time,10):x['folder']/'state.json' for x in history}
    windows=[('start',0.,.1),('load-turn',.45,.65),('unload-turn',1.,1.2)]
    comparisons=[]
    for label,start,end in windows:
        names=[]
        for dt,suffix in ((.025,'dt0025'),(.0125,'dt00125')):
            name=f'window-{label}-{suffix}';names.append(name)
            folder=run/'cases'/name
            if not folder.exists():create_config(run,name,dt=dt,start=start,end=end,initial=lookup[round(start,10)])
            if not (folder/'summary.json').exists() or read(folder/'summary.json')['status']!='passed_scoped':run_case(run,name)
        result=compare_cases(run,*names)
        comparisons.append(dict(window=label,coarse=names[0],fine=names[1],passed=result['passed']))
        write(run/'N03/windows.json',dict(utc=utc(),comparisons=comparisons,source_sha256=source_files(),
            scope='same committed window initial state; local time-step sensitivity only'))
        print(label,'engineering comparison passed:',result['passed'],flush=True)
    return comparisons


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);a=p.parse_args()
    with serial_lock(a.run):run_windows(a.run)
