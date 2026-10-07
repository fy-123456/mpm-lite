"""Attempt accounting; failures remain counted and accepted generations stay atomic."""
import time,resource
from .provenance import *

def total_attempts(run,stage=None):
    return sum(read(f)['attempts'] for f in (Path(run)/'attempts').glob('*.json') if stage is None or read(f)['stage']==stage)

def attempt(run,stage,case,callback,*,fault=False):
    run=Path(run);budget=read(run/'S0/experiment-budget.json')
    if total_attempts(run)>=budget['max_total_attempts'] or total_attempts(run,stage)>=budget['stages'][stage]:raise RuntimeError('dynamic attempt budget exceeded')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20>16:raise MemoryError('RSS budget')
    counter=total_attempts(run)+1;path=run/'attempts'/f'{counter:04d}.json';v=dict(stage=stage,case=case,attempts=1,accepted=False,expected_fault=fault,utc=utc());write(path,v);tick=time.perf_counter()
    try:
        value=callback();v['accepted']=True;return value
    except BaseException as e:v['error']=repr(e);raise
    finally:v.update(seconds=time.perf_counter()-tick,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20);write(path,v)

def update(run,text):
    with PROGRESS.open('a') as f:f.write('\n'+text+'\n')
