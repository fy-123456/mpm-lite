"""Pre-count attempted steps, including expected failures and recomputations."""
import time,resource
from .provenance import *

def attempt(run,stage,case,callback,*,fault=False):
    run=Path(run);records=[read(p) for p in (run/'attempts').glob('*.json')];budget=read(run/'S0/experiment-budget.json')
    if len(records)>=budget['max_total_attempts'] or sum(x['stage']==stage for x in records)>=budget['stages'][stage]:raise RuntimeError('dynamic budget exhausted')
    path=run/'attempts'/f'{len(records)+1:04d}.json';v=dict(stage=stage,case=case,attempts=1,accepted=False,expected_fault=fault,utc=utc());write(path,v);tick=time.perf_counter()
    try:
        value=callback();v['accepted']=True;return value
    except BaseException as e:v['error']=repr(e);raise
    finally:
        v.update(seconds=time.perf_counter()-tick,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20);write(path,v)

def update(text):
    with PROGRESS.open('a') as f:f.write('\n'+text+'\n')
