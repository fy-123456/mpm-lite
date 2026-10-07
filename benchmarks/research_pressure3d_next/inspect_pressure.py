"""New-process read-only validation of a committed D3 research checkpoint."""
import argparse
from .provenance import *
from .fixture import setup
from benchmarks.research_sequential_next.checkpoint import GenerationStore


def inspect(run,case):
    run=Path(run);verify(run)
    if (run/'release.json').exists():
        from .publication import audit_release
        audit_release(run,True)
    if case not in ('cases/pressure32-D3','cases/pressure128-D3','S5/resource-fault'):raise ValueError('unqualified case')
    folder=run/case;ident=read(folder/'identity.json');check(ROOT,ident['numerical_sources'])
    store=GenerationStore(folder,ident);records=store.history();old=store.load()['state'].digest();pointer=sha(store.pointer)
    c,m,cfg,expected=setup(run,ident['grid'],ident['fine'])
    if expected['coupling']!=ident['coupling']:raise ValueError('physical model identity changed')
    item=store.load(validator=c.validate);c.restore(item['state'])
    if c.state.digest()!=old or sha(store.pointer)!=pointer:raise ValueError('read-only load changed commit')
    return dict(status='passed_scoped',case=case,zero_step=True,state_step=c.state.step,time_s=c.state.time,state_digest=old,commit_pointer_unchanged=True,generations=len(records),full_device_validation=True,source_sha256=ident['numerical_sources'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--case',default='S5/resource-fault');p.add_argument('--report',type=Path);a=p.parse_args()
    with serial_lock(a.run):
        result=inspect(a.run,a.case)
        if a.report:
            if (a.run/'release.json').exists():raise ValueError('sealed report is read-only')
            write(a.report,result)
        print(result,flush=True)
