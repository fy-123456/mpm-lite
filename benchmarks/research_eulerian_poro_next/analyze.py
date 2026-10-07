"""Common-site short-window metrics and bounded per-call accounting."""
import argparse,json,time
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_unified_lite_poro.space import Space
from engine.aniso_phase1.research_eulerian_poro_next.bridge import ImprovedBridge
from .diagnose import write


def array_bytes(obj):
    seen=set()
    def visit(x):
        if id(x) in seen:return 0
        seen.add(id(x))
        if isinstance(x,np.ndarray):
            return visit(x.base) if isinstance(x.base,np.ndarray) else x.nbytes
        if isinstance(x,dict):return sum(visit(v) for v in x.values())
        if isinstance(x,(tuple,list,set)):return sum(visit(v) for v in x)
        if hasattr(x,'__dict__'):return visit(vars(x))
        return 0
    return visit(obj)


def analyze(run):
    s=Space();X,w,_=s.rule(4);N,_=s.basis(X);comparisons=[]
    pairs=[('PPC','director-short-ppc2','director-short-ppc4'),
           ('dt_halving','director-short-ppc4','director-halfdt'),
           ('dt_quarter','director-halfdt','director-quarterdt')]
    for label,left,right in pairs:
        with np.load(run/'S3'/left/'terminal.npz') as a,np.load(run/'S3'/right/'terminal.npz') as b:
            ua,ub=N@a['q'],N@b['q'];va,vb=N@a['v'],N@b['v']
            comparisons.append(dict(kind=label,left=left,right=right,time=.02,
                displacement_relative=float(np.linalg.norm(ua-ub)/np.linalg.norm(ub)),
                displacement_max_m=float(abs(ua-ub).max()),
                velocity_relative=float(np.linalg.norm(va-vb)/np.linalg.norm(vb)),
                velocity_max_m_s=float(abs(va-vb).max()),
                pressure_max_Pa=float(abs(a['p']-b['p']).max()),
                pressure_relative=float(np.linalg.norm(a['p']-b['p'])/np.linalg.norm(b['p'])),
                pressure_relative_to_initial_0p2=float(abs(a['p']-b['p']).max()/.2)))
    write(run/'S3/common-site-comparisons.json',dict(rows=comparisons,temporal_accuracy=False,
        reason='displacement converges but terminal velocity remains time-sensitive; do not certify phase accuracy'))
    with np.load(run/'S3/director-cycle/terminal.npz') as d:q=d['q'].copy();v=d['v'].copy();p=d['p'].copy()
    np.savez_compressed(run/'S2/replay-input.npz',q=q,v=v,p=p,time=np.array(.06))
    scaling=[]
    for ppc in (2,3,4,8):
        start=time.perf_counter();b=ImprovedBridge(s,ppc);op,_,_,_=b.prepare();prep=time.perf_counter()-start
        op.material(q);times=[];n0=op.material_evaluations
        for i in range(3):
            start=time.perf_counter();op.material(q);times.append(time.perf_counter()-start)
        scaling.append(dict(Np=len(b.state.particles.X),Nq=len(op.points),per_call_evaluations=(op.material_evaluations-n0)//3,
            prepare_s=prep,material_median_s=float(np.median(times)),material_range_s=[min(times),max(times)],
            nested_unique_numpy_bytes=array_bytes(op)))
    write(run/'S6/operator-scaling.json',dict(scope='single operator call; excludes process peak and Python overhead',rows=scaling))
    print(json.dumps(dict(comparisons=comparisons,scaling=scaling),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();analyze(a.run)
