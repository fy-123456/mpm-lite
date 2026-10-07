"""Fresh-process load and bounded-state field observations; not a GPU transfer."""
import argparse,json,time
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_absolute_state_next.state import MaterialStateModel
from engine.aniso_phase1.research_formal_pressure_next.state import load
from .geometry_check import write

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();case=a.run/'S4/closed';m=MaterialStateModel()
    identity=json.loads((case/'identity.json').read_text());s=load(case/'checkpoint.npz',m,identity);summary=json.loads((case/'summary.json').read_text())
    assert s.digest()==summary['state_digest']
    with np.load(case/'replay-probes.npz') as d:
        fields=m.fields(s,d['X'])
        gaps={k:float(np.max(abs(z-d[k]))) for k,z in zip(('x','F','v'),fields)}
    assert max(gaps.values())<1e-10
    observations=[];rng=np.random.default_rng(3)
    for n in (16,54,128):
        X=rng.uniform([.125,.375,.375],[.875,.625,.625],size=(n,3));t=time.perf_counter();x,F,v=m.fields(s,X)
        observations.append(dict(points=n,load_seconds=time.perf_counter()-t,min_probe_J=float(np.linalg.det(F).min()),state_bytes=s.q.nbytes+s.velocity.nbytes+s.p.nbytes,state_digest=s.digest()))
    write(a.run/'S3/replay.json',dict(digest=s.digest(),gaps=gaps,observations=observations,scope='absolute coupled q/v/p and cumulative ledgers; particle-count independent state, no solve timing claim'))
