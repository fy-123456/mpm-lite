"""Restore a nonzero absolute state in a fresh process, no particle histories."""
import argparse,json
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_absolute_state_next.state import MaterialStateModel
from engine.aniso_phase1.research_unified_lite_poro.model import sha_arrays
from .run import write

def inspect(path):
    m=MaterialStateModel();s=m.load(path);X=np.array([[.2,.5,.5],[.4,.45,.5],[.6,.55,.5],[.8,.5,.5]])
    x,F,v=m.fields(s,X)
    return dict(state_digest=s.digest(),fields_digest=sha_arrays(x,F,v),time=s.time,step=s.step,shape=list(s.q.shape),
        state_bytes=s.q.nbytes+s.velocity.nbytes,particle_arrays=0,scope='original material absolute state; no fluid or production transfer replay')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=inspect(a.checkpoint);write(a.output,r);print(json.dumps(r))
