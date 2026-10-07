"""Independent process: frozen residual/material/geometry without particles."""
import argparse,json
import numpy as np
from engine.aniso_phase1.research_unified_lite_poro.model import FrozenOperator
from engine.aniso_phase1.research_unified_lite_poro.checkpoint import load

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint');p.add_argument('--advance',action='store_true');a=p.parse_args()
    op=FrozenOperator.load_package(a.package);q=np.zeros((len(op.space.free),3));q[0,0]=1e-4
    E,g,_=op.material(q);V,G,H,J=op.geometry(q)
    result=dict(energy=E,force=g.tolist(),volume=V.tolist(),min_detF=J,particle_arrays_loaded=False,Nq=len(op.points))
    if a.advance:
        from engine.aniso_phase1.research_unified_lite_poro.solve import advance
        zero=np.zeros_like(q)
        q1,v1,p1,z,metrics=advance(op,zero,zero,np.full(op.top.cells,.2),.005,0.)
        result['independent_coupled_step']=dict(accepted=True,physical_attempts=1,metrics=metrics)
    if a.checkpoint:
        b=load(a.checkpoint);result['checkpoint_digest']=b.state.digest();result['checkpoint_step']=b.state.step
        result['particle_arrays_loaded_for_separate_checkpoint_check']=True
    from pathlib import Path
    Path(a.output).write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
