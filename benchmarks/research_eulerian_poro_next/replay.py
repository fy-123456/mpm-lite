"""Separate process restore or particle-free material/geometry replay."""
import argparse,json
from pathlib import Path
import numpy as np
from .diagnose import write


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path);p.add_argument('--package',type=Path)
    p.add_argument('--state',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.checkpoint:
        if a.package or a.state:raise ValueError('choose checkpoint recovery or isolated operator replay')
        from engine.aniso_phase1.research_eulerian_poro_next.checkpoint import load
        b=load(a.checkpoint)
        result=dict(state_digest=b.state.digest(),step=b.state.step,time=b.state.time,rule=b.rule,
                    scope='complete checkpoint restore; particles loaded intentionally')
    else:
        if not a.package or not a.state:raise ValueError('package and coefficient-only state required')
        from engine.aniso_phase1.research_unified_lite_poro.model import FrozenOperator
        op=FrozenOperator.load_package(a.package)
        with np.load(a.state,allow_pickle=False) as d:
            if set(d.files)!={'q','v','p','time'}:raise ValueError('coefficient-only state required')
            q=d['q'].copy()
        E,f,P=op.material(q);V,G,H,J=op.geometry(q)
        result=dict(energy_J=E,force=f.tolist(),volume=V.tolist(),min_detF=J,material_points=len(op.points),
                    particle_arrays_loaded=False,full_coupled_solve=False,
                    scope='fresh-process operator evaluation only; inherited solver, no extra physical attempt')
    write(a.output,result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
