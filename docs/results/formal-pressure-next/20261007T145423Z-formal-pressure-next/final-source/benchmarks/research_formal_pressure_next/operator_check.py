import argparse,time,json
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_absolute_state_next.state import MaterialStateModel
from engine.aniso_phase1.research_formal_pressure_next.geometry import Geometry
from engine.aniso_phase1.research_formal_pressure_next.material import Material
from engine.aniso_phase1.research_formal_pressure_next.solver import Solver
from .geometry_check import write

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);args=a.parse_args();start=time.perf_counter()
    m=MaterialStateModel();g=Geometry(m);op=Material(m,7);s=Solver(m,g,op,drained=True);old=s.rest();q=old.q.copy()
    rng=np.random.default_rng(41);q[m.r.free]=rng.normal(size=(len(m.r.free),3))*1e-6
    q[m.r.fixed[m.space.carrier_X[m.space.fixed_scalar_ids,0]>=.75],0]=-1e-5
    direction=np.zeros_like(q);direction[m.r.free]=rng.normal(size=(len(m.r.free),3))*1e-6
    w=np.r_[direction.ravel()[s.free],.003,-.002];p=np.array([.001,.002]);h=.0025
    # Same-state material screening, including total force and tangent, never mass/geometry.
    reference=op.evaluate(q,direction);lower=Material(m,5);candidate=lower.evaluate(q,direction)
    error=lambda a,b:float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))
    screen={k:error(candidate[k],reference[k]) for k in ('force','weak_moments','tangent_action')}
    screen.update(energy_absolute=abs(candidate['U']-reference['U']),reference_calls=reference['material_calls'],candidate_calls=candidate['material_calls'],candidate_accepted=all(v<.03 for v in screen.values()))
    write(args.run/'S2/material-screen.json',screen);print('material screen',screen,flush=True)
    exact=s.jvp(old,q,p,h,w);rows=[]
    for eps in (1e-3,3e-4):
        plus=s.residual(old,q+eps*direction,p+eps*w[-2:],h)['residual'];minus=s.residual(old,q-eps*direction,p-eps*w[-2:],h)['residual']
        fd=(plus-minus)/(2*eps)
        rows.append(dict(eps=eps,total_relative=error(exact,fd),force_relative=error(exact[:-2],fd[:-2]),mass_relative=error(exact[-2:],fd[-2:])))
        print('Jv',rows[-1],flush=True)
    report=dict(rows=rows,seconds=time.perf_counter()-start,material_calls=op.calls,material_seconds=op.seconds,geometry_points=g.points,
        scope='one mixed nonzero formal state, two epsilon directions; full q7 material and original stabilization; constant reference H')
    write(args.run/'S2/operator-check.json',report)
    assert all(x['force_relative']<.001 and x['mass_relative']<.001 for x in rows)
