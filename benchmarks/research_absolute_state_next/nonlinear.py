"""One full material quadrature endpoint audit, not a nonlinear trajectory."""
import argparse,time,resource
from pathlib import Path
import numpy as np
from .run import write
from engine.aniso_phase1.research_absolute_state_next.state import MaterialStateModel
from engine.aniso_phase1.research_b.tensor import TensorMaterialOperator,TensorRule

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();start=time.perf_counter()
    m=MaterialStateModel();state=m.load(a.run/'S2/final.npz');full=m.r.expand(state.q)
    print('full nonlinear endpoint starts',flush=True)
    op=TensorMaterialOperator(m.space,TensorRule.uniform(m.space.edges,7));out=op.evaluate(full)
    f=m.r.P.T@out['force'];lin=(m.r.K@state.q.ravel()).reshape(state.q.shape)
    energy_lin=.5*float(state.q.ravel()@m.r.K@state.q.ravel());err=float(np.linalg.norm(f-lin)/max(np.linalg.norm(f),1e-12))
    write(a.run/'S2/nonlinear-endpoint.json',dict(scope='full q7 nonlinear endpoint only; trajectory was linearized',force_relative_to_rest_tangent=err,
        nonlinear_U=out['U'],linear_U=energy_lin,energy_absolute=abs(out['U']-energy_lin),material_U=out['material_U'],stabilization_U=out['stabilization_U'],
        minJ=out['min_detF'],material_points=out['material_calls'],seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        passed=err<.05,gate_M_nonlinear_trajectory=False))
    print('full nonlinear endpoint completed; force gap',err,flush=True)
