import argparse,time
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_absolute_state_next.state import MaterialStateModel
from engine.aniso_phase1.research_formal_pressure_next.geometry import Geometry
from engine.aniso_phase1.research_formal_pressure_next.transaction import Transaction
from engine.aniso_phase1.research_formal_pressure_next.solver import Solver
from .geometry_check import write

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();start=time.perf_counter()
    m=MaterialStateModel();g=Geometry(m);tx=Transaction(m,g,primary=2,reserve=7);solver=Solver(m,g,tx.reserve)
    old=solver.rest();digest=old.digest();unit=np.zeros_like(old.q);unit[m.r.fixed[m.space.carrier_X[m.space.fixed_scalar_ids,0]>=.75],0]=1
    target=(unit*(-.1*.0025**2)).ravel()[solver.fixed]
    for op in (tx.primary,tx.reserve):op.deadline=start+600;op.limit=24
    trial,row=tx.step(old,.0025,target)
    direct=None
    if row['fallback']:direct,dr=solver.step(old,.0025,target)
    assert old.digest()==digest
    report=dict(actual_material_failure=row['fallback'],attempts=row['attempts'],unchanged_digest=digest,
        direct_q_gap=None if direct is None else float(abs(trial.q-direct.q).max()),direct_p_gap=None if direct is None else float(abs(trial.p-direct.p).max()),delta_U_rule=row['delta_U_rule'],
        seconds=time.perf_counter()-start,calls={'q2':tx.primary.calls,'q7':tx.reserve.calls},
        scope='one first formal coupled step; q2 diagnostic only; direct q7 equality required only after actual retry; fault control covered separately')
    write(a.run/'S3/transaction.json',report);print('transaction',row['attempts'],flush=True)
    if row['fallback']:
        assert np.allclose(trial.q,direct.q,rtol=1e-4,atol=1e-10) and np.allclose(trial.p,direct.p,rtol=1e-4,atol=1e-10)
    else:
        assert all(e<=tx.budget for e in row['attempts'][0]['comparison'].values())
