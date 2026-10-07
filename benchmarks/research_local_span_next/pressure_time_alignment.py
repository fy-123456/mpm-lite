"""Align the normal-storage witness to the already committed small-storage times."""
from pathlib import Path
import argparse
import numpy as np
import scipy.linalg as la
from .provenance import read,write,sha,register,serial_lock,source_files
from .coupling_study import setup
from engine.aniso_phase1.research_local_span_next.rt0 import TensorGridCoupling

def study(run):
    run=Path(run);previous=read(run/'S5/fixed-solid-transient.json');small=previous['records'][1];times=small['times_s']
    register(run,'S5/common-time-protocol.json',dict(reason='final plan audit found per-storage spectral steps did not satisfy common physical time requirement',
        existing_evidence_sha256=sha(run/'S5/fixed-solid-transient.json'),normal_storage=.2,times_s=times,new_fixed_solid_steps=4,new_coupled_steps=0,small_storage_reused=True,old_records_preserved=True))
    model,cfg=setup(run);c=TensorGridCoupling(model,cfg,times,storage=.2,fixed_solid=True)
    H=c.core.geometry.evaluate(c.state.q)['H'];B=c.core.B;C=c.core.capacity;rate=(B@la.solve(H,B.T,assume_a='pos'))/C[:,None]
    p0=np.full(2,.01);pe=np.full(2,.002);rows=[];errors=[]
    for t in times[1:]:
        row=c.step();rows.append(row);ref=pe+la.expm(-rate*t)@(p0-pe);errors.append(float(np.max(abs(row['pressure_Pa']-ref))))
    f=c.state.child_states['fluid'];mass=float(np.sum(np.asarray(f['content_m3'])-C*p0)+f['cumulative_boundary_m3'])
    if max(errors)>5e-4+.05*.01 or abs(mass)>1e-10 or min(x['darcy_dissipation_J'] for x in rows)<0 or max(x['true_scaled_residual'] for x in rows)>1:raise ValueError('common-time physical gate failed')
    normal=dict(storage=.2,times_s=times,steps=4,rows=rows,matrix_exponential_max_error_Pa=max(errors),cumulative_mass_defect_m3=mass,eigenvalues_s_inverse=la.eigvals(rate).real.tolist())
    write(run/'S5/fixed-common-time.json',dict(status='passed_scoped',times_s=times,records=[normal,small],old_result_sha256=sha(run/'S5/fixed-solid-transient.json'),new_fixed_steps=4,
        numeric_sources=source_files(),pressure_spatial_accuracy=False,general_monotonicity=False,interpretation='different storage materials at same physical times; each compared to its own same-system matrix exponential'))
    print('COMMON_TIME_FIXED',max(errors),mass,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
