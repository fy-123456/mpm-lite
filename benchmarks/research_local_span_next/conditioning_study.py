"""Diagnose mixed physical units without changing an accepted pressure trajectory."""
from pathlib import Path
import argparse,warnings
import numpy as np
import scipy.linalg as la
from .provenance import read,write,register,serial_lock
from .coupling_study import coupled

def study(run):
    run=Path(run);register(run,'S5/conditioning-protocol.json',dict(trigger='general dense solver reports low reciprocal condition on mixed physical-unit matrix',hypotheses=['unit and coordinate scaling','actual rank loss','unacceptable physical residual'],new_time_steps=0,no_regularization=True))
    c,m,cfg=coupled(run);core=c.core;h=c.times[1]-c.times[0];s=c.state;g=core.geometry.evaluate(s.q);H=g['H'];A=core.rest_matrix(h);n=len(m.ids);p0=np.asarray(s.child_states['fluid']['pressure_Pa']);p1=p0+h*c.source/core.capacity;pbar=(p0+p1)/2
    force=m.evaluate(s.q)['force']-core.alpha*np.einsum('k,kij->ij',pbar,g['gradient'])
    residual=np.r_[h*force[m.free].ravel(),core.capacity*(p1-p0)-h*c.source,-core.B.T@pbar+core.gb]
    rhs=-residual
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');raw=la.solve(A,rhs,assume_a='gen')
    row=1/np.max(abs(A),axis=1);scaled=row[:,None]*A;col=1/np.max(abs(scaled),axis=0);equilibrated=scaled*col[None,:]
    balanced=col*la.solve(equilibrated,row*rhs,assume_a='gen')
    def rcond(mat):
        lu,piv=la.lu_factor(mat);result,info=la.lapack.dgecon(lu,la.norm(mat,1))
        if info:raise ValueError('condition estimate failed')
        return float(result)
    tol=np.r_[np.full(n,1e-10),np.full(2,1e-12),np.full(11,1e-10)]
    errors=dict(raw_true_scaled_residual=float(np.max(abs(A@raw-rhs)/tol)),balanced_true_scaled_residual=float(np.max(abs(A@balanced-rhs)/tol)),
        displacement_increment_max_m=float(h*np.max(abs(raw[:n]-balanced[:n]))),pressure_increment_max_Pa=float(np.max(abs(raw[n:n+2]-balanced[n:n+2]))),flux_increment_max_m3_s=float(np.max(abs(raw[n+2:]-balanced[n+2:]))))
    rr,rb=rcond(A),rcond(equilibrated)
    if max(errors['raw_true_scaled_residual'],errors['balanced_true_scaled_residual'])>1 or errors['displacement_increment_max_m']>5e-5 or errors['pressure_increment_max_Pa']>5e-4:raise ValueError('condition warning affects physical solution')
    closure=read(run/'S5/coupled-closure.json')
    write(run/'S5/conditioning-analysis.json',dict(status='explained_scoped',raw_rcond_1norm=rr,equilibrated_rcond_1norm=rb,condition_improvement=rb/rr,warnings=[str(x.message) for x in caught],
        row_scale_range=[float(row.min()),float(row.max())],column_scale_range=[float(col.min()),float(col.max())],errors=errors,
        actual_trajectory_max_scaled_residual=closure['max_scaled_residual'],actual_mass_defect=closure['cumulative_mass_defect_m3'],actual_energy_defect=closure['max_energy_defect_J'],
        decision='retain accepted four-step general solve; no rank truncation or artificial diagonal; equilibration is a future numerical conditioning option',new_time_steps=0))
    print('MIXED_CONDITION',rr,rb,errors,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
