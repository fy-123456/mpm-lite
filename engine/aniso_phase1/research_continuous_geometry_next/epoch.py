"""Append an authenticated time epoch without changing residual normalization."""
import copy
import numpy as np
from engine.aniso_phase1.research_startup_substeps_next.schedule import validate_times
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_d.common_state import StateTransaction
from engine.aniso_phase1.research_restoring_rt0_next.fixture import physical_payload

def extension_times(original,fine=False,*,expected_prefix=None):
    old=validate_times(original)
    if expected_prefix is not None and not np.array_equal(old,np.asarray(expected_prefix)):raise ValueError('changed authenticated time prefix')
    if len(old)!=29 or old[-1]!=2e-4:raise ValueError('authenticated 28-step prefix required')
    out=np.r_[old,np.linspace(2e-4,3e-4,(16 if fine else 8)+1)[1:]]
    validate_times(out);out.setflags(write=False);return out

def install(c,fine=False,*,expected_prefix):
    c.validate(c.state);s=c.state;before=physical_payload(s);old=copy.deepcopy(c.identity)
    if s.step!=28 or s.time!=c.times[-1] or c.core.window!=2e-4 or np.any(c.source):raise ValueError('extension must start at authenticated zero-source 200us state')
    times=extension_times(c.times,fine,expected_prefix=expected_prefix);theta=np.r_[c.core.thetas,np.full(len(times)-len(c.times),.5)];theta.setflags(write=False)
    c.times=times;c.core.thetas=theta;c.core.identity=dict(c.core.identity,theta_schedule=theta.tolist(),window_s=2e-4,execution_end_s=float(times[-1]),epoch='authenticated-200-to-300us')
    c.core.signature=digest(c.core.identity);c.identity=dict(schema='continuous-geometry-time-epoch-v1',physical_parent=old,core=c.core.identity,times_s=times.tolist(),source_m3_s=c.source.tolist(),residual_normalization_window_s=2e-4,execution_end_s=float(times[-1]),inherited_steps=28);c.signature=digest(c.identity)
    s.child_states['fluid']['model']=c.core.signature;s.child_states['explicit_pressure_grid']=c.signature
    if physical_payload(s)!=before:raise ValueError('epoch changed physical history')
    c.core._transaction=StateTransaction(s,validator=c.validate);c.validate(s)
    return dict(status='passed_scoped',physical_payload_exact=True,physical_payload_sha256=digest(before),old_identity=old,new_identity=c.identity,old_cumulative_Dnum_J=s.child_states['fluid']['cumulative_numerical_dissipation_J'],residual_normalization_window_s=c.core.window,new_end_s=float(times[-1]))
