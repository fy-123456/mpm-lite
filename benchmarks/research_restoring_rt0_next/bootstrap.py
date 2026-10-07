from pathlib import Path
import argparse
import numpy as np
from .provenance import APP,SOLID,APP_SHA,read,write,register,source_files,serial_lock,resources
from .run import create_config
from benchmarks.research_phase_stress_next.time_study import history

def prepare(run):
    run=Path(run);create_config(run,'compatibility',end=.05,field_cache=True,display_frames=1)
    write(run/'S0/interface-contract.json',dict(parent=APP_SHA,space=read(run/'selected-space.json'),numeric_sources=source_files(),default_reuse_transpose_buffers=True,pressure_default=False,material_default='full_only'))
    write(run/'S0/dependency-invalidation.json',dict(space_changed=False,material_changed=False,old_q5_permissions_valid_only_in_parent=True,new_permissions_require_current_sources=True))

def check(run):
    run=Path(run);a=history(run/'cases/compatibility');b=history(SOLID/'cases/final-full')[:5];errors={}
    for key in ('q','velocity','predictor'):
        errors[key]=max(0. if getattr(x['state'],key) is None and getattr(y['state'],key) is None else float(np.max(abs(getattr(x['state'],key)-getattr(y['state'],key)))) for x,y in zip(a,b))
    for key in ('reaction_N','energy_balance_J','budget_defect_J'):
        errors[key]=max(abs(x[key]-y[key]) for x,y in zip(a[-1]['rows'],b[-1]['rows']))
    if len(a)!=5 or len(read(run/'cases/compatibility/segments.json'))!=2 or max(errors.values())>1e-8:raise ValueError('compatibility failed')
    write(run/'S0/compatibility.json',dict(status='passed_scoped',actual_new_process=True,errors=errors,resources=resources(),source_sha256=source_files()))
    print('COMPATIBILITY',errors,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','check']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
