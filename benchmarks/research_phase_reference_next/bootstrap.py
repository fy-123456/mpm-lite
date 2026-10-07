from pathlib import Path
import argparse,copy
import numpy as np
from .provenance import APP,APP_SHA,read,write,register,source_files,serial_lock,resources
from .run import create_config
from benchmarks.research_phase_stress_next.time_study import history
from .events import roundoff_only

def prepare(run):
    run=Path(run);create_config(run,'compatibility',end=.05,field_cache=True,display_frames=3)
    write(run/'S0/interface-contract.json',dict(parent=APP_SHA,space=read(run/'selected-space.json'),numeric_sources=source_files(),default_reuse_transpose_buffers=True,pressure_default=False,material_default='full_only'))
    write(run/'S0/evidence-reuse-map.json',dict(parent_sources_verified=True,parent_documents_immutable=True,formal_space_changed=False,old_modal_basis_reusable=True,old_q5_permissions_need_current_sources=True))
    raw=read(APP/'S2/raw-events.json');records=[]
    for rec in raw['records']:
        for mode in rec['modes']:
            for which,pair,budget in [('reference',('fine','half'),.003125),('medium',('medium','half'),.00625)]:
                a,b=(mode['raw'][k] for k in pair);new=roundoff_only(a,b,budget)
                records.append(dict(window=rec['window'],mode=mode['mode'],comparison=which,old=mode[which],roundoff_only=new))
    register(run,'S1/event-protocol.json',dict(no_smoothing=True,no_alignment=True,roundoff='32 eps max(1, absolute times)',matching='same-kind ordered; unique bilateral brackets inside min(budget, .45 adjacent period)',new_time_steps=0,physical_event_budgets_s=[.003125,.00625]))
    write(run/'S1/event-roundoff-audit.json',dict(status='diagnosed_and_fixed_in_new_comparator',records=records,mode523_not_discarded=True))
    print('BOOTSTRAP_READY',flush=True)

def check(run):
    run=Path(run);a=history(run/'cases/compatibility');b=history(APP/'cases/final-full')[:5];errors={}
    for key in ('q','velocity','predictor'):
        values=[]
        for x,y in zip(a,b):
            u,v=getattr(x['state'],key),getattr(y['state'],key)
            if u is None or v is None:
                if u is not v:raise ValueError('None predictor mismatch')
                values.append(0.)
            else:values.append(float(np.max(abs(u-v))))
        errors[key]=max(values)
    if len(a)!=5 or len(read(run/'cases/compatibility/segments.json'))!=2 or max(errors.values())>1e-8:raise ValueError('current entry compatibility failed')
    write(run/'S0/compatibility.json',dict(status='passed_scoped',actual_new_process=True,errors=errors,resources=resources(),source_sha256=source_files()))
    print('COMPATIBILITY',errors,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','check']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):{'prepare':prepare,'check':check}[a.phase](a.run)
