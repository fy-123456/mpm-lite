"""Read and validate a saved solid checkpoint, without advancing or rewriting it."""
from pathlib import Path
import argparse
from .provenance import read,sha,source_files,digest,serial_lock
from .run import load_model,make_stepper
from .publication import audit_release
from benchmarks.research_sequential_next.checkpoint import GenerationStore

def inspect(run,case=None):
    run=Path(run);sealed=(run/'release.json').exists()
    if sealed:audit_release(run)
    if case is None:case=read(run/'S6/final-protocol.json')['default_case']
    if Path(case).name!=case or case in ('.','..'):raise ValueError('single case name required')
    folder=run/'cases'/case;cfg=read(folder/'execution-protocol.json');identity=read(folder/'identity.json')
    if identity['numerical_source_sha256']!=source_files():raise ValueError('saved numerical source differs')
    m,_=load_model(run,cfg)
    if identity['model']!=m.identity:raise ValueError('saved model differs')
    loaded=GenerationStore(folder,identity).load()
    if loaded is None:raise ValueError('no committed generation')
    stepper=make_stepper(run,cfg,m,loaded['state']);stepper.validate(loaded['state'])
    return dict(case=case,accepted_rows=len(loaded['rows']),state_step=stepper.state.step,time_s=stepper.state.time,state_digest=stepper.state.digest(),active_material=stepper.state.child_states['identity']['material'],rule_policy=cfg['post_release']['rule_policy'],mass_sha256=digest(m.M.tolist()),zero_step=True,sealed=sealed,in_place_resume_allowed=not sealed)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--case');a=p.parse_args()
    with serial_lock(a.run):print(inspect(a.run,a.case),flush=True)
