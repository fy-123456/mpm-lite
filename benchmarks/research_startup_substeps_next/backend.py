"""Explicit scoped backend selection from authenticated saved inputs; zero-step CLI."""
import argparse
from .provenance import *
from .fixture import inherited_setup,new_setup
from benchmarks.research_phase_stress_next.time_study import history
from engine.aniso_phase1.research_restoring_rt0_next.fixture import physical_payload

def build_selected(run,scope,which=0,backend='auto'):
    run=Path(run);verify(run)
    if scope not in ('legacy','boundary32') or which not in (0,1) or backend not in ('auto','A','B'):raise ValueError('unsupported scope/state/backend')
    decision=read(run/'S4/backend-decision.json');entry=decision['scopes'][scope]
    chosen=('B' if entry['selected'] else 'A') if backend=='auto' else backend
    if chosen=='B' and not (entry['selected'] and decision['shared_transaction']['rollback_exact'] and decision['restart']['same_digest']):raise ValueError('shared backend not qualified in requested scope')
    protocol=read(run/'S4'/scope/'integration-protocol.json');source=protocol['source_states'][which];path=Path(source['path'])
    if sha(path)!=source['sha256']:raise ValueError('source state changed')
    item=next(x for x in history(path.parents[2]) if x['folder']==path.parent)
    c,m,cfg,bridge=inherited_setup(run,item['state']) if scope=='legacy' else new_setup(run,state=item['state'])
    before=physical_payload(c.state)
    if chosen=='B':
        from engine.aniso_phase1.research_boundary_reference_next.geometry import install
        bridge=install(c,bridge)
    if physical_payload(c.state)!=before:raise ValueError('selection changed physical state')
    selection=dict(schema='startup-substeps-scoped-backend-v1',scope=scope,which=which,backend=chosen,source_sha256=source['sha256'],physical_payload_sha256=digest(before),coupling=c.identity,implementation_sha256=sha(__file__),decision_sha256=sha(run/'S4/backend-decision.json'),explicit_new_branch=True,new_steps=0)
    return c,m,cfg,bridge,selection

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--scope',choices=['legacy','boundary32'],default='boundary32');p.add_argument('--which',type=int,choices=[0,1],default=0);p.add_argument('--backend',choices=['auto','A','B'],default='auto');p.add_argument('--report',type=Path);a=p.parse_args()
    with serial_lock(a.run):
        c,m,cfg,bridge,selection=build_selected(a.run,a.scope,a.which,a.backend)
        if a.report:
            if (a.run/'release.json').exists():raise ValueError('sealed run read only')
            write(a.report,dict(status='passed_scoped',selection=selection,state_digest=c.state.digest(),bridge=bridge,zero_step=True))
        print('SELECTED',selection['scope'],selection['backend'],'time',c.state.time,'zero steps',flush=True)
