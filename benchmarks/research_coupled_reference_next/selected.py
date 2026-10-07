"""Actually load the qualified backend; scoped performance adoption is explicit."""
import argparse
from .provenance import *
from .fixture import setup
from engine.aniso_phase1.research_coupled_reference_next.factored import FactoredGradientGeometry
from benchmarks.research_sequential_next.checkpoint import GenerationStore

def selected_setup(run):
    run=Path(run);verify(run);d=read(run/'S3/backend-decision.json');c,m,cfg,ident=setup(run)
    if d['selected']:
        if read(run/'S3/continuous-check.json')['status']!='passed_scoped' or not read(run/'S3/paired-performance.json')['candidate_eligible']:raise ValueError('candidate qualification missing')
        c.core.geometry=FactoredGradientGeometry.adopt(c.geometry)
    imp=dict(c.geometry.implementation);imp.pop('context',None)
    ident.update(backend=d['backend'],actual_backend_implementation=imp,numerical_sources=numerical_sources())
    return c,m,cfg,ident

def load(run):
    run=Path(run);mutable(run);c,m,cfg,ident=selected_setup(run)
    folder=run/'S3/continuous' if read(run/'S3/backend-decision.json')['selected'] else run/'S2/bridge'
    prior=read(folder/'identity.json');check(ROOT,prior['numerical_sources'])
    if prior['coupling']!=ident['coupling']:raise ValueError('selected physical identity differs')
    store=GenerationStore(folder,prior);store.history();record=store.load(validator=c.validate);c.restore(record['state'])
    write(run/'S5/selected-load-check.json',dict(status='passed_scoped',fresh_process=True,zero_steps=True,actual_backend=type(c.geometry).__name__,interface='benchmarks.research_coupled_reference_next.selected.selected_setup',source=str(folder),source_identity_sha256=sha(folder/'identity.json'),state_digest=c.state.digest(),step=c.state.step,execution_identity=ident))
    print('SELECTED_LOAD',type(c.geometry).__name__,c.state.step,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();load(a.run)
