"""Loadable selected research backend; capability gates precede candidate adoption."""
import argparse
from .provenance import *
from .fixture import setup
from engine.aniso_phase1.research_transverse_reference_next.batch_download import BatchDownloadGeometry
from benchmarks.research_sequential_next.checkpoint import GenerationStore


def selected_setup(run):
    run=Path(run);verify(run);d=read(run/'S3/backend-decision.json')
    c,m,cfg,ident=setup(run)
    if d['selected']:
        if read(run/'S3/continuous-check.json')['status']!='passed_scoped' or not read(run/'S3/paired-performance.json')['selected']:
            raise ValueError('selected backend missing qualification')
        check(ROOT,read(run/'S3/static/identity.json')['numerical_sources'])
        c.core.geometry=BatchDownloadGeometry.adopt(c.geometry)
    implementation=dict(c.geometry.implementation);implementation.pop('context',None)
    ident.update(backend=d['backend'],actual_backend_implementation=implementation,numerical_sources=all_sources())
    return c,m,cfg,ident


def check_load(run):
    run=Path(run);mutable(run);c,m,cfg,ident=selected_setup(run)
    folder=run/'S3/continuous' if read(run/'S3/backend-decision.json')['selected'] else run/'S2/bridge'
    prior=read(folder/'identity.json');check(ROOT,prior['numerical_sources']);store=GenerationStore(folder,prior);store.history();record=store.load(validator=c.validate);c.restore(record['state'])
    # Different execution wrapper, same authenticated physical state and equations.
    if prior['coupling']!=ident['coupling'] or c.state.digest()!=record['state'].digest():raise ValueError('selected loader changes physical payload')
    write(run/'S5/selected-load-check.json',dict(status='passed_scoped',fresh_process=True,zero_steps=True,actual_backend=type(c.geometry).__name__,interface='benchmarks.research_transverse_reference_next.selected.selected_setup',source=str(folder),source_identity_sha256=sha(folder/'identity.json'),state_digest=c.state.digest(),step=c.state.step,execution_identity=ident))
    print('SELECTED_LOAD',type(c.geometry).__name__,c.state.step,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();check_load(a.run)
