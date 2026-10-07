"""Load the qualified two-cell pressure backend without changing archived cases."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import read,write,sha,source_files,serial_lock
from .coupling_study import setup
from engine.aniso_phase1.research_observable_pressure_next.fast_rt0 import FastGridCoupling
from engine.aniso_phase1.research_observable_pressure_next.rt0 import BoundedGridCoupling
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish
from benchmarks.research_sequential_next.checkpoint import GenerationStore

def load_fixture(run,state=None,*,backend='selected'):
    run=Path(run)
    if backend not in ('selected','baseline'):raise ValueError('unknown pressure backend')
    fast=backend=='selected' and read(run/'S4/performance-decision.json')['pressure_fast_RT0']
    if fast and read(run/'S4/profile-candidate.json')['source_sha256']!=source_files():raise ValueError('pressure performance source changed')
    m,cfg=setup(run);times=read(run/'S3/pressure-time-protocol.json')['times_s'];source=read(run/'S3/fixed-2.json')['source_m3_s']
    c=(FastGridCoupling if fast else BoundedGridCoupling)(m,cfg,times,cells=2,source_m3_s=source,state=state)
    return c,m,cfg

def seed(run):
    run=Path(run);original=run/'cases/pressure-2';base=GenerationStore(original,read(original/'identity.json')).history()[0]['state']
    c,m,cfg=load_fixture(run,base);cache=CachedProbes(m);folder=run/'S4/runtime-check'
    if (folder/'identity.json').exists():raise ValueError('runtime check exists')
    original_identity=read(original/'identity.json')['coupling']
    if c.identity!=original_identity:raise ValueError('optimized backend changed physical identity')
    identity=dict(coupling=c.identity,numerical_source_sha256=source_files(),loader_sha256=sha(__file__))
    write(folder/'identity.json',identity);store=GenerationStore(folder,identity);store.save(c.state,[],frame=c.frame(cache));before=c.state.digest();pointer=read(store.pointer);checks=[]
    def precommit(stage,state):raise ValueError('controlled fast pressure precommit')
    def prepointer(stage):
        if stage=='before_pointer':raise OSError('controlled fast pressure pointer failure')
    for name,kwargs in [('before_commit',dict(inject_step=precommit)),('before_pointer',dict(inject_store=prepointer))]:
        try:advance_publish(c,store,[],**kwargs)
        except (ValueError,OSError):pass
        else:raise AssertionError('fault not observed')
        if c.state.digest()!=before or read(store.pointer)!=pointer:raise ValueError('fast pressure partial rollback')
        checks.append(dict(stage=name,complete_state_unchanged=True,pointer_unchanged=True))
    row=advance_publish(c,store,[],frame_builder=lambda st:c.frame(cache));errors={}
    with np.load(run/'S4/profiles/candidate-0/final.npz') as z:
        for key,name in [('q','q'),('velocity','v'),('predictor','predictor')]:errors[key]=float(np.max(abs(getattr(c.state,key)-z[name])))
    if max(errors.values())>1e-8 or row['true_scaled_residual']>1:raise ValueError('fast pressure recovery differs')
    write(run/'S4/runtime-seed.json',dict(status='passed_scoped',faults=checks,errors=errors,digest=c.state.digest(),physical_identity_unchanged=True,numerical_source_sha256=source_files(),loader_sha256=sha(__file__)))
    print('FAST_RUNTIME_SEEDED',errors,flush=True)

def reload(run):
    run=Path(run);folder=run/'S4/runtime-check';identity=read(folder/'identity.json')
    if identity['numerical_source_sha256']!=source_files() or identity['loader_sha256']!=sha(__file__):raise ValueError('runtime source drift')
    store=GenerationStore(folder,identity);saved=store.load();c,m,cfg=load_fixture(run,saved['state']);frame=c.frame(CachedProbes(m));errors={}
    with np.load(store.history()[-1]['folder']/'frame.npz') as z:
        for k in z.files:errors[k]=float(np.max(abs(z[k]-frame[k])))
    if c.state.digest()!=read(run/'S4/runtime-seed.json')['digest'] or max(errors.values())>1e-8:raise ValueError('reloaded optimized backend differs')
    write(run/'S4/runtime-check.json',dict(status='passed_scoped',actual_new_process=True,zero_step_reload=True,frame_errors=errors,source_sha256=source_files(),loader_sha256=sha(__file__),backend='FastGridCoupling',scope='two-cell small-storage seven-node intervals; measured paired states, no four-cell dynamics adoption',fallback_backend='baseline',transactions=read(run/'S4/runtime-seed.json')))
    print('FAST_RUNTIME_RELOAD_PASSED',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['seed','reload']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed evidence; load_fixture remains read-only')
        {'seed':seed,'reload':reload}[a.phase](a.run)
