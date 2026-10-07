"""Frozen three transverse cases and one explicit legacy bridge."""
from pathlib import Path
import json
from .provenance import read,all_sources
from .physics import baseline_model
from engine.aniso_phase1.research_pressure3d_next.coupled import BlockCoupling
from engine.aniso_phase1.research_transverse_next.initial import InitialCoupling,pressure_profile,CASES
from engine.aniso_phase1.research_transverse_next.geometry import ProfiledGeometry
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology

def numerical_sources():
    names={'__init__.py','provenance.py','base_config.py','config.py','spaces.py','physics.py','fixture.py','runtime.py','trajectory.py','observables.py'}
    engines={'__init__.py','initial.py','geometry.py','recovery.py'}
    return {k:v for k,v in all_sources().items() if (k.startswith('engine/') and Path(k).name in engines) or (k.startswith('benchmarks/') and Path(k).name in names)}

def setup(run,case='Y64',fine=False,profile=True):
    run=Path(run);protocol=read(run/'S0/coupled-protocol.json');grid=CASES[case][0]
    cuts=read(run/'S0/grid-protocol-inherited.json')['cuts'][grid];v=protocol['parameters']
    p,definition=pressure_profile(case,ReferenceTopology(cuts));times=protocol['times']['h']
    if fine:times=read(run/'S3/time-protocol.json')['full_times_s']
    m,cfg=baseline_model(run,pressure=True)
    inner=BlockCoupling(m,cfg,times,cuts=cuts,method='startup',source_m3_s=0.,alpha=v['alpha'],storage=v['storage'],pressure0=p,reservoir=v['reservoir_Pa'],mobility=v['mobility_scale']*MOBILITY)
    if profile:inner.core.geometry=ProfiledGeometry.adopt(inner.geometry)
    c=InitialCoupling(inner,definition)
    ident=dict(schema='transverse-branch-v1',coupling=c.identity,case_id=case,grid=grid,fine=fine,initial_digest=c.initial_digest,numerical_sources=numerical_sources())
    return c,m,cfg,json.loads(json.dumps(ident,allow_nan=False))
