"""Only registered four pressure grids; unchanged solid and source."""
from pathlib import Path
import json
from .provenance import read, all_sources
from .physics import baseline_model
from engine.aniso_phase1.research_pressure3d_next.coupled import BlockCoupling
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY

def numerical_sources():
    names={'__init__.py','provenance.py','base_config.py','config.py','spaces.py','physics.py','fixture.py','runtime.py','trajectory.py'}
    return {k:v for k,v in all_sources().items() if k.startswith('engine/') and Path(k).name in {'__init__.py','geometry.py','coupled.py'} or k.startswith('benchmarks/') and Path(k).name in names}

def setup(run,grid='base',fine=False):
    run=Path(run);p=read(run/'S0/coupled-protocol.json');cuts=read(run/'S0/grid-protocol-inherited.json')['cuts'][grid];v=p['parameters']
    m,cfg=baseline_model(run,pressure=True)
    c=BlockCoupling(m,cfg,p['times']['half' if fine else 'h'],cuts=cuts,method='startup',source_m3_s=0.,alpha=v['alpha'],storage=v['storage'],pressure0=v['pressure0_Pa'],reservoir=v['reservoir_Pa'],mobility=v['mobility_scale']*MOBILITY)
    ident=dict(schema='pressure3d-branch-v1',coupling=c.identity,grid=grid,fine=fine,numerical_sources=numerical_sources())
    # GenerationStore compares an in-memory identity with its JSON round trip.
    # Canonicalize containers; this does not alter numerical values or digests.
    return c,m,cfg,json.loads(json.dumps(ident,allow_nan=False))
