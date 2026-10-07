"""Explicit DV factory: physical identity inherited, execution identity separate."""
import json
from pathlib import Path
from .provenance import verify, read, all_sources
from benchmarks.research_transverse_next.spaces import load_selected
from benchmarks.research_transverse_next import config
from engine.aniso_phase1.research_pressure3d_next.coupled import BlockCoupling
from engine.aniso_phase1.research_transverse_next.initial import InitialCoupling, pressure_profile
from engine.aniso_phase1.research_transverse_next.device_volume import DeviceVolumeGeometry
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology


def setup(run, *, fine=False):
    import warp as wp
    import numpy as np
    from engine.aniso_phase1.research_phase_boundary_next.segments import VectorizedModel
    from engine.aniso_phase1.research_phase_stress_next.warm import install_warm
    run=Path(run);lock=verify(run);p=read(run/'S0/physical-contract.json')
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    choice=read(run/'selected-space.json');reduction,_=load_selected(choice['package'])
    cfg=config.make(lock['energy_scale_J'],end=.05,space=choice['package'],mass_order=7,full_order=7)
    m=install_warm(VectorizedModel(reduction,order=7,device='cuda:0',hold=0.))
    pressure,definition=pressure_profile('YZ128',ReferenceTopology(p['cuts']))
    if pressure.tolist()!=p['pressure0']:raise ValueError('frozen initial pressure differs')
    times=p['times']['h']
    if fine:
        times=read(run/'S2/time-protocol.json')['full_times_s']
    v=p['parameters']
    inner=BlockCoupling(m,cfg,times,cuts=p['cuts'],method='startup',source_m3_s=0.,alpha=v['alpha'],storage=v['storage'],pressure0=pressure,reservoir=v['reservoir_Pa'],mobility=np.array(p['mobility']))
    inner.core.geometry=DeviceVolumeGeometry.adopt(inner.geometry)
    c=InitialCoupling(inner,definition)
    sources={k:v for k,v in all_sources().items() if Path(k).name in ('__init__.py','provenance.py','fixture.py','runtime.py','coupling.py')}
    ident=dict(schema='transverse-reference-DV-v1',coupling=c.identity,case_id='YZ128',fine=fine,initial_digest=c.initial_digest,backend='DV',backend_implementation=c.geometry.implementation,numerical_sources=sources)
    # CUDA context pointers are runtime ownership, not persistent physical identity.
    ident['backend_implementation']=dict(ident['backend_implementation']);ident['backend_implementation'].pop('context',None)
    return c,m,cfg,json.loads(json.dumps(ident,allow_nan=False))
