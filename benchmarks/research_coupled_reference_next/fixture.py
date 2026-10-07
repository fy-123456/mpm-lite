"""BD factory with child-owned caches and identical physical equations."""
import json
import numpy as np
from .provenance import verify,read,numerical_sources,Path,APP,check,ROOT
from benchmarks.research_transverse_next.spaces import load_selected
from benchmarks.research_transverse_next import config
from engine.aniso_phase1.research_pressure3d_next.coupled import BlockCoupling
from engine.aniso_phase1.research_transverse_next.initial import InitialCoupling,pressure_profile
from engine.aniso_phase1.research_transverse_reference_next.batch_download import BatchDownloadGeometry
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology
from benchmarks.research_phase_stress_next.time_study import history

def setup(run):
    import warp as wp
    from engine.aniso_phase1.research_phase_boundary_next.segments import VectorizedModel
    from engine.aniso_phase1.research_phase_stress_next.warm import install_warm
    run=Path(run);lock=verify(run);p=read(run/'S0/physical-contract.json')
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    choice=read(run/'selected-space.json');reduction,_=load_selected(choice['package'])
    cfg=config.make(lock['energy_scale_J'],end=.05,space=choice['package'],mass_order=7,full_order=7)
    m=install_warm(VectorizedModel(reduction,order=7,device='cuda:0',hold=0.))
    top=ReferenceTopology(p['cuts']);pressure,definition=pressure_profile('YZ128',top)
    if top.cells!=128 or pressure.tolist()!=p['pressure0']:raise ValueError('wrong actual pressure fixture')
    v=p['parameters'];inner=BlockCoupling(m,cfg,p['times']['h'],cuts=p['cuts'],method='startup',source_m3_s=0.,alpha=v['alpha'],storage=v['storage'],pressure0=pressure,reservoir=v['reservoir_Pa'],mobility=np.array(p['mobility']))
    inner.core.geometry=BatchDownloadGeometry.adopt(inner.geometry);c=InitialCoupling(inner,definition)
    imp=dict(c.geometry.implementation);imp.pop('context',None)
    ident=dict(schema='coupled-reference-BD-v1',coupling=c.identity,case_id='YZ128',initial_digest=c.initial_digest,backend='BD',backend_implementation=imp,numerical_sources=numerical_sources())
    ident=json.loads(json.dumps(ident,allow_nan=False))
    if ident['coupling']!=read(APP/'S3/continuous/identity.json')['coupling']:raise ValueError('new adapter changed physical identity')
    return c,m,cfg,ident

def inputs():
    # Full prior history is authenticated by audit_parent; newest overlapping BD
    # generations replace the corresponding input states, without inventing paths.
    parent=Path(read(APP/'S0/checkpoint-sources.json')['records'][0]['path']).parent.parent
    check(ROOT,read(parent/'identity.json')['numerical_sources'])
    h={x['state'].step:x for x in history(parent)}
    check(ROOT,read(APP/'S3/continuous/identity.json')['numerical_sources'])
    for x in history(APP/'S3/continuous'):h[x['state'].step]=x
    return h
