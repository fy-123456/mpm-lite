"""Authenticated fixed BASELINE fixture for independent time/pressure research."""
from pathlib import Path
from .provenance import read,verify
from .spaces import load_selected
from . import config

def baseline_model(run,*,pressure=False):
    import warp as wp
    from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
    from engine.aniso_phase1.research_local_span_next.reuse import ReusedSegmentedModel
    run=Path(run);lock=verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    choice=read(run/'baseline-space.json');r,_=load_selected(choice['package'])
    cfg=config.make(lock['energy_scale_J'],end=.05,space=choice['package'],mass_order=7,full_order=7,reuse_transpose_buffers=not pressure)
    m=(SegmentedModel if pressure else ReusedSegmentedModel)(r,order=7,device='cuda:0',hold=0. if pressure else None)
    return m,cfg
