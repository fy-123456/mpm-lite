"""Explicit fixed schedules; no damping or implicit protocol migration."""
import copy
import math
from benchmarks.research_sequential_next.config import protocol as parent_protocol, validate as parent_validate, time_grid
from .provenance import digest


def validate(cfg):
    cfg = parent_validate(cfg)
    scenario=cfg.get('scenario',{})
    if scenario.get('fiber_angle_degrees') not in (None,45.) or scenario.get('peak_m',.005) not in (.005,.0075):
        raise ValueError('this entry supports F45 and explicitly adapted .005/.0075m peaks only')
    implementation=cfg.get('implementation',{})
    if cfg['device']=='cpu' or implementation.get('operator')!='segmented' or implementation.get('linearization_cache',False) or implementation.get('preconditioner','original')!='original':
        raise ValueError('post-release runner currently supports CUDA segmented, no tangent cache, original preconditioner')
    if not isinstance(implementation.get('force_only_responses',False),bool):raise ValueError('force-only switch must be boolean')
    extension = cfg.get('post_release', {})
    if extension.get('schema') != 'post-release-v1': raise ValueError('missing post-release protocol')
    if not isinstance(extension.get('field_cache',False),bool):raise ValueError('field_cache must be boolean')
    policy=extension.get('rule_policy')
    if policy not in ('full_only','q5_fixed','q5_with_full_retry'):raise ValueError('unsupported rule policy')
    if cfg['material_order']!=(7 if policy=='full_only' else 5):raise ValueError('policy/material rule mismatch')
    if policy!='full_only' and not extension.get('qualification'):raise ValueError('q5 requires bound qualification')
    if cfg['times'][0] != 0 and 'initial_state' not in cfg:
        raise ValueError('nonzero start requires authenticated committed state')
    return cfg


def make(scale, *, dt=.0125, start=0., end=1.6, times=None, initial=None, probe_shape=(33,7,7), display_frames=12,rule_policy='full_only',qualification=None,field_cache=False,peak_m=.005,force_only=False):
    cfg = parent_protocol(scale, dt=dt, start=start, end=end, material_order=7, device='cuda:0',
                          operator_mode='segmented')
    if times is not None:
        times = [float(t) for t in times]
        if len(times)<2 or abs(times[0]-start)>1e-12 or abs(times[-1]-end)>1e-12:
            raise ValueError('explicit time grid must match start/end')
        cfg['times'] = times
    cfg['probe_shape']=list(probe_shape);cfg['display_frames']=display_frames
    cfg['scenario']=dict(fiber_angle_degrees=45.,peak_m=peak_m)
    cfg['implementation']['force_only_responses']=force_only
    cfg['post_release']=dict(schema='post-release-v1',rule_policy=rule_policy,schedule='explicit-frozen-times',field_cache=field_cache)
    if qualification is not None:cfg['post_release']['qualification']=copy.deepcopy(qualification)
    cfg['material_order']=7 if rule_policy=='full_only' else 5
    if initial is not None: cfg['initial_state']=copy.deepcopy(initial)
    return validate(cfg)


def identity(cfg): return digest(validate(cfg))
