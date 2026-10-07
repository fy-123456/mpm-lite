"""Explicit numerical protocols; no resume across numerical identity changes."""
from __future__ import annotations
import copy
import math
from .provenance import digest


def time_grid(dt=0.05,start=0.,end=1.6):
    if not all(math.isfinite(x) for x in (dt,start,end)) or dt<=0 or not 0<=start<end<=1.6:
        raise ValueError('positive dt and ordered cycle interval required')
    breaks=[start,*[x for x in (.5,.6,1.1) if start<x<end],end]
    times=[float(start)]
    for a,b in zip(breaks[:-1],breaks[1:]):
        n=int(math.ceil((b-a)/dt-1e-10))
        times.extend(round(a+(b-a)*i/n,12) for i in range(1,n+1))
    return times


def protocol(energy_scale_J,*,dt=.05,start=0.,end=1.6,material_order=7,device='cuda:0',path_order=2,
             operator_mode=None,linearization_cache=False,cache_bytes=3<<30,cache_entries=4,
             preconditioner='original'):
    value=dict(schema='sequential-next-protocol-v1',seed=20260930,dtype='float64',
        times=time_grid(dt,start,end),material_order=material_order,mass_order=5,device=device,
        path_order=path_order,max_iters=8,display_frames=12,probe_shape=[33,7,7],
        solver=dict(residual_atol=1e-7,residual_force_atol_N=1e-5,residual_rtol=1e-5,ledger_atol_J=1e-7),
        acceptance=dict(min_detF=.1,boundary_atol=1e-8,energy_fraction=.01,energy_scale_J=energy_scale_J,
            field_rtol=.05,displacement_atol_m=5e-5,physical_velocity_atol_m_s=1e-4,
            PK1_atol_Pa=.02,reaction_atol_N=1e-4,backend_rtol=2e-5,material_rtol=.02,tangent_rtol=.03),
        resources=dict(max_rss_GiB=16,free_gpu_memory_fraction=.7,reference_soft_seconds=600,
            reference_hard_seconds=1200,min_system_free_GiB=5,max_diagnostic_hypotheses=3),
        physics=dict(original_Ks=True,all_mass_cross_terms=True,no_grid_transfer=True,no_damping=True,
                     no_artificial_mass=True,no_independent_APIC_microinertia=True),
        spatial_certified=False,temporal_certified=False,
        implementation=dict(operator=operator_mode or ('original' if device=='cpu' else 'segmented'),
            linearization_cache=linearization_cache,cache_bytes=cache_bytes,cache_entries=cache_entries,
            preconditioner=preconditioner))
    return validate(value)


def validate(value):
    value=copy.deepcopy(value)
    if value.get('schema')!='sequential-next-protocol-v1':raise ValueError('unknown protocol schema')
    times=value['times']
    if len(times)<2 or not all(math.isfinite(t) for t in times) or times[0]<0 or times[-1]>1.6:
        raise ValueError('invalid cycle time range')
    if any(b<=a for a,b in zip(times[:-1],times[1:])):raise ValueError('time grid must increase')
    for point in (.5,.6,1.1):
        if times[0]<point<times[-1] and not any(abs(t-point)<1e-12 for t in times):
            raise ValueError('time grid misses a loading breakpoint')
    scenario=value.get('scenario',{})
    angle=scenario.get('fiber_angle_degrees')
    if angle is not None and (not math.isfinite(angle) or not 0<=angle<180):raise ValueError('fiber angle must be in [0,180) degrees')
    peak=scenario.get('peak_m',.005)
    if not math.isfinite(peak) or peak<=0:raise ValueError('positive finite peak displacement required')
    for key in ('material_order','path_order','max_iters','display_frames'):
        if isinstance(value[key],bool) or not isinstance(value[key],int) or value[key]<1:raise ValueError('invalid '+key)
    if isinstance(value['mass_order'],bool) or not isinstance(value['mass_order'],int) or value['mass_order']<1 or value['dtype']!='float64':raise ValueError('positive qualified mass order and float64 required')
    if value['device']!='cpu' and not value['device'].startswith('cuda:'):raise ValueError('unsupported device')
    # Absent implementation preserves the original algorithm and content hash.
    implementation=value.get('implementation',{})
    mode=implementation.get('operator','original')
    if mode not in ('original','segmented'):raise ValueError('unknown material operator implementation')
    cache=implementation.get('linearization_cache',False)
    if not isinstance(cache,bool):raise ValueError('cache switch must be boolean')
    if mode=='segmented' and value['device']=='cpu':raise ValueError('segmented operator requires CUDA')
    if cache and mode!='segmented':raise ValueError('GPU linearization cache requires segmented operator')
    for key,default in [('cache_bytes',3<<30),('cache_entries',4)]:
        n=implementation.get(key,default)
        if isinstance(n,bool) or not isinstance(n,int) or n<1:raise ValueError('positive integer '+key+' required')
    if implementation.get('preconditioner','original') not in ('original','reuse_static'):
        raise ValueError('unknown AVF preconditioner')
    if any(not math.isfinite(x) or x<=0 for x in value['solver'].values()):raise ValueError('invalid solver tolerances')
    if any(not math.isfinite(x) or x<=0 for x in value['acceptance'].values()):raise ValueError('invalid acceptance budgets')
    if len(value['probe_shape'])!=3 or any(n<2 or not isinstance(n,int) for n in value['probe_shape']):raise ValueError('invalid probe shape')
    return value


def identity(value):
    return digest(validate(value))
