"""Run identity, practical predeclared gates and isolated resource accounting."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import numpy as np
import scipy
from engine.aniso_phase1.research_d.identity import ROOT,BASELINE,baseline_audit,own_sources,write_json

# User-authorized practical equivalence, not continuum/space certification.
GATES=dict(operator_relative=1e-7,operator_absolute=1e-10,symmetry_relative=1e-8,
           energy_fd_relative=1e-4,tangent_fd_relative=1e-3,
           field_relative=.005,field_absolute=1e-8,reaction_absolute_N=1e-8,
           constraint_absolute=1e-10,linear_rtol=[1e-6,1e-8],linear_atol=1e-12,
           work_absolute_J=1e-8,minimum_det_F=0.,speedup_candidate=.10,regression_limit=.03)
CASES=[dict(name='F45-small-q2',cells=[4,2,2],p=2,angle=45.,kf=200.),
       dict(name='F0-small-q2',cells=[4,2,2],p=2,angle=0.,kf=200.),
       dict(name='F90-small-q3',cells=[4,2,2],p=3,angle=90.,kf=200.),
       dict(name='F45-local-q4',cells=[2,2,2],p=4,angle=45.,kf=200.),
       dict(name='F30-stiff-q2',cells=[4,2,2],p=2,angle=30.,kf=2000.),
       dict(name='F45-medium-q2',cells=[12,6,6],p=2,angle=45.,kf=200.),
       dict(name='F45-large-q2',cells=[24,12,12],p=2,angle=45.,kf=200.)]


def command(args):
    try: return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()
    except (OSError,subprocess.CalledProcessError) as e: return str(e)


def environment():
    return dict(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),python=platform.python_version(),
        numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),
        affinity=sorted(os.sched_getaffinity(0)),load_average=list(os.getloadavg()),
        threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        gpu=command(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader']),
        gpu_processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']),
        system_free_bytes=shutil.disk_usage('/').free,data_free_bytes=shutil.disk_usage('/root/autodl-tmp').free,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        cpu_model=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),''))


def freeze(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    audit=baseline_audit(); write_json(out/'baseline-before.json',audit)
    if not audit['passed']: raise RuntimeError('v22 identity changed; explicit baseline update required')
    write_json(out/'protocol.json',dict(schema_version=1,baseline_sha256=BASELINE,producer='D',
        source_sha256=own_sources(),environment=environment(),seed=20260930,gates=GATES,cases=CASES,
        dtype='float64',q_convention='displacement for tensor; existing production velocity for SparseProbe',
        scope='same frozen discrete operators; no spatial certification, dynamic-space upgrade or default switch',
        physical_geometry=[[.125,.375,.375],[.875,.625,.625]],free_span=[.25,.75],displacement_m=.005,
        material=dict(mu=10.,lam=20.,kf=200.,controls='kf/angle variations explicitly named'),
        region='all free-span quadrature and both clamps; no excluded corners',
        performance='3 independent processes; each AB then BA, full solve + construction; warm reuse separately',
        resource_limits=dict(gpu_workspace_bytes=1<<30,local_batch_bytes=32<<20,local_total_bytes=256<<20,
                             root_low_bytes=5*1024**3),
        note='Relaxed practical tolerances explicitly requested by user; fixed before formal acceptance.'))


def problem(case):
    from engine.aniso_phase1.research_d.tensor import FrozenTensor
    from engine.aniso_phase1.beam_reference import reference_hessian
    e=[np.linspace(lo,hi,n+1) for lo,hi,n in zip((.25,.375,.375),(.75,.625,.625),case['cells'])]
    a=np.deg2rad(case['angle']); H=reference_hessian(kf=case['kf'],direction=(np.cos(a),np.sin(a),0.))
    return FrozenTensor(e,case['p'],H,name=case['name'])


def storage_check(cache):
    """Only relocate D-owned, regenerable cache when root drops below 5 GiB."""
    free=shutil.disk_usage('/').free
    record=dict(system_free_bytes=free,threshold_bytes=5*1024**3,migrated=False)
    if free>=5*1024**3: return str(cache),record
    data=Path('/root/autodl-tmp/mpm-lite-research-d')
    if os.stat(data.parent).st_dev==os.stat('/').st_dev: raise RuntimeError('data path is not a separate disk')
    data.mkdir(exist_ok=True); target=data/Path(cache).name
    cache=Path(cache)
    if cache.exists() and not target.exists(): shutil.move(str(cache),str(target)); record['migrated']=True
    target.mkdir(exist_ok=True)
    record.update(cache=str(target),system_free_after_bytes=shutil.disk_usage('/').free)
    return str(target),record
