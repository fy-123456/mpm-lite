"""N06 loadable release of the selected physical model and effective protocol."""
from pathlib import Path
import argparse
import shutil
import time
import numpy as np
from .provenance import PREVIOUS,ROOT,read,write,sha,serial_lock,source_files,verify,utc
from .checkpoint import GenerationStore


def load_reduction(run):
    from benchmarks.research_sequential.run import open_reduction
    run=Path(run);package=read(run/'model/model-package.json')
    if sha(run/'model/model-package.json')!=read(run/'N06/result.json')['model_package_sha256']:
        raise ValueError('changed common model package')
    if sha(run/'baseline-lock.json')!=package['baseline_sha256']:raise ValueError('common model baseline mismatch')
    for name,expected in package['files'].items():
        if sha(run/'model'/name)!=expected:raise ValueError('common model data changed')
    if sha(run/package['effective_protocol_path'])!=package['effective_protocol_sha256']:
        raise ValueError('effective protocol changed')
    if package['selected_space']!='original144':raise ValueError('unsupported selected physical space')
    reduction=open_reduction(PREVIOUS)
    if reduction.signature!=package['reduction_sha256']:raise ValueError('physical reduction changed')
    with np.load(run/'model/coordinates.npz',allow_pickle=False) as z:
        for key in ('P','offset','M','K','free','fixed'):
            if not np.array_equal(z[key],getattr(reduction,key)):raise ValueError('common coordinate data differ from actual model: '+key)
    return reduction


def publish(run):
    from .run import load_model
    from engine.aniso_phase1.research_sequential_next.model import PracticalModel
    from engine.aniso_phase1.research_d.common_kinetic import PointInertia
    run=Path(run);verify(run)
    decision=read(run/'N05/decision.json')
    if decision['selected']!='original144':raise ValueError('new space must complete dynamic qualification before publication')
    cfg=read(run/'cases/gpu-q7-dt0025/execution-protocol.json');gpu,_=load_model(run,cfg);r=gpu.reduction
    case=run/'cases/gpu-q7-dt0025';history=GenerationStore(case,read(case/'identity.json')).history()
    q=next(x['state'].q for x in history if abs(x['state'].time-.5)<1e-12)
    rng=np.random.default_rng(20260930);d=rng.normal(size=q.shape);d[gpu.fixed]=0.;d/=np.linalg.norm(d)
    begun=time.perf_counter();cpu=PracticalModel(r,order=7,device='cpu');a=cpu.evaluate(q,d);cpu_seconds=time.perf_counter()-begun
    begun=time.perf_counter();b=gpu.evaluate(q,d);gpu_seconds=time.perf_counter()-begun
    errors={key:float(np.linalg.norm(np.asarray(a[key])-b[key])/max(np.linalg.norm(a[key]),1e-12)) for key in ['U','force','tangent_action']}
    epsilon=1e-5;plus=gpu.evaluate(q+epsilon*d);minus=gpu.evaluate(q-epsilon*d)
    fd=(plus['force']-minus['force'])/(2*epsilon)
    derivative=float(np.linalg.norm(fd-b['tangent_action'])/np.linalg.norm(b['tangent_action']))
    grad=abs((plus['U']-minus['U'])/(2*epsilon)-float(np.sum(b['force']*d)))
    full_d=r.velocity(d)
    independent_mass=PointInertia(r.parent,order=6).apply(full_d)
    mass=float(np.linalg.norm(r.original_mass@full_d-independent_mass)/np.linalg.norm(independent_mass))
    if max(errors.values())>2e-5 or derivative>.002 or grad>1e-7 or mass>1e-8:
        raise ValueError(dict(backend=errors,derivative=derivative,energy_derivative=grad,mass=mass))
    folder=run/'model';folder.mkdir(exist_ok=True)
    if (folder/'model-package.json').exists():raise ValueError('do not replace a published physical model')
    shutil.copy2(PREVIOUS/'coordinates.npz',folder/'coordinates.npz')
    protocol_path='cases/gpu-q7-dt0025/execution-protocol.json'
    old=read(PREVIOUS/'common-model-package.json')
    package=dict(schema='sequential-next-common-model-v1',utc=utc(),selected_space='original144',
        parent_bundle_sha256=old['parent_sha256'],parent_space_sha256=old['parent_space_sha256'],
        baseline_sha256=sha(run/'baseline-lock.json'),reduction_sha256=r.signature,
        files={'coordinates.npz':sha(folder/'coordinates.npz')},effective_protocol_path=protocol_path,
        effective_protocol_sha256=sha(run/protocol_path),validating_source_sha256=source_files(),
        q_shape=list(q.shape),full_shape=[r.parent.ndof,3],free_shape=[len(r.free),3],
        displacement='q_full=P@w+offset',velocity='v_full=P@wdot',force='P.T @ positive potential gradient',
        tangent='P.T @ H @ P; quadrature weights applied once',mass='original complete M5 projected by P; all cross terms',
        probes=dict(shape=cfg['probe_shape'],order='Cartesian C order, xyz last',stress='PK1, Pa'),
        units=dict(length='m',time='s',stress='Pa',energy='J'),
        capabilities=dict(scene_stable=True,material_bounded=True,CPU_GPU_static=True,
            temporal_accuracy=False,spatial_accuracy=False,physical_3D_poroelasticity=False),
        source_policy='This package binds mathematics and its validation lineage. Every numerical case separately binds its actual code and solver protocol.')
    write(folder/'model-package.json',package)
    result=dict(status='passed_scoped',utc=utc(),model_package_sha256=sha(folder/'model-package.json'),
        selected='original144',reason='Two new budget144 candidates did not exceed the frozen reference-uncertainty promotion margin.',
        material_and_derivative=dict(CPU_GPU_relative=errors,directional_tangent_relative=derivative,energy_directional_absolute_J=grad),
        mass5_vs6_action_relative=mass,original_mass_unchanged=True,reduction_sha256=r.signature,
        CPU_same_state_seconds=cpu_seconds,GPU_same_state_seconds=gpu_seconds,
        rest_checks='reuse unchanged sealed S02-S03-algebra and S03-derivatives; all source/input hashes verified',
        new_space_time_requalification='not applicable: original144 retained; N03 real 32/64-step evidence applies',
        spatial_accuracy=False,temporal_accuracy=False)
    write(run/'N06/result.json',result)
    loaded=load_reduction(run)
    if loaded.signature!=r.signature:raise ValueError('published model failed actual reload')
    print(result,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):publish(a.run)
