"""Pre-register scope, tolerances and a bounded resource envelope before evaluation."""
import json
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_d.stage2.contracts import PARENT_SHA256,SPACE_SHA256,array_digest

PROTOCOL=dict(schema_version=2,parent_bundle_sha256=PARENT_SHA256,space_sha256=SPACE_SHA256,
 seed=20260930,scope='fixed common space, static finite-strain Hencky plus bilateral fiber and original Ks',
 dtype='float64',material_orders=[6,7],mass_order=5,density_kg_m3=1.,
 units=dict(length='m',energy='J',gradient='N',tangent='N/m',PK1='Pa'),
 tolerances=dict(operator=dict(rtol=1e-7,atol=1e-10,scale=1.),adjoint=dict(rtol=1e-10,atol=1e-10,scale=1.),
 energy_derivative=dict(rtol=1e-4,atol=1e-9,scale=1e-3),tangent_derivative=dict(rtol=1e-3,atol=1e-8,scale=1.),
 physical_field=dict(rtol=.005,atol=1e-8,scale=1e-3),residuals=[1e-6,1e-8],residual_scale_N=1.),
 fd_steps=[1e-4,3e-5,1e-5],minimum_singular_value=1e-6,
 curvature='full symmetric spectrum before PCG; original Hessian never projected',
 preconditioners=['none','diagonal','carrier_local_block','overlap_block'],
 resources=dict(cpu_threads=4,one_gpu=True,max_response_seconds=300,max_solve_seconds=1800,max_newton=30,max_linear=800,max_line_search=24,max_host_GiB=24,max_device_GiB=16),
 performance=dict(processes=3,orders=['AB','BA','AB'],minimum_gain=.1,maximum_other_regression=.03,requires_idle_gpu=True),
 selection='archived + parent perturbation; scaled fixed perturbation 3x is new-state integration control; no held-out/generalization claim',
 stopping='invalid detF is rejected; failed trials roll back; resource timeout records incomplete, never convergence',
 capabilities_not_granted=['dynamic_cycle','coupled_physics','continuum_spatial_accuracy','production_default'])

def freeze(folder,parent,space):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with np.load(Path(parent)/'input-states.npz') as z: arrays={k:z[k].copy() for k in z.files}
    a=arrays['state_archived_static'];b=arrays['state_fixed_perturbation_1e-4_m']
    arrays['state_perturbation_3e-4_m']=a+3*(b-a)
    rng=np.random.default_rng(PROTOCOL['seed']);d=rng.normal(size=space.q_shape);d/=np.linalg.norm(d)
    arrays['direction_mixed']=space.direction_coefficients(d)
    p=dict(PROTOCOL,vectors={k:array_digest(v) for k,v in arrays.items()})
    path=folder/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=p:raise ValueError('protocol already frozen differently')
    else:
        np.savez(folder/'vectors.npz',**arrays);path.write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')
    return p,arrays
