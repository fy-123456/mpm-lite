"""Recheck newest publication and freeze contracts before any new dynamics."""
import os
import numpy as np
from .provenance import *
from engine.aniso_phase1.research_transverse_next.initial import pressure_profile
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology

def main():
    pubs=[]
    for base,dirs,files in os.walk(ROOT/'docs/results',followlinks=True):
        dirs[:]=[d for d in dirs if d not in ('final-source','warp-cache','generations','source') and not d.endswith('-source')]
        if 'release.json' in files:
            p=Path(base)/'release.json';v=read(p)
            if isinstance(v,dict) and v.get('utc'):pubs.append(dict(path=str(p.parent),utc=v['utc'],sha256=sha(p),realpath=str(p.parent.resolve())))
    pubs.sort(key=lambda v:v['utc'])
    if pubs[-1]['path']!=str(APP) or pubs[-1]['sha256']!=APP_SHA:raise ValueError('newer release requires review')
    run=freeze()
    write(run/'S0/preimplementation-version-audit.json',dict(status='passed_scoped',publications=pubs,latest=pubs[-1],git='no .git; publication SHA is not a commit',prior_full_audit=read(Path('/tmp/transverse-base-audit.json'))['counts']))
    for name in ('coupled-protocol.json','grid-protocol-inherited.json'):
        (run/'S0'/name).write_bytes((APP/'S0'/name).read_bytes())
    protocol=read(run/'S0/coupled-protocol.json');cuts=read(run/'S0/grid-protocol-inherited.json')['cuts'];v=protocol['parameters'];cases={}
    for case,grid in [('Y64','y'),('Y128','yz'),('YZ128','yz')]:
        t=ReferenceTopology(cuts[grid]);p,d=pressure_profile(case,t);C=v['storage']*t.V0;delta=p-.2
        cases[case]=dict(status='registered_not_executed',definition=d,pressure_Pa=p.tolist(),volume_m3=t.V0.tolist(),initial_pressure_storage_J=float(.5*C@(p*p)),perturbation_energy_J=float(.5*C@(delta*delta)),initial_content_sum_m3=float(C@p),perturbation_volume_mean_Pa=float(t.V0@delta/t.V0.sum()))
    if abs(cases['Y64']['initial_content_sum_m3']-cases['Y128']['initial_content_sum_m3'])>1e-15:raise ValueError('initial projection mismatch')
    write(run/'S0/initial-condition-protocol.json',dict(status='registered',cases=cases,flux0_semantics='no accepted interval',rest_and_zero_velocity=True,E0_recomputed_per_case=True))
    write(run/'S0/physical-contract.json',dict(status='registered',coupled_protocol_sha256=sha(run/'S0/coupled-protocol.json'),cuts_sha256=sha(run/'S0/grid-protocol-inherited.json'),formal_space=read(run/'selected-space.json'),full_material_order=7,full_mass_order=7,source_zero=True,parameters=v,full_window_s=200e-6,execution_window_s=75e-6,linear_solver='unchanged general balanced LU',old_stabilization=True,baseline_release_sha256=APP_SHA))
    write(run/'S0/observation-contract.json',dict(status='registered',times_s=(np.arange(7)*12.5e-6).tolist(),frame_times_s=[25e-6,75e-6],pressure_atol_Pa=.001,pressure_rtol=.05,content_atol_m3=1e-10,flux_atol_m3_s=1e-10,displacement_atol_m=5e-5,velocity_atol_m_s=1e-4,stress_atol_Pa=.02,reaction_atol_N=1e-4,raw_microstep_rtol=.1,implementation_rtol=2e-5,hard_mass_m3=1e-10,hard_energy='1e-9J + .01*abs(case E0)',hard_minJ=.1,hard_residual=1,pressure_probe_ownership='searchsorted right, clipped on exterior',Y64_Az='unresolved/null',same_physical_probes=True,volume_weighted_pressure=True,flux_interval_integral=True))
    from .runtime import update
    update(run,f'S0已冻结：{run}。扫描{len(pubs)}个发布，直接父SHA={APP_SHA}。三初态总含量一致；扰动储能单独登记。96次总预算，S2/S3/S4=60/26/10，六帧。')
    print(run,flush=True)

if __name__=='__main__':main()
