"""Cross-check mass attribution using independently recorded geometric delta-volume."""
import argparse
import numpy as np
from .provenance import *
from .runtime import update
from benchmarks.research_phase_stress_next.time_study import history
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology


def main(run):
    run=Path(run);mutable(run);p=read(run/'S0/physical-contract.json');top=ReferenceTopology(p['cuts']);C=p['parameters']['storage']*top.V0;alpha=p['parameters']['alpha'];h=history(APP/'cases/YZ128');centres=np.mean(np.array(top.cell_bounds),axis=2)
    totals={};maxdv=0.
    for axis,name in ((1,'Ay'),(2,'Az')):
        eta=2*(centres[:,axis]-.5)/.25;w=top.V0*eta/(top.V0@(eta*eta));contributions=[]
        for prev,nxt in zip(h[:-1],h[1:]):
            row=nxt['rows'][-1];g=nxt['state'].child_states['fluid'];f=prev['state'].child_states['fluid'];dp=np.array(g['pressure_Pa'])-f['pressure_Pa'];dv=np.array(row['delta_volume_m3']);dc=np.array(g['content_m3'])-f['content_m3'];maxdv=max(maxdv,float(np.max(abs(alpha*dv-(dc-C*dp)))))
            contributions.append(float(w@(-alpha*dv/C)))
        totals[name]=sum(contributions)
    prior=read(run/'S1/coupled-model-difference.json');errors={k:abs(v-prior['total_modes'][k]['volume_contribution_Pa']) for k,v in totals.items()}
    passed=max(errors.values())<1e-8 and maxdv<1e-10
    write(run/'S1/geometric-volume-crosscheck.json',dict(status='passed_scoped' if passed else 'failed',mode_contributions_from_geometric_delta_volume_Pa=totals,mode_difference_from_content_reconstruction_Pa=errors,max_content_relation_defect_m3=maxdv,scope='direct geometric delta_volume from original solver ledgers, independent of reconstructed content identity; still not continuum truth'))
    # The detailed static timer inherited one initial construction evaluation.
    diag=read(run/'S3/DV-adjoint-profile.json')
    write(run/'S3/profile-interpretation.json',dict(status='limited',bridge_profile_source='S3/DV-profile.json',static_adjoint_source='S3/DV-adjoint-profile.json',static_adjoint_calls=diag['parent_adjoint_calls'],static_geometry_counters_include_initial_construction=True,do_not_subtract_nonmatching_counters=True,qualification_timers='paired whole-step wall time only',remaining_attribution='gather, download, P contraction and host allocations not fully separated',new_dynamic_steps=0))
    if not passed:raise ValueError('geometric pressure attribution differs')
    update(run,'异常归因复核：直接使用原求解器记录的几何delta_volume，重算横向压力体积贡献，与含量重建一致；结论不依赖单纯重排含量恒等式。细粒度静态计时含初始化一次，明确不拿混合计数做精确归因。')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
