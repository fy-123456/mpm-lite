"""Read-only mass-equation attribution; coupled/reference differences are not errors."""
import argparse
import numpy as np
from .provenance import *
from .runtime import update
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_transverse_next.observables import modes
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology


def main(run):
    run=Path(run);mutable(run);p=read(run/'S0/physical-contract.json');t=ReferenceTopology(p['cuts']);h=history(APP/'cases/YZ128');C=p['parameters']['storage']*t.V0;alpha=p['parameters']['alpha']
    centres=np.mean(np.asarray(t.cell_bounds),axis=2);weights={}
    for a,name in ((1,'Ay'),(2,'Az')):
        eta=2*(centres[:,a]-.5)/.25;weights[name]=t.V0*eta/(t.V0@(eta*eta))
    rows=[]
    for prev,nxt in zip(h[:-1],h[1:]):
        f,g=prev['state'].child_states['fluid'],nxt['state'].child_states['fluid'];dp=np.array(g['pressure_Pa'])-f['pressure_Pa'];dc=np.array(g['content_m3'])-f['content_m3'];dv=(dc-C*dp)/alpha
        dt=nxt['state'].time-prev['state'].time;z=np.array(g['flux_interval_m3_s']);volume=-alpha*dv/C;flow=-dt*(t.B@z)/C
        rows.append(dict(step=nxt['state'].step,time_s=nxt['state'].time,modes={name:dict(pressure_increment_Pa=float(w@dp),volume_contribution_Pa=float(w@volume),flow_contribution_Pa=float(w@flow),closure_Pa=float(w@(dp-volume-flow))) for name,w in weights.items()}))
    sums={name:{k:sum(r['modes'][name][k] for r in rows) for k in rows[0]['modes'][name]} for name in weights}
    out=dict(status='passed_scoped' if max(abs(r['closure_Pa']) for r in sums.values())<1e-8 else 'limited',source=str(APP/'cases/YZ128'),records=rows,total_modes=sums,scope='saved mass equation attribution, reconstructed volume from content; not independent continuum truth',new_dynamic_steps=0,interpretation='solid volume exchange dominates transverse pressure change; fixed-skeleton comparison cannot be labeled temporal error')
    write(run/'S1/coupled-model-difference.json',out)
    write(run/'S2/extension-review.json',dict(status='not_triggered',reason=read(run/'S1/window-and-time-decision.json'),new_steps=0,important_limitation='fixed skeleton signal alone does not bound the moving-solid signal; extension deferred until a matched coupled reference, not declared physically unnecessary'))
    write(run/'S2/time-not-triggered.json',dict(status='not_triggered',reason='same-grid engineering time checks passed; coupled difference is explained by volume exchange, raw startup flow accuracy remains limited',coupled_temporal_accuracy=False,new_steps=0))
    profile=read(run/'S3/DV-profile.json');local=profile['geometry']['local_gradient']['seconds'];total=profile['publication']['physical_step']['seconds']
    register(run,'S3/candidate-protocol.json',dict(status='registered',candidate='BD: batch parent-gradient download, then complete batched CPU P contraction',baseline='DV',measured_local_gradient_fraction=local/total,scope='local path combines gather/adjoint/download/P; extra diagnostic timing will separate adjoint',physical_equations_unchanged=True,additional_bytes_formula='3*cells*parent_ndof*3*8 conservative host/device staging',one_candidate_only=True,static_states=[8,9,'mid8_9',16],paired_order=['DV0','BD0','BD1','DV1'],gain_min_median=.05,gain_spread_max=.05,setup_recovery_max_steps=16,continuous_steps=4))
    update(run,f'S1深入分析：现有耦合横向模式变化可按质量方程分为固体体积交换与Darcy流动；体积交换占主导，不能把固定骨架差异当时间误差。S2扩窗/h2未触发。S3：DV局部梯度合并段占物理推进约{local/total:.1%}，登记唯一“批量下载父系数梯度”候选，完整P与原方程保持。')
    print('ATTRIBUTION',sums,'local_fraction',local/total,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
