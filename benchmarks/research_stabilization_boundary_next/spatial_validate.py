"""Validate the already-frozen single candidate, one live model per process."""
from pathlib import Path
import argparse,gc,time
import numpy as np
import scipy.linalg as la
from .provenance import *
from .spatial import PB,CROSS,OP,NAME
from benchmarks.research_spatial_phase_next.spaces import reopen_candidate
from benchmarks.research_phase_boundary_next.solver import bounded_static
from benchmarks.research_cross_direction_next.spatial_study import material,nodal
from benchmarks.research_reference_next.field_audit import compare_nodal,invariants
from benchmarks.research_sequential_next.compare import metric,regions
from benchmarks.research_sequential_next.run import probe_frame
from engine.aniso_phase1.research_sequential.condensation import CondensedModel
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_d.common_kinetic import PointInertia
from engine.aniso_phase1.tensor_metrics import sampling

def load(run):
    import warp as wp
    run=Path(run);verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');target=run/'S4/candidates'/NAME
    expected=read(run/'S4/candidate-package.json')['sha256']
    if sha(target/'space-package.json')!=expected:raise ValueError('frozen single candidate changed')
    cr,p=reopen_candidate(target)
    with np.load(target/'space.npz') as z:initial=z['initial'].copy()
    return target,cr,initial

def operators(run):
    target,cr,initial=load(run);run=Path(run);rng=np.random.default_rng(20261005);d=rng.normal(size=(cr.parent.ndof,3));d/=la.norm(d)
    p7=PointInertia(cr.parent,order=7).apply(d);p8=PointInertia(cr.parent,order=8).apply(d);mass=dict(transform=metric(cr.original_mass@d,p7,1e-10,2e-5),sufficiency=metric(p7,p8,1e-10,2e-5))
    q=cr.project(initial);dq=rng.normal(size=q.shape);dq[cr.fixed]=0;dq/=la.norm(dq)
    gpu=SegmentedModel(cr,order=7,device='cuda:0',hold=.005);a=gpu.evaluate(q,dq);del gpu;gc.collect();cpu=CondensedModel(cr,order=7,device='cpu',hold=.005);b=cpu.evaluate(q,dq);del cpu;gc.collect();hardware={k:metric(a[k],b[k],1e-8,2e-5) for k in ('U','force','tangent_action')}
    if not all(v['passed'] for v in [*mass.values(),*hardware.values()]):raise ValueError('candidate operators')
    write(run/'S4/operator-check.json',dict(status='passed_scoped',mass=mass,cpu_gpu=hardware,invariants=invariants(cr.parent),rank=cr.audit,no_artificial_mass=True,complete_cross_mass=True,same_frozen_candidate=True,driver_sha256=sha(__file__)));print('SPACE_OPERATORS',hardware,flush=True)

def static(run,angle):
    run=Path(run)
    if read(run/'S4/operator-check.json')['status']!='passed_scoped':raise ValueError('operators first')
    target,cr,initial=load(run);path=target/f'F{angle}-search.json'
    if path.exists():raise ValueError('static solve already attempted')
    rr=material(cr,angle);dq=np.random.default_rng(20261005).normal(size=rr.P.shape[1:]+(3,));dq[rr.fixed]=0;dq/=la.norm(dq)
    m=SegmentedModel(rr,order=7,device='cuda:0',hold=.005);state,stats=bounded_static(m,initial,path,240);eval7=m.evaluate(state.q,dq);frame=probe_frame(m,state,[33,7,7]);full=rr.expand(state.q);del m;gc.collect()
    high=SegmentedModel(rr,order=8,device='cuda:0',hold=.005);eval8=high.evaluate(state.q,dq);del high;gc.collect();integration={k:metric(eval7[k],eval8[k],at,rt) for k,at,rt in [('material_U',1e-10,.02),('material_force',1e-8,.02),('tangent_action',1e-8,.03)]}
    refs={45:PHASE_REFERENCE/'S2/R5/data.npz',52.5:CROSS/'S1/reserved/R5/state.npz',60:OP/'S2/direction/R5/state-fields.npz',48.75:PB/'S2/reserved/R5/state.npz'}
    reactions={45:PHASE_REFERENCE/'S2/R5/result.json',52.5:CROSS/'S1/reserved/R5/result.json',60:OP/'S2/direction/R5/result.json',48.75:PB/'S2/reserved/R5/result.json'}
    with np.load(PB/'S2/reserved/R5/state.npz') as z:edges=tuple(z[f'e{i}'].copy() for i in range(3))
    with np.load(refs[angle]) as z:nodes=z['nodes'].copy()
    refnodal=(edges,6,nodes);errors=compare_nodal(nodal(rr,full),refnodal,rr.parent.A,rr.parent.params);reaction=metric(stats['reaction_N'],read(reactions[angle])['solve']['reaction_N'],1e-4,.05)
    from engine.aniso_phase1.tensor_reference import apply_axis
    refu=nodes.reshape(tuple(6*(len(e)-1)+1 for e in edges)+(3,))
    axes=[np.linspace(e[0],e[-1],n) for e,n in zip(edges,(33,7,7))]
    # Same tensor interpolation as the reference space, without a second model.
    for axis,(e,x) in enumerate(zip(edges,axes)):refu=apply_axis(sampling(e,6,x),refu,axis)
    disp={k:metric(frame['x']-frame['X'],refu,5e-5,.05,w) for k,w in regions(frame['X']).items()}
    old=read(PB/'S2/training-comparison.json')['records'];reserved=read(PB/'S2/reserved-direction-check.json');audit=read(PB/'S2/reference-audit.json');key=str(int(angle)) if int(angle)==angle else str(angle)
    old_errors=reserved['new'] if angle==48.75 else old[key]['errors'];adjacent=reserved['adjacent'] if angle==48.75 else audit['records'][key]['adjacent'];local=[]
    if not all(v['passed'] for row in errors.values() for v in row.values()):local.append('regional stress budget')
    if not all(v['passed'] for v in [reaction,*disp.values(),*integration.values()]):local.append('reaction/displacement/material rule')
    for reg,row in errors.items():
        for key2,v in row.items():
            if v['absolute']-old_errors[reg][key2]['absolute']>max(2*adjacent[reg][key2]['absolute'],.1*v['budget']):local.append('regression '+reg+'/'+key2)
    np.savez_compressed(target/f'F{angle}.npz',q=state.q,full=full,**frame);write(run/f'S4/static-{angle}.json',dict(status='passed_scoped' if not local else 'space_limited',angle=angle,errors=errors,old=old_errors,reaction=reaction,displacement=disp,integration=integration,solve=stats,reasons=local,reference_sha256=sha(refs[angle]),driver_sha256=sha(__file__)))
    print('NEW_SPACE_STATIC',angle,errors['interior'],local,flush=True)

def finish(run):
    run=Path(run);results={str(a):read(run/f'S4/static-{float(a)}.json') for a in (45,52.5,60,48.75)};reasons=[x for v in results.values() for x in v['reasons']];row=results['60'];goal=row['errors']['interior']['fiber_PK1'];oldgoal=row['old']['interior']['fiber_PK1'];uncert=read(PB/'S2/reference-audit.json')['records']['60']['adjacent']['interior']['fiber_PK1']['absolute'];gain=oldgoal['absolute']-goal['absolute'];resolvable=oldgoal['absolute']>=.02
    if gain<.1*oldgoal['absolute'] or gain<=2*uncert:reasons.append('primary stress gain not resolved above 10% and twice reference difference')
    if not resolvable:reasons.append('primary static error already below absolute engineering scale; percentage gain alone is insufficient')
    write(run/'S4/static-comparison.json',dict(status='space_limited' if reasons else 'passed_scoped',records=results,reasons=reasons,primary_improvement_Pa=gain,reference_uncertainty_Pa=uncert,primary_above_absolute_scale=resolvable,new_static_solves=4,reference_solves=0,historical_48_75_not_hidden=True))
    write(run/'S4/space-decision.json',dict(status='space_limited' if reasons else 'pending_local_dynamics',reasons=reasons,new_spaces=1,new_static_solves=4,new_dynamic_attempts=0,static_passed=not reasons,formal_space_changed=False,global_spatial_accuracy=False,dynamic_eligibility=not reasons,original_stabilization_preserved=True,resource_failure_fixed_by_serial_processes=True));print('SPACE_DECISION',not reasons,reasons,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['operators','static','finish']);p.add_argument('--run',type=Path,required=True);p.add_argument('--angle',type=float,choices=[45,52.5,60,48.75]);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='static':static(a.run,a.angle)
        else:globals()[a.phase](a.run)
