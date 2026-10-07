"""Same-input equivalence and actual engineering spatial comparison."""
import argparse
import numpy as np
from .provenance import *
from .runtime import update
from .trajectory import case_name,FIELDS
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_continuous_geometry_next.continuous import state_checks
from benchmarks.research_pressure_startup_next.coupling_review import fluid,context,compare_fluid
from benchmarks.research_pressure_window_next.candidate_study import fields,good
from benchmarks.research_sequential_next.compare import metric,regions
from engine.aniso_phase1.research_startup_substeps_next.schedule import aggregate,interval_rows,OBSERVATIONS

def probes(folder):
    paths=sorted(Path(folder).glob('probes-*.npz')) or [Path(folder)/'probes.npz'];out={}
    for p in paths:
        with np.load(p) as z:
            X=z['X'].copy();d=z['fiber'].copy()
            for i,t in enumerate(z['times']):out[round(float(t),14)]={k:z[k][i].copy() for k in FIELDS}
    return X,d,out

def balances(folder):
    h=history(folder);a,b=h[0]['state'],h[-1]['state'];rows=h[-1]['rows'];f=a.child_states['fluid'];g=b.child_states['fluid'];E0=read(Path(folder)/'execution-protocol.json')['initial_energy_J']
    mass=float(np.sum(np.asarray(g['content_m3'])-f['content_m3'])+g['cumulative_boundary_m3']-f['cumulative_boundary_m3']-np.sum(np.asarray(g['cumulative_source_m3'])-f['cumulative_source_m3']))
    energy=rows[-1]['total_energy_J']-E0+sum(r['darcy_dissipation_J']+r['numerical_dissipation_J']-r['external_work_J']-r['source_work_J']-r['reservoir_work_J'] for r in rows)
    dn=g['cumulative_numerical_dissipation_J']-sum(r['numerical_dissipation_J'] for r in rows)
    out=dict(mass_defect_m3=mass,energy_balance_J=energy,Dnum_defect_J=dn,initial_energy_J=E0,min_detF=min(r['min_detF'] for r in rows),max_residual_fraction=max(r['true_scaled_residual'] for r in rows),minimum_pressure_Pa=min(min(v['state'].child_states['fluid']['pressure_Pa']) for v in h),zero_source=not np.any(g['cumulative_source_m3']),dissipation_nonnegative=all(r['darcy_dissipation_J']>=0 and r['numerical_dissipation_J']>=0 for r in rows))
    out['passed']=abs(mass)<1e-10 and abs(energy)<1e-9+.01*abs(E0) and abs(dn)<1e-12 and out['min_detF']>.1 and out['max_residual_fraction']<=1 and out['minimum_pressure_Pa']>=0 and out['zero_source'] and out['dissipation_nonnegative'];return out

def coarse(run):
    run=Path(run);folder=run/'cases'/case_name('base');actual=history(folder);ref=history(TRAJECTORY/'cases/boundary32-h');out=[]
    for a in actual:
        b=ref[a['state'].step];sa,sb=a['state'],b['state']
        if sa.predictor is None or sb.predictor is None:
            if sa.predictor is not None or sb.predictor is not None:raise ValueError('initial predictor semantics differ')
            # t=0 has no prior velocity predictor; preserve that fact explicitly.
            sa=sa.clone();sb=sb.clone();sa.predictor=np.zeros_like(sa.q);sb.predictor=np.zeros_like(sb.q)
        checks=state_checks(sa,sb)
        if a['rows']:
            for k in ('reaction_N','total_energy_J','energy_balance_J','pressure_solid_work_J','pressure_fluid_work_J'):
                checks[k]=metric(a['rows'][-1][k],b['rows'][-1][k],1e-8 if k=='reaction_N' else 1e-12,2e-5)
        out.append(dict(step=a['state'].step,checks=checks,passed=all(v['passed'] for v in checks.values())))
    bal=balances(folder);passed=all(x['passed'] for x in out) and bal['passed']
    write(run/'S2/coarse-equivalence.json',dict(status='passed_scoped' if passed else 'failed',records=out,balances=bal,source=str(TRAJECTORY/'cases/boundary32-h')))
    if not passed:raise ValueError('D3 coarse differs from authenticated original')
    print('COARSE_EQUIVALENCE',passed,bal,flush=True)

def compare(run,fine_time=False):
    run=Path(run);a=run/'cases'/case_name('yz' if fine_time else 'base');b=run/'cases'/case_name('yz',fine_time);cuts=read(run/'S0/grid-protocol-inherited.json')['cuts'];storage=read(run/'S0/coupled-protocol.json')['parameters']['storage']
    ca=context(cuts['yz' if fine_time else 'base'],storage);cb=context(cuts['yz'],storage)
    va,t,rows=fluid(a);vb,tt,rhs=fluid(b);obs=OBSERVATIONS[:7];flow=compare_fluid(ca,cb,aggregate(va,t,obs),aggregate(vb,tt,obs),obs)
    X,d,pa=probes(a);Y,_,pb=probes(b)
    if not np.array_equal(X,Y):raise ValueError('different physical probes')
    ea=interval_rows(rows,obs,average=['reaction_N']);eb=interval_rows(rhs,obs,average=['reaction_N']);solid=[]
    for i,t0 in enumerate(obs[1:]):
        t0=round(float(t0),14);fa,fb=pa[t0],pb[t0];checks=fields(dict(X=X,**fa),dict(X=X,**fb),d)
        for region,w in regions(X).items():
            checks[region].update({k:metric(fa[k],fb[k],.02,.05,w) for k in ('PK1_total','Cauchy_skeleton','Cauchy_total')})
            checks[region]['fiber_total']=metric(np.einsum('i,...ij,j->...',d,fa['PK1_total'],d),np.einsum('i,...ij,j->...',d,fb['PK1_total'],d),.02,.05,w)
        reaction=metric(ea[i]['reaction_N'],eb[i]['reaction_N'],1e-4,.05)
        solid.append(dict(time_s=t0,checks=checks,reaction=reaction,passed=good(checks) and reaction['passed'],displacement_signal_m=float(np.max(abs(fb['x']-X))),max_displacement_difference_m=float(np.max(abs(fa['x']-fb['x'])))))
    bal=[balances(a),balances(b)];passed=flow['status']=='passed_scoped' and all(x['passed'] for x in [*solid,*bal])
    out=dict(status='passed_scoped' if passed else 'limited',flow=flow,solid=solid,balances=bal,range_s=[0.,75e-6],time_refinement=fine_time,full_spatial_certificate=False)
    name='time-separation' if fine_time else 'spatial-comparison';write(run/f'S2/{name}.json',out)
    if not fine_time:
        va,t,_=fluid(TRAJECTORY/'cases/boundary32-h');vb,tt,_=fluid(TRAJECTORY/'cases/boundary32-half');timecheck=compare_fluid(ca,ca,aggregate(va,t,obs),aggregate(vb,tt,obs),obs)
        maximum=max(flow['max_budget_ratios'].values());inherited=max(timecheck['max_budget_ratios'].values());needed=not passed or inherited>.25 and maximum>.5
        write(run/'S2/time-entry-decision.json',dict(status='triggered' if needed else 'not_triggered',need_half=needed,inherited_coarse_time_budget_ratio=inherited,spatial_budget_ratio=maximum,reason='separate time influence on important spatial discrepancy' if needed else 'two-grid engineering differences leave sufficient margin; global accuracy remains unqualified',inherited_time=timecheck))
    update(run,f'S2 {name}：{out["status"]}，流体最坏预算比{max(flow["max_budget_ratios"].values()):.4g}；质量/能量硬条件分别为{[x["passed"] for x in bal]}。')
    print(name,out['status'],flow['max_budget_ratios'],bal,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['coarse','spatial','time']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):coarse(a.run) if a.phase=='coarse' else compare(a.run,a.phase=='time')
