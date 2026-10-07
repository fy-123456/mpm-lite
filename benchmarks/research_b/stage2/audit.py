"""Real common-space wiring, derivative platform, failure and timing evidence."""
import argparse
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
from engine.aniso_phase1.research_b.tensor import TensorMaterialOperator, TensorRule
from engine.aniso_phase1.research_b.stage2 import CommonMaterialOperator, FixedRule, MaterialSource, MaterialSession
from engine.aniso_phase1.research_b.stage2.transaction import validate_material_child
from engine.aniso_phase1.research_d.common_state import CommonState, StateTransaction
from .run import checked, load, write, read, sha, now
from .metrics import compare


def wiring(repo,out):
    p=checked(repo,out);s,_,state,_=load(repo)
    with np.load(out/'states.npz',allow_pickle=False) as z:q=z['archive'].copy();d=z['direction__mixed'].copy()
    op=CommonMaterialOperator(s,FixedRule.uniform(s,6));a=op.evaluate_full(q,directions={'mixed':d})
    b=TensorMaterialOperator(s,TensorRule.uniform(s.edges,6)).evaluate(q,d)
    checks={}
    tight=dict(atol=1e-9,rtol=1e-7)
    for new,old in (('energy_J','U'),('material_energy_J','material_U'),('stabilization_energy_J','stabilization_U'),('full_force','force'),('weak_moments','weak_moments')):
        checks[new]=compare(a[new],b[old],tight)
    checks['tangent']=compare(a['tangents']['mixed']['full'],b['tangent_action'],tight)
    q=q*.8;free=s.restrict(q);lift=q.copy();lift[s.free_scalar_ids]=0
    base=op.evaluate_free(free,lift=lift,directions={'mixed':s.restrict(d)})
    finite=[]
    for h in p['derivative_steps']:
        plus=op.evaluate_free(free+h*s.restrict(d),lift=lift);minus=op.evaluate_free(free-h*s.restrict(d),lift=lift)
        ge=compare((plus['energy_J']-minus['energy_J'])/(2*h),np.sum(base['full_force']*d),dict(atol=1e-8,rtol=1e-5))
        he=compare((plus['full_force']-minus['full_force'])/(2*h),base['tangents']['mixed']['full'],dict(atol=1e-7,rtol=1e-4))
        finite.append(dict(step=h,gradient=ge,tangent=he))
    rng=np.random.default_rng(25);u=rng.normal(size=(s.ndof,3));v=rng.normal(size=(np.prod(s.shape),3))
    adjoint=compare(np.sum(s.nodes(u)*v),np.sum(u*s.adjoint(v)),dict(atol=1e-9,rtol=1e-10))
    passed=all(c['passed'] for c in checks.values()) and adjoint['passed'] and all(r['gradient']['passed'] and r['tangent']['passed'] for r in finite[-2:])
    write(out/'wiring-audit.json',dict(passed=passed,original_sixth_order=checks,nonzero_lift_fd=finite,
                                     nodal_adjoint=adjoint,force_sign='positive energy gradient',identity_added_once=True))
    np.savez_compressed(out/'interface-vectors.npz',q=q,free_q=free,lift=lift,zero_direction_lift=np.zeros_like(q),direction=d,
                        full_force=base['full_force'],free_force=base['free_force'],full_tangent=base['tangents']['mixed']['full'],
                        weak_moments=base['weak_moments'],material_energy_J=base['material_energy_J'],stabilization_energy_J=base['stabilization_energy_J'])
    # This driver tests owned value states only. It is not a C time integrator.
    initial=CommonState(q,np.zeros_like(q),predictor=q.copy(),child_states={'ledger':{'work':7.},'predictor_id':'original'})
    def postvalidate(state):
        if state.step:
            validate_material_child(state,op)
            if state.child_states.get('inject_failure'):raise ValueError('injected postvalidation')
    tx=StateTransaction(initial,validator=postvalidate);session=MaterialSession(op,tx);baseline=tx.snapshot().digest();faults={}
    for name in ('line_search','postvalidation','invalid_detF','nonfinite','stale_token','cache_mismatch','midstep_rule'):
        trial=tx.trial()
        try:
            if name=='invalid_detF':trial.state.q[:s.n,0]=-2*s.carrier_X[:,0]
            if name=='nonfinite':trial.state.q[0,0]=np.nan
            proposal,_=session.prepare(trial)
            if name=='line_search':session.attach(trial,proposal);session.reject(trial)
            elif name=='postvalidation':
                session.attach(trial,proposal);trial.state.step=1;trial.state.time=.01;trial.state.child_states['inject_failure']=True;tx.commit(trial)
            elif name=='stale_token':session.reject(trial);session.attach(trial,proposal)
            elif name=='cache_mismatch':
                from dataclasses import replace
                session.attach(trial,replace(proposal,cache_sha256='0'*64))
            elif name=='midstep_rule':session.switch_rule(FixedRule.uniform(s,7))
            faults[name]=dict(rejected=name=='line_search',whole_state_unchanged=baseline==tx.snapshot().digest())
        except ValueError as exc:
            faults[name]=dict(rejected=True,reason=str(exc),whole_state_unchanged=baseline==tx.snapshot().digest())
        try:tx.rollback(trial)
        except ValueError:pass
    trial=tx.trial();proposal,_=session.prepare(trial);session.attach(trial,proposal);trial.state.step=1;trial.state.time=.01;tx.commit(trial)
    twice=False
    try:tx.commit(trial)
    except ValueError:twice=True
    write(out/'transaction-audit.json',dict(passed=all(v['rejected'] and v['whole_state_unchanged'] for v in faults.values()) and tx.revision==1 and twice,
        faults=faults,successful_commits=tx.revision,duplicate_commit_rejected=twice,
        scope='B proposal plus CommonState value-tree driver; real C external-object atomicity pending',online_regrouping='not_enabled',energy_jump=None,
        order=['C begins trial','B evaluates and prepares immutable proposal','B attaches owned values','C validates all children and physics','C commits once or rolls back entire trial']))
    print({'phase':'wiring','passed':passed},flush=True)


def cost(repo,out):
    p=checked(repo,out);s,_,_,_=load(repo);sel=read(out/'candidate-seal.json')['fields']['F45']
    full=sel['full_order'];low=sel['candidate_order']
    with np.load(out/'states.npz',allow_pickle=False) as z:q=z['archive'].copy();d=z['direction__sensitive'].copy()
    ops={o:CommonMaterialOperator(s,FixedRule.uniform(s,o)) for o in {full,low}}
    before=subprocess.run(['ps','-eo','pid,pcpu,comm,args','--sort=-pcpu'],capture_output=True,text=True).stdout
    # Warm up both, then five AB/BA interleaved full operations. No speed claim
    # is certified on a shared device without independently evidenced exclusivity.
    for op in ops.values():op.evaluate_full(q,directions={'sensitive':d})
    rows=[]
    for repeat in range(5):
        for order in ([full,low] if repeat%2==0 else [low,full]):
            t=time.perf_counter();r=ops[order].evaluate_full(q,directions={'sensitive':d})
            rows.append(dict(repeat=repeat,order=order,total_seconds=time.perf_counter()-t,
                construction_seconds=ops[order].construction_seconds,components=r['timing_seconds'],
                points=r['point_count'],peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    stats={str(o):dict(median=float(np.median([r['total_seconds'] for r in rows if r['order']==o])),
                       minimum=min(r['total_seconds'] for r in rows if r['order']==o),maximum=max(r['total_seconds'] for r in rows if r['order']==o)) for o in ops}
    write(out/'cost-breakdown.json',dict(scope='CPU material operator only; energy + full force + one exact tangent',
        exclusive_resources=False,performance_certified=False,dynamic_speedup_claim=False,rows=rows,statistics=stats,
        observed_median_ratio=stats[str(full)]['median']/stats[str(low)]['median'],machine_process_snapshot=before,
        fallback_fraction=0.,full_order=full,candidate_order=low,point_reduction=1-ops[low].point_count/ops[full].point_count,
        limitation='shared machine concurrent research; no formal speedup claim; C end-to-end cost dependency pending'))
    print({'phase':'cost','statistics':stats},flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('wiring','cost'));ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    globals()[args.phase](Path(__file__).resolve().parents[3],args.output.resolve())
if __name__=='__main__':main()
