"""Three new bounded scenarios, sufficient material rules, and owned rule retry."""
from pathlib import Path
import argparse
import copy
import gc
import time
import numpy as np
import warp as wp
from .provenance import read,write,serial_lock,register_study,utc,sha,ROOT
from .checkpoint import GenerationStore
from .config import protocol
from .run import load_model,probe_frame
from .compare import metric,regions
from .diagnostics import weak_moments
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF,StepRejected
from engine.aniso_phase1.research_sequential_next.rules import FixedRuleRetry
from engine.aniso_phase1.research_d.stage2.contracts import array_digest

SCENARIOS=[dict(name='angle37',fiber_angle_degrees=37.,peak_m=.005),
           dict(name='angle58',fiber_angle_degrees=58.,peak_m=.005),
           dict(name='peak0075',fiber_angle_degrees=None,peak_m=.0075)]


def settings(run,scenario,order):
    baseline=read(Path(run)/'baseline-lock.json')
    cfg=protocol(baseline['energy_scale_J'],dt=.025,end=.2,material_order=order)
    cfg['scenario']={k:v for k,v in scenario.items() if k!='name'}
    return cfg


def compare_fields(model,a,b,cfg):
    fa=probe_frame(model,a,cfg['probe_shape']);fb=probe_frame(model,b,cfg['probe_shape']);weights=regions(fa['X'])
    direction=model.parent.params.fiber_direction;out={}
    for name,w in weights.items():
        out[name]={k:metric(fa[k]-(fa['X'] if k=='x' else 0),fb[k]-(fb['X'] if k=='x' else 0),atol,.05,w)
                   for k,atol in [('x',5e-5),('velocity',1e-4),('PK1',.02)]}
        x=np.einsum('i,...ij,j->...',direction,fa['PK1'],direction);y=np.einsum('i,...ij,j->...',direction,fb['PK1'],direction)
        out[name]['fiber_PK1']=metric(x,y,.02,.05,w)
    return out


def material_metrics(a,b,wa,wb):
    result={}
    for key,absolute,rtol in [('material_U',1e-10,.02),('material_force',1e-8,.02),('tangent_action',1e-8,.03)]:
        result[key]=metric(a[key],b[key],absolute,rtol)
    # Constant moments have force-length units, coordinate moments one extra
    # length; component-specific records avoid hiding an individual thin slab.
    result['weak_moments']=dict(overall=metric(wa,wb,1e-10,.02),
        per_slab_probe=[[metric(x,y,1e-10,.02) for x,y in zip(sa,sb)] for sa,sb in zip(wa,wb)])
    result['passed']=all(result[k]['passed'] for k in ['material_U','material_force','tangent_action']) and result['weak_moments']['overall']['passed'] and all(v['passed'] for row in result['weak_moments']['per_slab_probe'] for v in row)
    return result


def study(run):
    run=Path(run);source_case=run/'cases/gpu-q7-dt0025'
    history=GenerationStore(source_case,read(source_case/'identity.json')).history();states={round(v['state'].time,10):v['state'] for v in history}
    prereg=register_study(run,'N10/protocol.json',dict(utc=utc(),scenarios=SCENARIOS,seed=1010,
        seen_angle_registry=dict(path='docs/results/parallel-v22/A/20260930T0405Z-A-pilot/generalization-protocol.json',
          sha256=sha(ROOT/'docs/results/parallel-v22/A/20260930T0405Z-A-pilot/generalization-protocol.json'),angles=[15,30,45,60,75]),
        registry_search='25 existing protocol/config/case/handoff JSON files, no explicit angle37/58 or peak0.0075 hit; scoped novelty, not exhaustive all historical state amplitudes',
        material_states=[.5,1.1],material_state_source='existing displacement states under new material; kinematic probes, not claimed new-material dynamic trajectories',
        directions=['seeded mixed free direction','largest coefficient row of original free rest stiffness as sensitive probe'],
        time_step=.025,short_steps=8,mass_order=5,full_order=7,check_order=8,compressed_order=5,
        tolerances=dict(material_rtol=.02,tangent_rtol=.03,field_rtol=.05,energy_atol_J=1e-10,force_atol_N=1e-8,weak_moment_atol=1e-10),
        full_cycle='at most peak0075, after short validation; separate fixed full/compact cases may be created at final source freeze'))
    results=[];last_models=None
    for scenario in SCENARIOS:
        begun=time.perf_counter();cfg=settings(run,scenario,7);models={}
        for order in [7,8,5]:models[order],_=load_model(run,settings(run,scenario,order))
        m=models[7];rng=np.random.default_rng(1010);random=rng.normal(size=(m.space.ndof,3));random[m.fixed]=0;random/=np.linalg.norm(random)
        sensitive=np.zeros_like(random);j=int(np.argmax(np.diag(m.rest_K)[m.ids]));sensitive.ravel()[m.ids[j]]=1.
        directions=[random,sensitive]
        # Check the reassembled material operator independently on full-space
        # local-only directions, where stabilization contributes exactly zero.
        rest=[];n=m.parent.ndof
        for i in range(2):
            full_d=np.zeros((n,3));full_d[m.parent.n:]=rng.normal(size=(n-m.parent.n,3));full_d/=np.linalg.norm(full_d)
            lin=m.operator.prepare(np.zeros_like(full_d));direct=m.operator.action(lin,full_d).ravel()
            algebraic=m.reduction.original_stiffness@full_d.ravel()
            err=float(np.linalg.norm(direct-algebraic)/max(np.linalg.norm(direct),1e-12));rest.append(err)
        if max(rest)>2e-5:raise ValueError('new material rest stiffness failed independent GPU action')
        mass_equal=all(np.array_equal(m.M,z.M) for z in models.values())
        if not mass_equal:raise ValueError('material rules changed mass')
        checks=[]
        for t in [.5,1.1]:
            q=states[t].q*(scenario['peak_m']/.005)
            weak={o:weak_moments(z,q) for o,z in models.items()}
            for i,d in enumerate(directions):
                values={o:z.evaluate(q,d) for o,z in models.items()}
                checks.append(dict(time=t,direction=i,
                    sufficient=material_metrics(values[7],values[8],weak[7],weak[8]),
                    compressed=material_metrics(values[5],values[7],weak[5],weak[7]),
                    min_detF=min(z['min_detF'] for z in values.values())))
        sufficient=all(x['sufficient']['passed'] for x in checks);compressed=all(x['compressed']['passed'] for x in checks)
        if not sufficient:raise ValueError('seven/eight rule insufficient for registered new scene: '+scenario['name'])
        # Release eight-order buffers before the actual dynamics pair.
        del models[8];gc.collect()
        steppers={o:ValidatedAVF(z,settings(run,scenario,o)) for o,z in models.items()}
        rows={o:[] for o in steppers};field_checks=[]
        for step in range(8):
            for order,integrator in steppers.items():rows[order].append(integrator.step(.025))
            fields=compare_fields(m,steppers[5].state,steppers[7].state,cfg)
            reaction=metric(rows[5][-1]['reaction_N'],rows[7][-1]['reaction_N'],1e-4,.05)
            field_checks.append(dict(step=step+1,time=steppers[7].state.time,regions=fields,reaction=reaction,
                passed=reaction['passed'] and all(v['passed'] for region in fields.values() for v in region.values())))
        dynamic=all(x['passed'] for x in field_checks)
        # Retry is tested on the real target space, including child proposal
        # ownership and durable whitelist checkpoint reload.
        controller=FixedRuleRetry(models[5],models[7],cfg);initial=controller.state;attempt_rules=[]
        def children(candidate,values):values['E_test_values']={'p':[.1,.2],'time':candidate.time};return values
        def fail(attempt,where,state):
            if where=='before_material':attempt_rules.append(state.child_states['identity']['material'])
            if attempt==0 and where=='after_prepare':raise ValueError('intentional compressed rule failure after child proposal')
        row=controller.step(.025,prepare_children=children,inject=fail)
        sticky=controller.state.child_states['identity']==models[7].identity
        control_initial=models[7].rest();control=ValidatedAVF(models[7],cfg,control_initial);control.step(.025,prepare_children=children)
        difference=max(float(np.max(abs(control.state.q-controller.state.q))),float(np.max(abs(control.state.velocity-controller.state.velocity))))
        if not sticky or difference>1e-8 or row['material_attempts']!=2:raise ValueError('real fixed-rule retry failed')
        store=GenerationStore(run/'N10/retry-checkpoints'/scenario['name'],controller.identity)
        store.save(initial,[]);store.save(controller.state,[row]);restored=store.load(validator=controller.validate)
        if restored['state'].digest()!=controller.state.digest():raise ValueError('rule whitelist checkpoint restore mismatch')
        resumed=FixedRuleRetry(models[5],models[7],cfg,restored['state']);resumed.step(.025);controller.step(.025)
        if resumed.state.digest()!=controller.state.digest():
            # Wall-clock timing is observational in last_ledger. Numerical
            # q/v/predictor and rules are the continuation invariants.
            for attr in ['q','velocity','predictor']:
                if not np.array_equal(getattr(resumed.state,attr),getattr(controller.state,attr)):raise ValueError('rule resume state differs')
        result=dict(name=scenario['name'],scenario=scenario,status='passed_scoped' if compressed and dynamic else 'compressed_rule_limited',
            material_reassembly=m.scenario_audit,independent_rest_action_relative=rest,mass_unchanged=mass_equal,mass_sha256=array_digest(m.M),
            checks=checks,sufficient_material=sufficient,compressed_material=compressed,short_dynamic=dynamic,
            dynamics=dict(rows=rows,comparisons=field_checks),
            rule_retry=dict(attempt_rules=attempt_rules,accepted_rule=row['material_rule'],attempts=row['material_attempts'],
                q_v_difference=difference,whole_state_rollback=True,checkpoint_reload=True,resume=True,energy_shift_J=row['rule_switch_energy_J']),
            seconds=time.perf_counter()-begun,gpu_memory=m.operator.memory_budget.report())
        write(run/f'N10/{scenario["name"]}.json',result);results.append(result)
        print(scenario['name'],'sufficient',sufficient,'compact',compressed,'dynamic',dynamic,'seconds',round(result['seconds'],2),flush=True)
        del models,steppers,m,controller,resumed,control;gc.collect()
    result=dict(utc=utc(),status='passed_scoped',scenarios=[dict(name=x['name'],status=x['status'],sufficient=x['sufficient_material'],
        compressed=x['compressed_material'],dynamic=x['short_dynamic']) for x in results],
        full_cycle_candidate='peak0075',full_cycle_status='pending final frozen-source cycle',
        physical_space_original144=True,scientific_spatial_time_accuracy_certified=False)
    write(run/'N10/result.json',result)
    caps=read(run/'capabilities.json');caps['N10']=dict(status='passed_scoped',evidence='N10/result.json',full_cycle=False);write(run/'capabilities.json',caps)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
