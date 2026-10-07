"""v13 phase-by-phase time error, zero-work hold budget and unload residuals."""
import json
from pathlib import Path
import numpy as np
from benchmarks.aniso_unresolved_history import OUT,BASE,LEVELS,MODES,load,write
from benchmarks.aniso_dynamic_space import stress
from benchmarks.aniso_projected_history import read_case
PHASES={'ramp':(.05,.5),'hold':(.5,.6),'unload':(.6,1.1),'final_hold':(1.1,1.2),'whole':(.05,1.2)}
BUDGET=('transfer_roundtrip_delta','boundary_projection_delta','solve_delta','final_projection_damping_delta','kinetic_metric_change','selective_dissipation_delta','stabilization_rebuild_delta')


def main():
    p=load(OUT/'protocol.json');data={};checks={};frames={};stressfields={};phase_metrics={};audits=[]
    for name,cfg in p['configs'].items():
        rows,c=read_case(OUT/'cases'/name);data[name]=rows
        c.update(max_stage_budget_error=max(abs(r['stage_budget_error']) for r in rows),max_filter_energy_increase=max(r['selective_dissipation_delta'] for r in rows),max_filter_momentum_change=max(r.get('momentum_change_norm',0.) for r in rows))
        c['passed']=c['passed'] and c['max_stage_budget_error']<1e-12 and c['max_filter_energy_increase']<1e-14 and c['max_filter_momentum_change']<1e-12
        checks[name]=c
        with np.load(OUT/'cases'/name/'frames.npz') as z:frames[name]={k:z[k].copy() for k in z.files}
        z=frames[name];stressfields[name]=stress(z['F'].reshape(-1,3,3),cfg).reshape(z['F'].shape)
        index=lambda t:int(np.argmin(abs(z['time']-t)))
        norm=lambda a:float(np.sqrt(np.mean(np.sum(a*a,axis=(-2,-1)))))
        P=stressfields[name];Ppeak=P[index(.5)];Fpeak=z['F'][index(.5)]-np.eye(3)
        read_at=lambda t:rows[round(t/cfg['dt'])-1]
        metrics={}
        for phase,(a,b) in [('hold',(.5,.6)),('final_hold',(1.1,1.2))]:
            ra,rb=read_at(a),read_at(b);segment=rows[round(a/cfg['dt']):round(b/cfg['dt'])]
            stage={k:sum(r[k] for r in segment) for k in BUDGET};delta=rb['mechanical']-ra['mechanical'];work=rb['loading_work']-ra['loading_work']
            metrics[phase]=dict(start=a,end=b,steps=len(segment),external_work_J=work,mechanical_change_J=delta,mechanical_change_over_start_elastic=delta/max(abs(ra['elastic']),1e-30),
                elastic_change_J=rb['elastic']-ra['elastic'],kinetic_change_J=rb['kinetic']-ra['kinetic'],affine_kinetic_start_J=ra['kinetic_affine'],affine_kinetic_end_J=rb['kinetic_affine'],
                P_change_over_end=norm(P[index(b)]-P[index(a)])/max(norm(P[index(b)]),1e-30),P_change_over_load_peak=norm(P[index(b)]-P[index(a)])/max(norm(Ppeak),1e-30),
                kinetic_to_elastic_end=rb['kinetic']/max(abs(rb['elastic']),1e-30),stages_J=stage,stage_sum_error=abs(delta-sum(stage.values())),
                positive_energy_steps=sum(r['delta_mechanical']>1e-12 for r in segment),max_step_energy_increase_J=max(r['delta_mechanical'] for r in segment),zero_work_passed=abs(work)<1e-15,net_energy_nonincrease=delta<=1e-12)
            assert metrics[phase]['stage_sum_error']<1e-12 and abs(work)<1e-15
        final=rows[-1];metrics['residual']=dict(time=1.2,P_rms_Pa=norm(P[-1]),P_over_load_peak=norm(P[-1])/max(norm(Ppeak),1e-30),F_rms=norm(z['F'][-1]-np.eye(3)),F_over_load_peak=norm(z['F'][-1]-np.eye(3))/max(norm(Fpeak),1e-30),reaction_N=final['right_force'],elastic_J=final['elastic'],kinetic_J=final['kinetic'],net_loading_work_J=final['loading_work'],particle_displacement_rms=float(np.sqrt(np.mean(np.sum((z['x'][-1]-z['x'][0])**2,axis=1)))))
        phase_metrics[name]=metrics
        audits.extend(load(f)|{'case':name} for f in (OUT/'cases'/name).glob('audit-*.json'))
    refinement={}
    for mode in MODES:
        refinement[mode]={}
        for phase,(a,b) in PHASES.items():
            pairs=[]
            for i,(l0,l1) in enumerate(zip(list(LEVELS)[:-1],list(LEVELS)[1:])):
                na,nb=mode+'-'+l0,mode+'-'+l1;za,zb=frames[na],frames[nb];np.testing.assert_allclose(za['time'],zb['time'],atol=1e-11,rtol=0)
                keep=(za['time']>=a-1e-10)&(za['time']<=b+1e-10);t=za['time'][keep]
                norm=lambda q:float(np.sqrt(np.trapezoid(np.mean(np.sum(q*q,axis=(2,3)),axis=1),t)/(t[-1]-t[0])))
                r=dict(pair=[l0,l1])
                for key,fa,fb in [('P',stressfields[na][keep],stressfields[nb][keep]),('F',za['F'][keep]-np.eye(3),zb['F'][keep]-np.eye(3))]:
                    r[key+'_absolute']=norm(fa-fb);r[key+'_relative']=r[key+'_absolute']/max(norm(fb),1e-30)
                    r[key+'_terminal_relative']=float(np.linalg.norm(fa[-1]-fb[-1])/max(np.linalg.norm(fb[-1]),1e-30))
                ta=np.array([r['time'] for r in data[na]]);ta=ta[(ta>=a-1e-10)&(ta<=b+1e-10)]
                ra=np.interp(ta,[r['time'] for r in data[na]],[r['right_force'] for r in data[na]]);rb=np.interp(ta,[r['time'] for r in data[nb]],[r['right_force'] for r in data[nb]])
                rms=lambda q:float(np.sqrt(np.trapezoid(q*q,ta)/(ta[-1]-ta[0])))
                r['reaction_absolute']=rms(ra-rb);r['reaction_relative']=r['reaction_absolute']/max(rms(rb),.001)
                ua=np.interp(ta,[v['time'] for v in data[na]],[v['elastic'] for v in data[na]]);ub=np.interp(ta,[v['time'] for v in data[nb]],[v['elastic'] for v in data[nb]])
                r['energy_absolute']=rms(ua-ub);r['energy_relative']=rms(ua-ub)/max(rms(ub),1e-30)
                if i:
                    for key in ('P','F','reaction','energy'):r[key+'_order']=-float(np.log2(r[key+'_absolute']/pairs[-1][key+'_absolute']))
                pairs.append(r)
            refinement[mode][phase]=dict(pairs=pairs,last_2pct_passed=max(pairs[-1][k+'_relative'] for k in ('P','F','reaction','energy'))<=.02,last_terminal_2pct_passed=max(pairs[-1][k+'_terminal_relative'] for k in ('P','F'))<=.02,order_screen_passed=all(r[k+'_order']>=.5 for r in pairs[1:] for k in ('P','F','reaction','energy')))
    anchor=[]
    for level in LEVELS:
        name='baseline-'+level;cfg=p['configs'][name];rows=data[name][:round(.5/cfg['dt'])]
        old=[json.loads(l) for l in (BASE/'v12/cases'/('F45-'+level)/'steps.jsonl').read_text().splitlines()]
        errors={k:max(abs(a[k]-b[k]) for a,b in zip(rows,old)) for k in ('right_force','elastic','kinetic','stabilization_energy')}
        with np.load(BASE/'v12/cases'/('F45-'+level)/'frames.npz') as z:errors['F']=float(np.max(abs(frames[name]['F'][:21]-z['F'])))
        assert max(errors.values())<1e-12,(level,errors);anchor.append(dict(level=level,errors=errors))
    result=dict(completed=all(c['completed'] for c in checks.values()),physical_checks_passed=all(c['passed'] for c in checks.values()),checks=checks,trajectories=len(checks),steps=sum(c['steps'] for c in checks.values()),refinement=refinement,phase_metrics=phase_metrics,
        history_closure_max=max(r['frozen_F_max'] for r in audits),frozen_history_rate_max=max(r['frozen_eta'] for r in audits),moving_history_rate_max=max(r['moving_eta'] for r in audits),total_history_rate_max=max(r['total_eta'] for r in audits),particle_commit_max=max(r['particle_F_update_max'] for r in audits),snapshot_count=len(audits),baseline_v12_anchor=anchor,
        weak_all_phase_time_2pct_passed=all(r['last_2pct_passed'] and r['last_terminal_2pct_passed'] for r in refinement['weak'].values()),weak_all_phase_orders_passed=all(r['order_screen_passed'] for r in refinement['weak'].values()),
        weak_all_holds_nonincrease=all(phase_metrics['weak-'+l][h]['net_energy_nonincrease'] for l in LEVELS for h in ('hold','final_hold')),default_changed=False,overall_accuracy_accepted=False)
    write(OUT/'summary.json',result)
    for mode in MODES:print(mode,'load P',refinement[mode]['ramp']['pairs'][-1]['P_relative'],'whole P',refinement[mode]['whole']['pairs'][-1]['P_relative'],'hold',phase_metrics[mode+'-fourth']['hold'],'residual',phase_metrics[mode+'-fourth']['residual'])
    print('physical',result['physical_checks_passed'],'history',result['history_closure_max'])

if __name__=='__main__':main()
