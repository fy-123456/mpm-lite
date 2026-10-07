"""Raw phase reactions, material stress histories, and complete energy ledgers."""
import json,argparse
import numpy as np
from benchmarks.aniso_v20_common import *
from benchmarks.aniso_v17_analysis import comparison
STAGES=dict(ramp=(0.,.5),hold=(.5,.6),unload=(.6,1.1),final_hold=(1.1,1.6))
def rows(folder):return [json.loads(v) for v in (folder/'steps.jsonl').read_text().splitlines()]
def rms(a):return float(np.sqrt(np.mean(np.asarray(a)**2)))
def compare(a,b,V,phase=False):
    ra,rb=rows(a),rows(b);fa=np.load(a/'stress.npz');fb=np.load(b/'stress.npz');ratio=round((fa['time'][1]-fa['time'][0])/(fb['time'][1]-fb['time'][0]));assert len(ra)*ratio==len(rb)
    result=comparison(fa['P'],fb['P'][::ratio],V,fa['time']);ar=np.array([r['reaction_N'] for r in ra]);br=np.array([r['reaction_N'] for r in rb])[ratio-1::ratio];result.update(raw_reaction_absolute_N=rms(ar-br),raw_reaction_relative=rms(ar-br)/rms(br),raw_reference_rms_N=rms(br))
    if phase:
        dt=ra[0]['dt'];result['stages']={}
        for name,(lo,hi) in STAGES.items():
            i,j=round(lo/dt),round(hi/dt);stress=comparison(fa['P'][i:j+1],fb['P'][::ratio][i:j+1],V,fa['time'][i:j+1]);stress.update(raw_reaction_absolute_N=rms(ar[i:j]-br[i:j]),raw_reaction_relative=rms(ar[i:j]-br[i:j])/rms(br[i:j]),raw_reference_rms_N=rms(br[i:j]));result['stages'][name]=stress
    return result

def ledger(folder):
    rr=rows(folder);status=load(folder/'status.json');keys=['boundary_work_J','constraint_kinetic_loss_J','metric_change_J','kinetic_force_work_defect_J','potential_quadrature_error_J'];out=dict(completed=status['completed'],steps=len(rr),dt=rr[0]['dt'],terminal=rr[-1],sums_J={k:sum(r[k] for r in rr) for k in keys},maxima={k:max(abs(r[k]) for r in rr) for k in ['budget_defect_J','history_commit_max','endpoint_velocity_constraint','grip_velocity_error','material_null_constraint','endpoint_free_impulse_error']},static_checks=status['static_gates'])
    s=out['sums_J'];out['closure_J']=rr[-1]['total_J']-s['boundary_work_J']+s['constraint_kinetic_loss_J']-s['metric_change_J']-s['kinetic_force_work_defect_J']-s['potential_quadrature_error_J'];assert abs(out['closure_J'])<1e-12
    out['roundoff_merit_accepts']=sum(r.get('roundoff_merit_accepts',0) for r in rr)
    out['stages']={};dt=rr[0]['dt']
    for name,(lo,hi) in STAGES.items():
        i,j=round(lo/dt),round(hi/dt)
        if j>len(rr):continue
        part=rr[i:j];entry={k:sum(r[k] for r in part) for k in keys};entry['energy_change_J']=part[-1]['total_J']-(rr[i-1]['total_J'] if i else 0.);entry['raw_reaction_rms_N']=rms([r['reaction_N'] for r in part]);entry['stress_rms_Pa']=rms([r['stress_rms_Pa'] for r in part]);out['stages'][name]=entry
    return out

def short():
    batch=load(OUT/'snapshot-batch.json');assert batch['completed'];_,e,_,_,_=controlled_case();result={}
    for t in (.85,1.4):
        for kind in ('sampled','gauss3'):
            for geometry in ('fixed','moving'):
                name=f'{t:.2f}-{kind}-{geometry}';pairs=[compare(OUT/'snapshot-runs'/f'{name}-{i}',OUT/'snapshot-runs'/f'{name}-{i+1}',e.V) for i in range(2)];result[name]=dict(pairs=pairs,last_stress_passed=max(pairs[-1]['relative'],pairs[-1]['terminal_relative'])<.02,last_raw_reaction_passed=pairs[-1]['raw_reaction_relative']<.02)
    geometry_pairs={}
    for t in (.85,1.4):
        for kind in ('sampled','gauss3'):
            a=OUT/'snapshot-runs'/f'{t:.2f}-{kind}-fixed-2';b=OUT/'snapshot-runs'/f'{t:.2f}-{kind}-moving-2';geometry_pairs[f'{t:.2f}-{kind}']=compare(a,b,e.V)
    write(OUT/'snapshot-acceptance.json',dict(completed=True,cases=result,fixed_moving_same_dt=geometry_pairs,initial_integrals={r['name']:r['initial'] for r in batch['records'] if r['name'].endswith('-fixed-0')},no_initial_step_removed=True,no_reaction_smoothing=True))

def cycle():
    dest=OUT/'fast-cycle';manifest=load(OUT/'completed-case-paths.json');assert manifest['completed'];_,e,_,_,_=controlled_case();cases={}
    paths={n:ROOT/v['path'] for n,v in manifest['cases'].items()}
    for p in load(dest/'cycle-protocol.json')['cases']:
        folder=paths[p['name']];cases[p['name']]=ledger(folder)
    complete=all(r['completed'] for r in cases.values());primary=all(cases[f'gauss3-condensed-L{i}']['completed'] for i in range(4));pairs=[]
    if primary:
        pairs=[compare(paths[f'gauss3-condensed-L{i}'],paths[f'gauss3-condensed-L{i+1}'],e.V,True) for i in range(3)]
    bridges={}
    for name in ('sampled-condensed-ablation','gauss3-original-ablation'):
        if cases[name]['completed'] and cases['gauss3-condensed-L2']['completed']:bridges[name]=compare(paths[name],paths['gauss3-condensed-L2'],e.V,True)
    if cases['gauss3-condensed-L2']['completed']:bridges['v19-sampled-original']=compare(BASE/'v19/cases/cycle-L2',paths['gauss3-condensed-L2'],e.V,True)
    if (OUT/'cases/gauss3-condensed-L0/status.json').exists() and load(OUT/'cases/gauss3-condensed-L0/status.json')['completed']:bridges['same_implementation_full_cycle']=compare(OUT/'cases/gauss3-condensed-L0',paths['gauss3-condensed-L0'],e.V,True)
    stress_pass=primary and max([pairs[-1]['relative'],pairs[-1]['terminal_relative']]+[r[k] for r in pairs[-1]['stages'].values() for k in ('relative','terminal_relative')])<.02
    reaction_pass=primary and max([pairs[-1]['raw_reaction_relative']]+[r['raw_reaction_relative'] for r in pairs[-1]['stages'].values()])<.02
    loss=[cases[f'gauss3-condensed-L{i}']['sums_J']['constraint_kinetic_loss_J'] for i in range(4)];loss_decreases=all(a>=b for a,b in zip(loss[:-1],loss[1:]));write(OUT/'cycle-acceptance.json',dict(completed=complete,primary_completed=primary,cases=cases,pairs=pairs,same_dt_comparisons=bridges,stress_time_passed=stress_pass,raw_reaction_time_passed=reaction_pass,constraint_loss_decreases=loss_decreases,passed=complete and stress_pass and reaction_pass and loss_decreases,spatial_acceptance_separate=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['short','cycle']);a=p.parse_args();short() if a.action=='short' else cycle()
