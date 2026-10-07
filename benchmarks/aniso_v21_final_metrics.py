"""Held-out final reference evaluation after choices are frozen on training data."""
import time
from benchmarks.aniso_v21_common import *
from benchmarks.aniso_v21_metrics import compare_many

def main(summary_name='reference-extension-summary.json'):
    while not (OUT/'adaptive/gain/summary.json').exists():time.sleep(10)
    paths={f'gain/round{k}':OUT/'adaptive/gain'/f'round{k}.npz' for k in range(1,7)};fields={n:read_field(p) for n,p in paths.items()}
    for stage,reference,p in [('training',BASE/'v11-reference-q3/cases/local2.npz',3),('validation',OUT/'reference/local3-q4.npz',None)]:
        target=OUT/f'gain-{stage}.json'
        if target.exists():continue
        values=compare_many(fields,read_field(reference,p));R=load(reference.with_suffix('.json'))['reaction_N'];cases={}
        for n,regions_ in values.items():
            r=load(paths[n].with_suffix('.json'))['materials']['F45'];cases[n]=dict(scalar_local_dofs=r['scalar_local_dofs'],reaction_relative=abs(r['reaction_N']-R)/abs(R),regions=regions_,source_sha256=sha(paths[n]));print(stage,n,cases[n],flush=True)
        write(target,dict(completed=True,cases=cases,reference=str(reference.relative_to(ROOT)),reference_sha256=sha(reference),held_out=stage=='validation',reference_continuum_certified=False))
    while not all((OUT/n).exists() and load(OUT/n)['completed'] for n in ['space-training.json','adaptive-training.json',summary_name,'space-limit.json']):time.sleep(10)
    selected={};training=load(OUT/'space-training.json')['cases']
    for mesh in ('coarse','graded'):
        selected[mesh+'/baseline']=OUT/'space'/mesh/'F45-stress-00.npz'
        for order in ('stress','geometric'):
            names=[n for n in training if n.startswith(mesh+'/F45-'+order) and not n.endswith('-00')];name=min(names,key=lambda n:training[n]['regions']['global']['stress_relative']);selected[name]=OUT/'space'/f'{name}.npz'
    for filename,orders in [('adaptive-training.json',['stress','geometric']),('gain-training.json',['gain'])]:
        rows=load(OUT/filename)['cases']
        for order in orders:
            name=min([n for n in rows if n.startswith(order+'/')],key=lambda n:rows[n]['regions']['global']['stress_relative']);selected['adaptive/'+name]=OUT/'adaptive'/f'{name}.npz'
    selected['adaptive/stress/round6']=OUT/'adaptive/stress/round6.npz';selected['full-graded-Q2-limit']=OUT/'space-limit/graded-q2.npz';selected['v20-local150']=OUT/'legacy-local150.npz';write(OUT/'final-candidate-selection.json',dict(selection='Minimum global stress difference to archived training Q3 within each family, plus the predeclared final-round matched-budget control; final reference is not used for selection.',cases={n:str(p.relative_to(ROOT)) for n,p in selected.items()},used_final_reference_for_selection=False))
    reference=OUT/load(OUT/summary_name)['latest_reference'];fields={n:read_field(p) for n,p in selected.items()};metrics=compare_many(fields,read_field(reference));R=load(reference.with_suffix('.json'))['reaction_N'];cases={}
    for name,regions_ in metrics.items():
        record=load(selected[name].with_suffix('.json'))
        if 'materials' in record:record=record['materials']['F45']
        reaction=record['reaction_N']
        if name=='full-graded-Q2-limit':reaction+=load(OUT/'space-limit.json')['unchanged_patch_minimum_reaction_N']
        cases[name]=dict(regions=regions_,reaction_relative=abs(reaction-R)/abs(R),reaction_N=reaction,scalar_local_dofs=record.get('scalar_local_dofs'),source_sha256=sha(selected[name]),stress_passed=all(regions_[r]['stress_relative']<.02 for r in ('global','grip','interior')),fiber_passed=all(regions_[r]['fiber_strain_relative']<.02 for r in ('global','grip','interior')));print('final',name,cases[name],flush=True)
    write(OUT/'final-spatial-acceptance.json',dict(completed=True,cases=cases,reference=str(reference.relative_to(ROOT)),reference_sha256=sha(reference),reference_self_checks_passed=load(OUT/summary_name)['stress_and_fiber_reference_passed'],continuum_error_bound_proved=False,quadrature_order=read_field(reference)[1]+1,no_stress_smoothing=True,scope='Linear static spatial comparisons, separate from the unchanged v20 dynamic candidate.'))
if __name__=='__main__':main()
