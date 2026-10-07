"""Prespecified final spaces evaluated on an independent, common reference."""
import argparse,time
from benchmarks.aniso_v22_common import *

def main(training=False):
    plans={f'q{p}-local144':(OUT/f'space/q{p}/round6.npz',OUT/f'space/q{p}/summary.json') for p in (2,3,4)}
    plans['q4-multiscale144']=(OUT/'multiscale/space/q4/round6.npz',OUT/'multiscale/space/q4/summary.json')
    plans['v21-local144']=(BASE/'v21/adaptive/gain/round6.npz',None)
    plans['q2-full']=(BASE/'v21/space-limit/graded-q2.npz',None)
    for p in (3,4):plans[f'q{p}-full']=(OUT/f'space-limit/q{p}.npz',OUT/'space-limit.json')
    protocol=OUT/'metrics-protocol.json'
    if not protocol.exists():write(protocol,dict(cases={n:str(p.relative_to(ROOT)) for n,(p,_) in plans.items()},candidate_selection='Fixed final 144-scalar-function budgets for all four spaces; no held-out best-round selection.',reference='reference/level1-q4.npz',regions='Same v21 masks including hard grip volumes. No smoothing and no corner exclusion.',source_sha256={n:sha(ROOT/n) for n in ['benchmarks/aniso_v22_metrics.py','engine/aniso_phase1/tensor_metrics.py']}))
    if training:refpath=BASE/'v21/reference-extension/level1-q4.npz';refresult=load(refpath.with_suffix('.json'))
    else:
        while not (OUT/'reference-summary.json').exists() or not load(OUT/'reference-summary.json')['completed']:time.sleep(10)
        refpath=OUT/'reference/level1-q4.npz';refresult=load(refpath.with_suffix('.json'))
    ref=read_field(refpath);records={};dest=OUT/('training-acceptance.json' if training else 'final-spatial-acceptance.json')
    for name,(path,dependency) in plans.items():
        while not path.exists() or dependency is not None and (not dependency.exists() or not load(dependency)['completed']):time.sleep(10)
        v=compare(read_field(path),ref)
        if name=='v21-local144':reaction=load(BASE/'v21/adaptive/gain/round6.json')['materials']['F45']['reaction_N']
        elif name.endswith('-full'):
            reaction=load(path.with_suffix('.json'))['reaction_N']+load(BASE/'v20/local-relaxation.json')['patch_only_minimum_reaction_N']
        else:reaction=load(path.with_suffix('.json'))['materials']['F45']['reaction_N']
        v.update(reaction_N=reaction,reaction_relative=abs(reaction-refresult['reaction_N'])/abs(refresult['reaction_N']),source_sha256=sha(path));records[name]=v
        write(dest,dict(completed=False,reference=str(refpath.relative_to(ROOT)),reference_sha256=sha(refpath),cases=records));print('METRIC',name,{k:(w['stress_relative'],w['fiber_strain_relative']) for k,w in v['regions'].items()},flush=True)
    write(dest,dict(completed=True,reference=str(refpath.relative_to(ROOT)),reference_sha256=sha(refpath),reference_self_checks_passed=False if training else load(OUT/'reference-summary.json')['all_stress_and_fiber_passed'],continuum_error_bound_proved=False,cases=records,no_stress_smoothing=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',action='store_true');a=p.parse_args();main(a.training)
