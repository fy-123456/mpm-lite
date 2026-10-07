"""Parallel independent field comparisons once final reference solves exist.

The reference solver independently finishes its p comparison. No report is
published until that comparison, h comparisons, and all candidate metrics end.
"""
import time,shutil,json
from concurrent.futures import ThreadPoolExecutor
from benchmarks.aniso_v21_common import *

def completed(path):
    try:return path.exists() and load(path).get('completed',load(path).get('passed',False))
    except json.JSONDecodeError:return False

def main():
    finalpath=OUT/'reference-extension/level1-q4.npz';status=finalpath.with_suffix('.json')
    while not (finalpath.exists() and completed(status) and completed(OUT/'final-spatial-acceptance.json') and completed(OUT/'reference-self-checks.json')):time.sleep(5)
    folder=OUT/'prior-final-level0';folder.mkdir(exist_ok=False)
    for name in ['final-spatial-acceptance.json','reference-self-checks.json','reference-h_q3.json','reference-h_q4.json','final-candidate-selection.json']:shutil.move(OUT/name,folder/name)
    write(OUT/'reference-solution-ready.json',dict(completed=True,latest_reference='reference-extension/level1-q4.npz',stress_and_fiber_reference_passed=False,self_checks_pending=True,source_sha256=sha(finalpath),scope='Completed field for independent comparisons; not reference certification.'))
    from benchmarks.aniso_v21_final_metrics import main as metrics
    def hcheck(p):
        previous=OUT/f'reference-extension/level0-q{p}.npz';current=OUT/f'reference-extension/level1-q{p}.npz';v=compare(read_field(previous),read_field(current));v.update(previous_sha256=sha(previous),current_sha256=sha(current));write(OUT/f'reference-h_q{p}.json',v);print('hcheck',p,v['regions'],flush=True);return v
    print('parallel candidate, Q3 h and Q4 h comparisons',flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        candidate=pool.submit(metrics,'reference-solution-ready.json');a=pool.submit(hcheck,3);b=pool.submit(hcheck,4);candidate.result();h3=a.result();h4=b.result()
    while not completed(OUT/'reference-completed-summary.json'):time.sleep(5)
    source=load(OUT/'reference-completed-summary.json');pairs={'h_q3':h3,'h_q4':h4,'p_q3_q4':source['pairs']['level1-cross']};checks={r:dict(stress_passed=all(p['regions'][r]['stress_relative']<.02 for p in pairs.values()),fiber_passed=all(p['regions'][r]['fiber_strain_relative']<.02 for p in pairs.values())) for r in ('global','grip','interior','deep_interior')};passed=all(v['stress_passed'] and v['fiber_passed'] for v in checks.values());write(OUT/'reference-self-checks.json',dict(completed=True,pairs=pairs,regions=checks,all_stress_and_fiber_passed=passed,previous_q3='reference-extension/level0-q3.npz',previous_q4='reference-extension/level0-q4.npz',latest_reference=str(finalpath.relative_to(ROOT)),latest_reference_sha256=sha(finalpath),continuum_error_bound_proved=False,resource_limit_reached=False))
    final=load(OUT/'final-spatial-acceptance.json');final['reference_self_checks_passed']=passed;write(OUT/'final-spatial-acceptance.json',final);write(OUT/'finish-status.json',dict(completed=True,final_reference='reference-extension/level1-q4.npz',earlier_metrics_preserved='prior-final-level0',independent_comparisons_parallel=True));print('FINAL METRICS READY',flush=True)
if __name__=='__main__':main()
