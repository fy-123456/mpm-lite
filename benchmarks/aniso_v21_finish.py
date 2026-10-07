"""Publish final metrics after completing both planned reference levels."""
import time,shutil
from concurrent.futures import ThreadPoolExecutor
from benchmarks.aniso_v21_common import *

def main():
    needed=['reference-completed-summary.json','final-spatial-acceptance.json','reference-self-checks.json']
    while not all((OUT/n).exists() and load(OUT/n)['completed'] for n in needed):time.sleep(10)
    folder=OUT/'prior-final-level0';folder.mkdir(exist_ok=False)
    for name in ['final-spatial-acceptance.json','reference-self-checks.json','reference-h_q3.json','reference-h_q4.json','final-candidate-selection.json']:shutil.move(OUT/name,folder/name)
    from benchmarks.aniso_v21_final_metrics import main as metrics
    from benchmarks.aniso_v21_reference_audit import main as audit
    print('parallel final candidate metrics and reference h/p checks',flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=[pool.submit(metrics,'reference-completed-summary.json'),pool.submit(audit,'reference-completed-summary.json','reference-extension/level0-q3.npz','reference-extension/level0-q4.npz')]
        for result in pending:result.result()
    write(OUT/'finish-status.json',dict(completed=True,final_reference='reference-extension/level1-q4.npz',earlier_metrics_preserved='prior-final-level0'));print('FINAL METRICS READY',flush=True)
if __name__=='__main__':main()
