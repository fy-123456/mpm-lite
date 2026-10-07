"""Finish evidence products after every actual trajectory and oracle completes."""
import time
from benchmarks.aniso_v20_common import *

def main():
    required=['completed-case-paths.json','independent-audits.json','runtime-quadrature.json','space-diagnosis.json','local-stiffness-checks.json'];last=0
    while True:
        pending=[n for n in required if not (OUT/n).exists() or not load(OUT/n).get('completed')]
        if not pending:break
        if time.monotonic()-last>60:print('pending',pending,flush=True);last=time.monotonic()
        time.sleep(10)
    from benchmarks.aniso_v20_analysis import cycle
    from benchmarks.aniso_v20_ablation import main as ablation
    from benchmarks.aniso_v20_condensation import main as condensation
    from benchmarks.aniso_v20_figures import main as figures
    from benchmarks.aniso_v20_archive import tests
    from benchmarks.aniso_v20_report import main as report
    for name,fn in [('acceptance',cycle),('ablation',ablation),('mode audit',condensation),('figures',figures),('test manifest',tests),('report',report)]:
        print('processing',name,flush=True);fn()
    write(OUT/'postprocess-status.json',dict(completed=True,archive_pending_visual_review=True));print('REPORT READY',flush=True)
if __name__=='__main__':main()
