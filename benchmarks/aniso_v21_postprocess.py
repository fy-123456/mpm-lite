"""Generate final plots/report only after every requested static study completes."""
import time
from benchmarks.aniso_v21_common import *

def main():
    required=['finish-status.json','static-audits.json','candidate-nonlinear-audits.json','candidate-gain-nonlinear-audit.json'];last=0
    while True:
        pending=[n for n in required if not (OUT/n).exists() or not load(OUT/n).get('completed')]
        if not pending:break
        if time.monotonic()-last>60:print('pending',pending,flush=True);last=time.monotonic()
        time.sleep(10)
    from benchmarks.aniso_v21_figures import main as figures
    from benchmarks.aniso_v21_archive import tests
    from benchmarks.aniso_v21_report import main as report
    for name,fn in [('figures',figures),('test manifest',tests),('report',report)]:print('processing',name,flush=True);fn()
    write(OUT/'postprocess-status.json',dict(completed=True,visual_review_completed=False,archive_pending=True));print('REPORT READY',flush=True)
if __name__=='__main__':main()
