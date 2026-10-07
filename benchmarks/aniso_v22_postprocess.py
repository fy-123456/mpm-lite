"""Generate reviewable outputs only after all prescribed studies finish."""
import time
from benchmarks.aniso_v22_common import *
from benchmarks.aniso_v22_report import main as report

def main():
    dependencies=['reference-summary.json','final-spatial-acceptance.json','static-audits.json','nonlinear-audits.json','multiscale-audit.json']
    while not all((OUT/n).exists() and load(OUT/n).get('completed') for n in dependencies):time.sleep(10)
    report();write(OUT/'postprocess-status.json',dict(completed=True,visual_review_completed=False,archive_pending=True));print('READY FOR VISUAL REVIEW',flush=True)
if __name__=='__main__':main()
