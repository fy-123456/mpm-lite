"""Add overlapping half-span corrections to coordinate diagonal deformation.

The three new patches remain bounded, have zero artificial-face trace, and
never cover the entire free span. Physical cross-section boundaries are natural.
Their vectors solve the original local residual, never a reference fit.
"""
import numpy as np
from benchmarks.aniso_v22_common import *
from benchmarks import aniso_v22_space as runner
from engine.aniso_phase1.stress_local_space import patches

def main():
    evidence=load(OUT/'limit-training.json');assert evidence['completed'];folder=OUT/'multiscale';folder.mkdir(exist_ok=False)
    ps=patches()
    for lo,hi in ((.25,.5625),(.4375,.75),(.34375,.65625)):
        ps.append(dict(center=[(lo+hi)/2,.5,.5],lo=[lo,.375,.375],hi=[hi,.625,.625]))
    names=['benchmarks/aniso_v22_multiscale.py','benchmarks/aniso_v22_space.py','engine/aniso_phase1/high_order_space.py','engine/aniso_phase1/stress_local_space.py','tests/test_aniso_v22_space.py']
    protocol=load(OUT/'space-protocol.json');protocol.update(source_sha256={n:sha(ROOT/n) for n in names},degrees=[4],patches=ps,reason='Q4 full-space interior stress is below 1%, while limited local corrections remain much less accurate. Test larger bounded supports at the SAME 144-function budget.',evidence_sha256=sha(OUT/'limit-training.json'),reference_field_not_copied=True,largest_patch_free_span_fraction=.625)
    write(folder/'space-protocol.json',protocol);runner.OUT=folder;runner.patches=lambda:ps;runner.run(4)
if __name__=='__main__':main()
