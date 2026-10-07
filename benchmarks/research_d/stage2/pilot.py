import json,os,time,resource
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_d.frozen_inputs import load_frozen_inputs
from benchmarks.research_d.stage2.bootstrap import PARENT,PIN
from benchmarks.research_d.stage2.protocol import freeze
ROOT=Path.cwd();OUT=ROOT/'docs/results/parallel-v22-stage2/D/20260930-D-common-static'
OUT.mkdir(parents=True,exist_ok=True)
s,_,_,audit=load_frozen_inputs(ROOT/PARENT,ROOT,expected_sha256=PIN)
(OUT/'baseline-readonly-check.json').write_text(json.dumps(audit,indent=2))
p,v=freeze(OUT,ROOT/PARENT,s)
t=time.perf_counter();r=s.response(s.q0,s.restrict(v['direction_mixed']),order=6);elapsed=time.perf_counter()-t
np.savez(OUT/'pilot-reference.npz',**r)
record=dict(seconds=elapsed,energy=r['U'],min_detF=r['min_detF'],free_residual=float(np.linalg.norm(r['force'])),material_points=r['material_calls'],host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,pid=os.getpid())
(OUT/'resource-pilot.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
