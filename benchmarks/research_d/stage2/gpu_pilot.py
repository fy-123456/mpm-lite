import json,time,os
from pathlib import Path
import numpy as np
import warp as wp
from engine.aniso_phase1.research_d.frozen_inputs import load_frozen_inputs
from engine.aniso_phase1.research_d.stage2.gpu_operator import GPUOperator
from benchmarks.research_d.stage2.bootstrap import PARENT,PIN
wp.config.kernel_cache_dir=os.environ.get('WARP_CACHE_PATH','/root/autodl-tmp/mpm-lite-research-d/stage2/warp-cache')
root=Path.cwd();out=root/'docs/results/parallel-v22-stage2/D/20260930-D-common-static'
s,_,_,_=load_frozen_inputs(root/PARENT,root,expected_sha256=PIN)
with np.load(out/'vectors.npz') as z:d=s.restrict(z['direction_mixed'])
t=time.perf_counter();gpu=GPUOperator(s);print('build',gpu.build_seconds,flush=True)
t0=time.perf_counter();r=gpu.evaluate(s.q0,d);elapsed=time.perf_counter()-t0
with np.load(out/'pilot-reference.npz') as cpu:
 errors={k:float(np.linalg.norm(r[k]-cpu[k])/max(np.linalg.norm(cpu[k]),1e-12)) for k in ('U','full_force','tangent_action')}
np.savez(out/'gpu-pilot.npz',**r)
record=dict(build_seconds=gpu.build_seconds,response_and_tangent_seconds=elapsed,total_seconds=time.perf_counter()-t,errors=errors,min_detF=r['min_detF'])
(out/'gpu-pilot.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
