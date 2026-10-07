"""CPU/CUDA material/operator equivalence; shared-device timings are not used."""
import argparse
from pathlib import Path
import subprocess
import numpy as np
import warp as wp
from .run import ROOT, sha, write
from .scenes import source_rule, states, operator, directions, gradient_map, PARAMS, SPACE_ID
from engine.aniso_phase1.research_b.rules import compress
from engine.aniso_phase1.research_b.gpu_adapter import RepresentativeGPUOperator


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    wp.config.kernel_cache_dir='/tmp/mpm-lite-research-b-warp-cache'
    wp.init()
    source=ROOT/'engine/aniso_phase1/research_d/material.py';before=sha(source)
    sources={str(source.relative_to(ROOT)):before,
        'engine/aniso_phase1/research_b/gpu_adapter.py':sha(ROOT/'engine/aniso_phase1/research_b/gpu_adapter.py'),
        'benchmarks/research_b/gpu_probe.py':sha(Path(__file__))}
    rows=[];raw={};passed=True
    for scene in states('development'):
        if not scene['id'].endswith('/hold'):continue
        rule=compress(source_rule(scene['angle'],scene['mixture']),8,'representatives')
        cpu=operator(rule);gpu=RepresentativeGPUOperator(rule,gradient_map,PARAMS,SPACE_ID,9)
        for label,d in directions().items():
            a,b=cpu.evaluate(scene['q'],d),gpu.evaluate(scene['q'],d);wp.synchronize()
            errors={}
            for key,floor in [('U',1e-8),('PK1',1e-5),('force',1e-5),('tangent_action',1e-4)]:
                absolute=float(np.linalg.norm(a[key]-b[key]));relative=absolute/max(float(np.linalg.norm(a[key])),floor)
                errors[key]=dict(absolute=absolute,relative=relative,passed=relative<2e-5 or absolute<1e-9)
                passed &= errors[key]['passed']
                raw[scene['family']+'-'+label+'-cpu-'+key]=a[key]
                raw[scene['family']+'-'+label+'-cuda-'+key]=b[key]
            rows.append(dict(scene=scene['id'],direction=label,samples=len(rule.weight),errors=errors))
    after=sha(source)
    if before!=after:raise RuntimeError('D material backend changed while checking; keep evidence and rerun under a new output')
    device=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
    write(args.output/'cuda-equivalence.json',dict(source_sha256=sources,device=device,warp=wp.__version__,
        dtype='float64',rows=rows,passed=bool(passed),scope='representative sample material energy, PK1, assembled force and exact tangent only',
        general_conditional_A4=False,full_tensor_v22_GPU=False,full_dynamic_CUDA=False,
        timing_claim=False,device_exclusive=False,cache=wp.config.kernel_cache_dir))
    np.savez_compressed(args.output/'cuda-equivalence-raw.npz',**raw)
    print('CUDA_EQUIVALENCE',passed,flush=True)


if __name__=='__main__':main()
