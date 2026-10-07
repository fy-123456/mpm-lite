"""Independent-process AB/BA full static solves on a frozen search aid.

The GPU-built initial search matrix is explicitly charged to cold totals.
This does not certify a from-scratch CPU-only construction or production use.
"""
import argparse,gc,json,os,platform,resource,subprocess,time,threading
from pathlib import Path
import numpy as np
import warp as wp
from benchmarks.research_d.stage2.bootstrap import PARENT,PIN
from benchmarks.research_d.stage2.validate import probes
from engine.aniso_phase1.research_d.frozen_inputs import load_frozen_inputs
from engine.aniso_phase1.research_d.stage2.contracts import sha
from engine.aniso_phase1.research_d.stage2.gpu_operator import GPUOperator
from engine.aniso_phase1.research_d.stage2.solver import equilibrate


def measure(folder,index):
    folder=Path(folder);p=json.loads((folder/'protocol.json').read_text())
    state=json.loads((folder/'correctness-acceptance.json').read_text())
    if not state['passed']:raise ValueError('correctness must pass before timing')
    wp.config.kernel_cache_dir='/root/autodl-tmp/mpm-lite-research-d/stage2/warp-cache'
    device_before=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
    if device_before:raise RuntimeError('GPU occupied; reschedule exclusive timing')
    root=Path.cwd();started=time.perf_counter();s,_,_,_=load_frozen_inputs(root/PARENT,root,expected_sha256=PIN)
    shared_load=time.perf_counter()-started
    with np.load(folder/'search-matrix.npz') as z:h=z['hessian']
    build=json.loads((folder/'curvature.json').read_text())['build_seconds']
    pilot=json.loads((folder/'gpu-pilot.json').read_text())
    first_setup=pilot['build_seconds']+pilot['response_and_tangent_seconds']
    costs=[];samples=[];stop=threading.Event()
    def monitor():
        while not stop.wait(.5):
            sample=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
            samples.append(dict(time=time.time(),gpu=sample,load=os.getloadavg()))
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    try:
        order=['cpu','gpu'] if index%2==0 else ['gpu','cpu']
        for backend in order:
            begun=time.perf_counter();construct=time.perf_counter()
            op=GPUOperator(s) if backend=='gpu' else None
            construct=time.perf_counter()-construct
            evaluate=op.evaluate if op is not None else lambda q:s.response(q,order=6)
            # Cold solve contains module loading, all mapping, transfer,
            # synchronization, original-force stopping and output probes.
            for label in ('cold','warm','repeat_for_AA'):
                t=time.perf_counter();q,r,info=equilibrate(evaluate,s.q0,h)
                probe=probes(s,q)
                if op is not None:wp.synchronize_device(op.device)
                duration=time.perf_counter()-t
                if not info['converged']:raise RuntimeError('timed solve did not converge')
                costs.append(dict(backend=backend,label=label,seconds=duration,construction_seconds=construct,
                    total_cold_including_shared_setup=duration+construct+shared_load+build+first_setup,
                    residual=info['trace'][-1]['residual_N'],iterations=len(info['trace'])-1,
                    energy=r['U'],PK1_norm=float(np.linalg.norm(probe['PK1'])),
                    output_evaluation_included=True,transfer_and_sync_included=True))
            del op,evaluate;gc.collect()
    finally:stop.set();worker.join()
    result=dict(process=index,pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),measurement_class='shared_host_diagnostic; exclusive GPU',other_processes=subprocess.check_output(['ps','-eo','pid,pcpu,comm','--sort=-pcpu'],text=True).splitlines()[:15],order=order,costs=costs,shared_parent_load_seconds=shared_load,
        search_matrix_build_seconds=build,search_matrix_sha256=sha(folder/'search-matrix.npz'),
        host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu_load_samples=samples,
        scope='fixed archived q0 and fixed 0.005 m lift; sealed GPU-built search aid; same physical error and residual',
        cold_includes_jit_cache_load=True,initial_compile_and_first_operator_seconds=pilot['response_and_tangent_seconds'],
        shared_initial_setup_seconds=first_setup,
        no_from_scratch_cpu_only_claim=True)
    (folder/f'performance-{index}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('process','order','costs')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--index',type=int,required=True);a=p.parse_args();measure(a.folder,a.index)
