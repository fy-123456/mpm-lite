"""One static cost probe of the registered batch-four optimization opportunity."""
import argparse,time,subprocess
import numpy as np
import warp as wp
from .provenance import *
from .continuous import setup
from .runtime import update
from benchmarks.research_phase_stress_next.time_study import history
from engine.aniso_phase1.research_cost_phase_next.pressure import cell_geometry_kernel

def probe(run):
    run=Path(run);mutable(run);before=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],capture_output=True,text=True).stdout
    c,m,ident,bridge,parent=setup(run);g=c.geometry;q=history(APP/'cases/boundary32-h')[8]['state'].q;wp.synchronize_device(m.device);start=time.perf_counter();F=g.field(q);cof=wp.empty_like(F);J=wp.empty(g.count,dtype=wp.float64,device=m.device);unused=wp.empty_like(J);bad=wp.zeros(1,dtype=wp.int32,device=m.device);wp.launch(cell_geometry_kernel,dim=g.count,inputs=[F,cof,J,unused,bad,.1],device=m.device)
    if bad.numpy()[0]:raise ValueError('cost probe geometry invalid')
    jj=J.numpy();field=time.perf_counter()-start;records=[]
    for mp,(a,b,layout,weights) in zip(g.local_maps,g.local):
        t=time.perf_counter();d=mp.gradient_adjoint(cof[a:b],layout,weights);queued=time.perf_counter();grad=d.numpy().reshape(m.parent.ndof,3);downloaded=time.perf_counter();reduced=m.reduction.P.T@grad;end=time.perf_counter()
        records.append(dict(enqueue_s=queued-t,download_and_remaining_GPU_upper_s=downloaded-queued,CPU_contraction_s=end-downloaded,adjoint_s=end-t,small_output_bytes=grad.nbytes))
    t=time.perf_counter();g.assembler.assemble(F);wp.synchronize_device(m.device);rt0=time.perf_counter()-t;total=time.perf_counter()-start
    # Optimistic upper bound: all download waits + CPU products disappear. GPU
    # adjoint kernels, allocation and enqueue are still required by batch-four.
    removable=sum(r['download_and_remaining_GPU_upper_s']+r['CPU_contraction_s'] for r in records);geom_fraction=max(r['B']['geometry_inclusive_s']/r['B']['advance_s'] for r in read(APP/'S4/boundary32/paired-performance.json')['records']);upper=geom_fraction*removable/total
    shared=len(before.strip().splitlines())>1;eligible=upper>=.05 and not shared;metadata=g.local_bytes+m.reduction.P.nbytes;temporary=4*m.parent.ndof*3*8+4*m.reduction.P.shape[1]*3*8
    result=dict(status='registered' if eligible else 'not_triggered',candidate='G1',candidate_count=1,batch_size=4,layout='four small parent gradients -> same P.T -> four reduced gradients; no retained nodal duals',source_step=8,source_sha256=read(run/'S0/state-contract.json')['states']['8']['sha256'],records=records,field_s=field,RT0_s=rt0,total_geometry_s=total,optimistic_removable_s=removable,optimistic_whole_step_gain_upper=upper,inherited_geometry_fraction=geom_fraction,prediction_note='optimistic upper bound includes GPU waits which batching cannot remove; not a measured speedup',metadata_total_bytes=metadata,new_temporary_bytes=temporary,metadata_cap_bytes=min(256*2**20,int(.02*m.operator.memory_budget.initial_free)),new_temporary_cap_bytes=64*2**20,resources=m.operator.memory_budget.report(),GPU_before=before,shared_GPU=shared,reason='eligible for exactly one candidate' if eligible else ('shared GPU prevents performance certification' if shared else 'even optimistic removable wait/contraction bound predicts less than 5% whole-step benefit'),new_dynamic_attempts=0)
    register(run,'S2/optimization-protocol.json',result);print('G1_PREFLIGHT',result['status'],upper,flush=True);update(run,f'S2 G1预评估：{result["status"]}；即使把下载等待和CPU回缩全部去掉，整步乐观收益上界约{upper:.2%}。'+result['reason'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):probe(a.run)
