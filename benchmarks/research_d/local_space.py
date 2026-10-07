"""D3/D5 profiling of real local residual corrections and CPU orthogonalization."""
import argparse
from pathlib import Path
import time
import numpy as np
import scipy.linalg as la
import warp as wp
from engine.aniso_phase1.research_d.precondition import Schwarz,tensor_blocks,BatchedSchwarz,local_corrections
from engine.aniso_phase1.research_d.identity import write_json,own_sources
from .protocol import problem,CASES,environment,GATES,storage_check


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',required=True);args=parser.parse_args()
    cache,storage=storage_check('/tmp/mpm-lite-research-d-cache');wp.config.kernel_cache_dir=cache;wp.init()
    p=problem(CASES[3]);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'protocol.json',dict(problem=p.identity,source_sha256=own_sources(),environment=environment(),
        storage=storage,gates=GATES,rhs='physical lifting residual at zero free displacement',widths=[3,4,5],
        orthogonalization='CPU pivoted QR; not a new accepted A-space; diagnostic correction basis only'))
    rows=[]
    for width in (3,4,5):
        t=time.perf_counter();A=p.assembled();assembly=time.perf_counter()-t
        M=Schwarz(A,tensor_blocks(p.free_shape,width,1))
        t=time.perf_counter();expected=[(ids,w*(inv@(w*p.rhs[ids]))) for ids,w,inv in M.blocks];cpu_apply=time.perf_counter()-t
        t=time.perf_counter();batch=BatchedSchwarz(M);rd=wp.array(p.rhs,dtype=wp.float64,device='cuda:0');wp.synchronize()
        upload=time.perf_counter()-t
        local_corrections(batch,rd)  # separately excluded warmup
        t=time.perf_counter();actual=local_corrections(batch,rd);gpu_apply=time.perf_counter()-t
        expected_map={tuple(ids):v for ids,v in expected}
        errors=[np.linalg.norm(v-expected_map[tuple(ids)]) for ids,v in actual]
        scale=np.sqrt(sum(v@v for _,v in expected));relative=float(np.linalg.norm(errors)/max(scale,1e-12))
        t=time.perf_counter();Z=np.zeros((p.size,len(actual)))
        for j,(ids,v) in enumerate(actual):Z[ids,j]=v
        basis_assembly=time.perf_counter()-t
        t=time.perf_counter();Q,R,pivots=la.qr(Z,mode='economic',pivoting=True);orthogonalization=time.perf_counter()-t
        residual=float(np.linalg.norm(Q@R-Z[:,pivots])/max(np.linalg.norm(Z),1e-12))
        rows.append(dict(width=width,blocks=len(actual),free_dofs=p.size,assembly_seconds=assembly,
            local_factor_seconds=M.build_seconds,cpu_local_application_seconds=cpu_apply,
            gpu_upload_seconds=upload,gpu_local_application_and_readback_seconds=gpu_apply,
            basis_assembly_seconds=basis_assembly,orthogonalization_seconds=orthogonalization,
            cpu_total_seconds=assembly+M.build_seconds+cpu_apply+basis_assembly+orthogonalization,
            gpu_total_seconds=assembly+M.build_seconds+upload+gpu_apply+basis_assembly+orthogonalization,
            local_relative=relative,qr_relative=residual,device_factor_bytes=batch.memory_bytes,
            cpu_basis_bytes=Z.nbytes+Q.nbytes+R.nbytes,rank=int(np.linalg.matrix_rank(R)),
            passed=relative<=GATES['operator_relative'] and residual<=GATES['operator_relative']))
        np.savez_compressed(out/f'width-{width}.npz',correction_basis=Z,R=R,pivots=pivots)
    write_json(out/'summary.json',dict(passed=all(r['passed'] for r in rows),records=rows,
        decision='measure total setup; local GPU kernels alone do not establish end-to-end benefit',
        CPU_fallbacks=['factor construction','basis assembly','pivoted QR']))

if __name__=='__main__':main()
