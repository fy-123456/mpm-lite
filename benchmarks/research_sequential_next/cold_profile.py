"""One fresh-process, empty-kernel-cache cold trial for N07."""
from pathlib import Path
import argparse
import copy
import time
from .provenance import read,write,serial_lock,source_files,utc,verify


def cold(run,order):
    started=time.perf_counter();run=Path(run);destination=run/'N07'/f'cold-q{order}'
    if destination.exists():raise ValueError('cold trial must use a previously absent cache directory')
    destination.mkdir()
    import warp as wp
    wp.config.kernel_cache_dir=str(destination.resolve()/'warp-cache')
    from .model_package import load_reduction
    from .checkpoint import GenerationStore
    from engine.aniso_phase1.research_sequential_next.profiling import ProfiledModel
    from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
    verify(run);reduction=load_reduction(run)
    folder=run/'cases/gpu-q7-dt0025';cfg=read(folder/'execution-protocol.json');cfg['material_order']=order
    state=next(x['state'] for x in GenerationStore(folder,read(folder/'identity.json')).history() if abs(x['state'].time-.45)<1e-12)
    before_build=time.perf_counter();model=ProfiledModel(reduction,order=order,device='cuda:0');build=time.perf_counter()-before_build
    state.child_states['identity']=copy.deepcopy(model.identity)
    before_step=time.perf_counter();stepper=ValidatedAVF(model,cfg,state)
    rows=[stepper.step(.025),stepper.step(.025)];advance=time.perf_counter()-before_step
    result=dict(utc=utc(),source_sha256=source_files(),order=order,initial_time=.45,final_time=stepper.state.time,
        empty_kernel_cache=True,build_seconds=build,initialization_and_steps_seconds=advance,
        entry_wall_seconds=time.perf_counter()-started,partitions_seconds=dict(model.operator.timings),
        rows=rows,finite_scene=True,scope='one cold trial; external process wall seconds recorded by the shell timer')
    write(destination/'result.json',result)
    print({k:v for k,v in result.items() if k not in ('source_sha256','rows')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--order',type=int,choices=[5,7],required=True);a=p.parse_args()
    with serial_lock(a.run):cold(a.run,a.order)
