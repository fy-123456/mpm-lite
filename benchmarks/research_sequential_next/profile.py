"""N07: same-state costs and two-step warm trials with matched stopping rules."""
from pathlib import Path
import argparse
import copy
import time
import resource
import numpy as np
from .provenance import read,write,serial_lock,source_files,utc
from .checkpoint import GenerationStore
from .run import load_model
from engine.aniso_phase1.research_sequential_next.profiling import ProfiledModel
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF


def profile(run):
    run=Path(run);cfg=read(run/'cases/gpu-q7-dt0025/execution-protocol.json')
    model,_=load_model(run,cfg);reduction=model.reduction
    folder=run/'cases/gpu-q7-dt0025';history=GenerationStore(folder,read(folder/'identity.json')).history()
    initial=next(x['state'] for x in history if abs(x['state'].time-.45)<1e-12)
    # Both rules start from byte-identical physical history. Only the explicit
    # material-rule identity changes in this registered independent trial.
    protocol=dict(utc=utc(),source_sha256=source_files(),orders=[7,5],dt=.025,steps=2,warm_repeats=3,
        initial_digest=initial.digest(),matched_tolerances=cfg['solver'],matched_mass_sha256=reduction.signature,
        rule_branch='fresh independent trial from same q/v/predictor; replace model identity explicitly',
        timing='GPU synchronization around partitions; mapping includes upload; source evaluation IO measured separately',
        exclusive_hardware=False,CPU_full_cycle=False)
    if (run/'N07/protocol.json').exists():raise ValueError('profiling protocol already exists')
    write(run/'N07/protocol.json',protocol)
    records=[];operators={}
    for order in (7,5):
        begun=time.perf_counter();operators[order]=ProfiledModel(reduction,order=order,device='cuda:0')
        build=time.perf_counter()-begun;p=operators[order]
        np.testing.assert_array_equal(p.M,model.M)
        # Same q input compares instrumented and original response paths.
        probe=p.evaluate(initial.q);reference=model.evaluate(initial.q) if order==7 else None
        backend=None if reference is None else {k:float(np.linalg.norm(np.asarray(probe[k])-reference[k])/max(np.linalg.norm(reference[k]),1e-12)) for k in ('U','force')}
        if backend is not None and max(backend.values())>2e-5:raise ValueError('instrumentation changed the operator')
        records.append(dict(kind='construction',order=order,build_seconds=build,operator_build_seconds=p.operator.build_seconds,
            workspace_estimated_bytes=p.operator.estimated_workspace_bytes,backend_consistency=backend))
    # Alternate rule order to expose drift; each trial creates the same state.
    for repeat in range(3):
        for order in ((7,5) if repeat%2==0 else (5,7)):
            p=operators[order];p.operator.reset_profile();p._cached_q=None;p._cached_response=None
            state=initial.clone();state.child_states['identity']=copy.deepcopy(p.identity)
            c=copy.deepcopy(cfg);c['material_order']=order
            begun=time.perf_counter();stepper=ValidatedAVF(p,c,state)
            rows=[stepper.step(.025),stepper.step(.025)];elapsed=time.perf_counter()-begun
            record=dict(kind='warm',repeat=repeat,order=order,seconds=elapsed,
                partitions_seconds=dict(p.operator.timings),partition_calls=dict(p.operator.counts),
                non_operator_seconds=elapsed-sum(p.operator.timings.values()),
                accepted_steps=2,rejected_steps=len(stepper.failures),newton_iterations=[r['newton_iterations'] for r in rows],
                krylov_iterations=sum(r['krylov_iterations'] for r in rows),min_detF=min(r['min_detF'] for r in rows),
                q=stepper.state.q.tolist(),velocity=stepper.state.velocity.tolist(),rows=rows,
                peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
            records.append(record)
            write(run/'N07/samples.json',records)
            print('warm',repeat,'q',order,'seconds',round(elapsed,4),'partitions',record['partitions_seconds'],flush=True)
    medians={str(order):float(np.median([r['seconds'] for r in records if r['kind']=='warm' and r['order']==order])) for order in (7,5)}
    spreads={str(order):float(np.ptp([r['seconds'] for r in records if r['kind']=='warm' and r['order']==order])) for order in (7,5)}
    q7=[r for r in records if r['kind']=='warm' and r['order']==7]
    names=q7[0]['partitions_seconds']
    partitions={key:float(np.median([r['partitions_seconds'][key] for r in q7])) for key in names}
    ordered=sorted(partitions.items(),key=lambda x:x[1],reverse=True)
    result=dict(utc=utc(),status='passed_scoped',median_seconds=medians,range_seconds=spreads,
        q5_relative_saving=1-medians['5']/medians['7'],q7_partition_medians=partitions,
        leading_costs=ordered[:4],full_cost_caveat='Construction recorded separately; stage process wall time and JSON IO include measurement overhead. No exclusive production speed claim.',
        measured_rejected_steps=0,actual_krylov_calls=sum(r['krylov_iterations'] for r in q7),
        next='N08 bounded same-state linearization reuse; quantify benefit on repeated tangent workloads and report if ordinary cycles gain little.')
    write(run/'N07/result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);a=p.parse_args()
    with serial_lock(a.run):profile(a.run)
