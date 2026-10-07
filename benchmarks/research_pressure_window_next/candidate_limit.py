"""Close an interrupted research branch using verified commits, without steps."""
from pathlib import Path
import argparse
from .provenance import *
from benchmarks.research_sequential_next.checkpoint import GenerationStore


def close(run):
    run=Path(run);verify(run);histories=[]
    for name in ('candidate-h','candidate-h-continuation'):
        folder=run/'cases'/name
        identity=read(folder/'identity.json')
        if identity['numerical_source_sha256']!=source_files():raise ValueError('candidate numerical source drift')
        histories.append(GenerationStore(folder,identity).history())
    first,tail=histories
    if first[-1]['state'].digest()!=tail[0]['state'].digest():
        raise ValueError('candidate continuation did not preserve exact state')
    rows=first[-1]['rows']+tail[-1]['rows']
    if len(rows)!=62 or tail[-1]['state'].step!=62:raise ValueError('unexpected safe prefix')
    reason='CUDA conservative reserve guard rejected trial step63; no failed state committed. Repeated intermittent device-memory changes remain unisolated; stop this research branch under the plan bounded-anomaly rule.'
    write(run/'S3/resource-limited-prefix.json',dict(status='limited',reason=reason,accepted_steps=62,failed_step_attempts=1,last_step=62,last_time_s=tail[-1]['state'].time,last_digest=tail[-1]['state'].digest(),last_pointer_sha256=sha(run/'cases/candidate-h-continuation/CURRENT.json'),minimum_detF=min(r['min_detF'] for r in rows),remaining_coarse_steps=2,fine_steps_not_started=128,source_hashes_unchanged=True,zero_new_steps=True,root_cause_confirmed=False,excluded=['projection and mass solve residual explain no numerical blow-up','generation chain and resumed state identity verified'],unresolved=['transient device use versus allocator reservations','holding two models may increase headroom requirement; not isolated']))
    write(run/'S3/candidate-time-check.json',dict(status='limited',actual_steps=62,failed_step_attempts=1,candidate_time_refinement_passed=False,cross_space_fine_passed=False,completed_coarse_window=False,completed_fine_window=False,self_comparison=[],records=[dict(case='candidate-h-partial',steps=62,comparisons=[])],formal_fine_is_not_continuum_truth=True,reason=reason))
    write(run/'S3/research-space-decision.json',dict(status='limited',candidate='balanced-direction-snapshot6',static_qualified=True,projection_passed=True,short_dynamic_stable=False,completed_prefix_stable=True,time_refinement=False,cross_space_fine=False,formal_space_changed=False,spatial_accuracy=False,full_dynamic_cycle=False,q5=False,pressure=False,actual_steps=62,reason='Initial stabilization energy and physical acceleration differ; mass equilibration leaves physical acceleration essentially unchanged. Temporal and spatial causes are not fully separated because the resource-limited 62-step prefix is incomplete.',next_experiment='In an isolated device window with one model resident, finish the preserved h prefix and a separately registered h/2 reference before changing projection, stabilization or training.',failure_criterion='Stop if conservative memory guard or detF/energy/residual gate fails; do not change damping, mass or tolerance to force agreement.'))
    print('CANDIDATE_LIMITED',len(rows),tail[-1]['state'].time,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args()
    with serial_lock(a.run):close(a.run)
