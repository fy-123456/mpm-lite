"""Seal the actual final implementation, bounded scene checks and release index."""
from pathlib import Path
import argparse
import copy
import json
import numpy as np
from .provenance import ROOT,PREVIOUS,COMMON,read,write,sha,check,source_files,verify,register_study,serial_lock,utc,environment
from .run import create_config,load_model
from .checkpoint import GenerationStore
from .compare import compare_cases
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF

CASES=['seg-q7-dt0025','seg-peak0075-q7','seg-peak0075-q5']


def prepare(run):
    run=Path(run)
    register_study(run,'N12/protocol.json',dict(utc=utc(),cases=CASES,dt=.025,steps_per_full_case=64,
        display_frames=12,default_operator='segmented',cache=False,preconditioner='original',
        restart_case=CASES[0],restart_after_steps=2,complete_cycles=3,
        inherited_temporal_spatial_limits=True,scope='final code, same equations, practical scene and persistence validation'))
    create_config(run,CASES[0],dt=.025,operator_mode='segmented')
    create_config(run,CASES[1],dt=.025,operator_mode='segmented',peak_m=.0075)
    create_config(run,CASES[2],dt=.025,operator_mode='segmented',material_order=5,peak_m=.0075)
    write(run/'N12/environment.json',environment())
    return CASES


def check_sources(run):
    p=read(Path(run)/'N12/protocol.json');current=source_files()
    if current!=p['source_sha256']:raise ValueError('final numerical implementation changed after registration')
    check(Path(run)/'N12/protocol-source',current)
    return current


def restart_check(run):
    run=Path(run);check_sources(run);folder=run/'cases'/CASES[0]
    cfg=read(folder/'execution-protocol.json');identity=read(folder/'identity.json')
    chain=GenerationStore(folder,identity).history()
    if len(chain)!=3 or chain[-1]['state'].step!=2:raise ValueError('restart witness requires exactly two durable steps')
    write(run/'N12/restart-witness.json',dict(utc=utc(),generation=read(folder/'CURRENT.json'),state_digest=chain[-1]['state'].digest(),
        identity_sha256=sha(folder/'identity.json'),source_sha256=source_files(),
        context='first CLI process returned after two accepted steps; next cycle invocation must load this exact generation'))


def analyze(run):
    run=Path(run);sources=check_sources(run);baseline=verify(run)
    for name in CASES:
        result=read(run/'cases'/name/'summary.json')
        if result['status']!='passed_scoped' or result['steps']!=64 or result['end_s']!=1.6:raise ValueError('incomplete final cycle: '+name)
        if result['source_sha256']!=sources:raise ValueError('final cycle used other source')
    comparisons={}
    for a,b in [(CASES[0],'gpu-q7-dt0025'),(CASES[2],CASES[1])]:
        comparisons[a+'--'+b]=compare_cases(run,a,b)
        if not comparisons[a+'--'+b]['passed']:raise ValueError('final physical field comparison failed')
    folder=run/'cases'/CASES[0];identity=read(folder/'identity.json');chain=GenerationStore(folder,identity).history()
    witness=read(run/'N12/restart-witness.json')
    if chain[2]['state'].digest()!=witness['state_digest'] or chain[2]['folder'].name!=witness['generation']['generation']:
        raise ValueError('first process committed prefix changed')
    if len(read(folder/'segments.json'))!=2:raise ValueError('actual CLI restart segments were not retained')
    cfg=read(folder/'execution-protocol.json');model,_=load_model(run,cfg)
    # Four short steps from the same rest state give an uninterrupted control.
    stepper=ValidatedAVF(model,cfg)
    rows=[stepper.step(cfg['times'][i+1]-cfg['times'][i]) for i in range(4)]
    expected=chain[4]['state'];difference={key:float(np.max(abs(getattr(expected,key)-getattr(stepper.state,key)))) for key in ['q','velocity','predictor']}
    if max(difference.values())>1e-8:raise ValueError('process restart changes physical history')
    numeric_ledger=[]
    for a,b in zip(rows,chain[4]['rows']):
        diff={key:abs(float(a[key])-float(b[key])) for key in a if isinstance(a[key],(int,float)) and key!='wall_seconds'}
        numeric_ledger.append(diff)
        if max(diff.values(),default=0)>1e-8:raise ValueError('process restart changes step ledger')
    old_counts=dict(sources=check(ROOT,baseline['old_source_sha256']),inputs=check(ROOT,baseline['input_sha256']),
        previous_artifacts=check(PREVIOUS,read(PREVIOUS/'artifact-sha256.json')))
    result=dict(utc=utc(),status='passed_scoped',cycles={name:read(run/'cases'/name/'summary.json') for name in CASES},
        comparisons={key:dict(passed=value['passed'],path='comparisons/'+key+'.json') for key,value in comparisons.items()},
        restart=dict(actual_two_process_resume=True,prefix_digest_preserved=True,q_v_predictor_max_difference=difference,
            uninterrupted_four_step_control=True,numeric_ledger_differences=numeric_ledger),
        final_source_verified=True,old_inputs_verified=old_counts,
        still_limited=['time_accuracy','spatial_reference_accuracy'],physical_3D_coupling=False)
    write(run/'N12/scene-result.json',result)
    return result


def seal(run):
    run=Path(run);sources=check_sources(run);verify(run)
    result=read(run/'N12/scene-result.json');audit=read(run/'N12/requirement-audit.json')
    if not audit.get('all_requirements_accounted_for') or audit.get('unresolved_requirements'):
        raise ValueError('requirement-by-requirement completion audit is incomplete')
    tests=read(run/'N12/tests-result.json')
    if not tests['passed']:raise ValueError('final risk tests missing or failed')
    for name in CASES:
        meta=read(run/'visualization'/name/'metadata.json')
        if meta['accepted_steps']!=64 or meta['committed_time_s']!=1.6:raise ValueError('final visualization incomplete')
    caps=read(run/'capabilities.json');caps['N08']['full_cycle_integration']=True;caps['N10']['full_cycle']=True
    caps['N12']=dict(status='passed_scoped',evidence='N12/result.json')
    write(run/'capabilities.json',caps)
    write(run/'N12/result.json',dict(result,tests=tests,completion_audit='N12/requirement-audit.json',capabilities='capabilities.json'))
    final_sources=dict(sources)
    for p in sorted((ROOT/'tests/research_sequential_next').glob('*.py')):final_sources[str(p.relative_to(ROOT))]=sha(p)
    viz=ROOT/'benchmarks/research_sequential_next/visualize.py';final_sources[str(viz.relative_to(ROOT))]=sha(viz)
    for name,h in final_sources.items():
        dest=run/'final-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    write(run/'final-source-sha256.json',final_sources)
    excluded={'artifact-sha256.json','release.json'}
    files={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file() and 'warp-cache' not in p.parts and p.name not in excluded}
    write(run/'artifact-sha256.json',files)
    release=dict(schema='sequential-next-practical-release-v1',utc=utc(),physical_path=str(run.resolve()),
        mathematical_package=dict(path='model/model-package.json',sha256=sha(run/'model/model-package.json')),
        baseline_sha256=sha(run/'baseline-lock.json'),numerical_sources=dict(path='final-source-sha256.json',sha256=sha(run/'final-source-sha256.json')),
        artifacts=dict(path='artifact-sha256.json',sha256=sha(run/'artifact-sha256.json')),
        default_case=CASES[0],cases={name:dict(protocol='cases/'+name+'/execution-protocol.json',protocol_file_sha256=sha(run/'cases'/name/'execution-protocol.json'),
            identity_sha256=sha(run/'cases'/name/'identity.json')) for name in CASES},
        operator='segmented FP64 CSR',material_order_default=7,mass_order=5,linearization_cache_default=False,preconditioner_default='original',
        temporal_accuracy=False,spatial_accuracy=False,independent_2D_physics=True,physical_3D_coupling=False)
    write(run/'release.json',release)
    check(run,read(run/'artifact-sha256.json'));check(ROOT,final_sources)
    return release


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','witness','analyze','seal']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        answer={'prepare':prepare,'witness':restart_check,'analyze':analyze,'seal':seal}[a.phase](a.run)
        print(json.dumps(answer if a.phase in ['prepare','seal'] else dict(phase=a.phase,complete=True),ensure_ascii=False),flush=True)
