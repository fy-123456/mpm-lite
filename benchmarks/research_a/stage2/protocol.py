"""Freeze source and inputs, consume the parent from an independent checkout."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
from benchmarks.research_a.protocol import write_json, sha
from engine.aniso_phase1.research_a.stage2.cases import registered_cases

ROOT = Path(__file__).resolve().parents[3]
PARENT_REL = Path('docs/results/parallel-v22/integration/20260930T054100Z-common-inputs')
PARENT_SHA = '55682a7b8e90b62c1306818cdf9c1f060b4286a174da069ce2217aa5b53b3c7c'
SPACE_SHA = '7422b2b099127bebe9c144ea6cac94409bde861aadb94693b7fc650b397c0332'
DATA = Path('/root/autodl-tmp/mpm-lite-research-a/stage2')
TRUSTED_PARENT = Path('/root/autodl-tmp/mpm-lite-research-d/common-inputs/20260930T054100Z-common-inputs')
PREFIXES = ('engine/aniso_phase1/research_a/stage2', 'benchmarks/research_a/stage2', 'tests/research_a/stage2')
GIB = 1024**3


def extension_manifest(root=ROOT):
    return {str(p.relative_to(root)): sha(p) for prefix in PREFIXES for p in sorted((root/prefix).glob('*.py'))}


def verify_parent(root=ROOT):
    from engine.aniso_phase1.research_d.frozen_inputs import verify_frozen_inputs
    result=verify_frozen_inputs(root/PARENT_REL, root, trusted_data_root=TRUSTED_PARENT, expected_sha256=PARENT_SHA)
    if result['space_manifest_sha256'] != SPACE_SHA: raise ValueError('Unexpected parent space')
    return result


def run_path(run_id):
    if not run_id or Path(run_id).name != run_id or run_id in ('.','..'): raise ValueError('Invalid run id')
    return ROOT/'docs/results/parallel-v22-stage2/A'/run_id


def matrix():
    specs=[]
    def add(case='F45',degree=4,family='v22-overlap',score='stress-correction',budget=144,mode='regenerate',sequence=None):
        name=f'{case}-q{degree}-{family}-{score}-{budget}-{mode}'
        if sequence: name+='-fixed-selection'
        if not any(s['name']==name for s in specs):
            specs.append(dict(name=name,case=case,degree=degree,family=family,score=score,budget=budget,mode=mode,sequence=sequence))
        return name
    for b in (72,144,288): add(budget=b)
    for f in ('v22-original','wide-overlap','fiber-rect'): add(family=f)
    original=specs[1]['name']
    for p in (2,3): add(degree=p,sequence=original); add(degree=p)
    for score in ('energy','residual','geometric'): add(score=score)
    for name in list(registered_cases())[1:]:
        add(case=name)
        if name != 'tall-shear': add(case=name,mode='reuse-F45')
    return specs


def attach_history(original, target):
    """Keep archived source members; link only missing historical assets."""
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(original, target_is_directory=original.is_dir())
    elif target.is_dir() and original.is_dir():
        for child in original.iterdir(): attach_history(child, target/child.name)
    elif target.is_file() and original.is_file() and sha(target)==sha(original):
        return
    else: raise ValueError('Conflicting frozen source/history member: '+str(target))


def freeze(run_id):
    verified=verify_parent()
    if shutil.disk_usage('/').free < 5*GIB: raise RuntimeError('Root below 5 GiB: migrate expendable A assets before starting')
    if os.stat(DATA.parent).st_dev == os.stat('/').st_dev: raise RuntimeError('Data storage must be on a separate filesystem')
    if shutil.disk_usage(DATA.parent).free < 30*GIB: raise RuntimeError('Insufficient data headroom')
    out=run_path(run_id); out.mkdir(parents=True,exist_ok=False)
    data=DATA/run_id; data.mkdir(parents=True,exist_ok=False)
    arrays=data/'arrays'; arrays.mkdir(); (out/'arrays').symlink_to(arrays,target_is_directory=True)
    source=data/'source'; source.mkdir()
    # Restore only the audited source archive. Read-only historical inputs remain explicit links.
    with zipfile.ZipFile(ROOT/PARENT_REL/'code-snapshot.zip') as archive:
        for item in archive.infolist():
            path=Path(item.filename)
            if path.is_absolute() or '..' in path.parts: raise ValueError('Unsafe snapshot member')
        archive.extractall(source)
    for prefix in PREFIXES:
        for path in (ROOT/prefix).glob('*.py'):
            target=source/path.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
    history=source/'docs/results/lite-aniso-mainline'; history.parent.mkdir(parents=True,exist_ok=True)
    attach_history(ROOT/'docs/results/lite-aniso-mainline', history)
    parent=source/PARENT_REL; parent.parent.mkdir(parents=True,exist_ok=True); attach_history(ROOT/PARENT_REL, parent)
    oldA=source/'docs/results/parallel-v22/A'; oldA.parent.mkdir(parents=True,exist_ok=True); attach_history(ROOT/'docs/results/parallel-v22/A', oldA)
    general=ROOT/'docs/results/parallel-v22/A/20260930T0405Z-A-pilot/generalization-protocol.json'
    protocol=dict(schema_version=2,producer='research_a.stage2',utc=datetime.now(timezone.utc).isoformat(),
        parent_bundle_sha256=PARENT_SHA,parent_space_manifest_sha256=SPACE_SHA,
        extension_source_sha256=extension_manifest(),source_checkout=str(source),original_repository=str(ROOT),
        python=sys.executable,seed=220930,dtype='float64',device='cpu',threads=2,
        cases={n:c.as_dict() for n,c in registered_cases().items()},candidates=matrix(),
        generalization_protocol_sha256=sha(general),case_construction_does_not_count_as_hidden_evaluation=True,
        rounds=6,patches_per_round={'72':4,'144':8,'288':16},final_round=6,
        rank_policy='twice reorthogonalize, SVD threshold 1e-8; rank shortfall fails explicitly; no stiffness shift',
        local_solve_relative_tolerance=2e-8,
        selected_policy=dict(family='v22-overlap',score='stress-correction',degree=4,budget=144),
        selection='predeclared original online policy; never chosen from final reference errors',
        development_reference='docs/results/lite-aniso-mainline/v21/reference-extension/level1-q4.npz',
        old_v22_reference='already observed historical development data, never a new hidden reference',
        hidden_access='A10 only after candidate, strategy and all eight result/failure seals; not implemented in kickoff',
        implementation_gates=dict(free_residual=1e-6,work_relative=2e-5,boundary_m=1e-8,trace=1e-10,
            orthogonality=1e-7,energy_derivative_absolute_J=1e-7,tangent_relative=2e-3,
            rotation_energy_absolute_J=1e-7,rotation_force_relative=1e-4,min_detF=.05),
        user_tolerance_policy='Engineering scene correctness and stability; relaxed new derivative checks, old parent decisions unchanged',
        scientific_targets=dict(reference_stress=.005,reference_fiber=.005,reference_reaction=.0025,
            candidate_stress=.02,candidate_fiber=.02,candidate_reaction=.01),
        near_zero_scales=dict(stress_Pa=.001,fiber_strain=.00001,reaction_N=.00001,moment_Nm=.000001),
        resources=dict(memory_limit_GiB=24,node_limit=14000000,max_reference_seconds=3600,
            max_candidate_seconds=3600,system_floor_GiB=5,concurrent_large_jobs=1,threads=2,
            timings='exploratory shared-host wall time; no fair speedup claim',formal_repeats_required=3),
        parent_read_only=True,auto_promote=False,
        capabilities=dict(static_operator=False,bounded_material_reference=False,dynamic_cycle=False,cuda=False,coupled_physics=False))
    write_json(out/'protocol.json',protocol)
    inputs={str(general.relative_to(ROOT)):sha(general)}
    for rel in ('v11-reference-q3/cases/local2.npz','v19/space/reconstruction32.npz','v20/space/F45-r32-e0.npz','v21/reference-extension/level1-q4.npz'):
        path=ROOT/'docs/results/lite-aniso-mainline'/rel; inputs[str(path.relative_to(ROOT))]=sha(path)
    write_json(out/'parent-inputs.json',dict(parent=verified,inputs_sha256=inputs,system_free_bytes=shutil.disk_usage('/').free,
        data_free_bytes=shutil.disk_usage(DATA).free,arrays=str(arrays),independent_source=str(source),
        numpy_import_note='Current frozen support_family already imports NumPy; plan text stale; no parent edit',
        cache_migration_note='Shared pip cache was already migrated by research B; A leaves its target untouched'))
    write_json(out/'resource-registration.json',dict(owner='A',threads=2,large_concurrency=1,shared_host=True,
        fair_timing_reserved=False,formal_timing_status='not executed; requires uncontended registered window',
        predeclared_stop='24 GiB RSS, 3600 s per candidate/reference, root <5 GiB or data <5 GiB'))
    return out


def require_frozen(run):
    run=Path(run); p=json.loads((run/'protocol.json').read_text())
    if extension_manifest()!=p['extension_source_sha256']: raise RuntimeError('Extension source changed: create new run')
    if ROOT.resolve()!=Path(p['source_checkout']).resolve(): raise RuntimeError('Execute from the independent audited source checkout')
    verify_parent()
    for rel,digest in json.loads((run/'parent-inputs.json').read_text())['inputs_sha256'].items():
        if sha(ROOT/rel)!=digest: raise RuntimeError('Historical input changed: '+rel)
    return p


def save_array(run,relative,**arrays):
    import numpy as np
    path=Path(run)/'arrays'/relative; path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    with path.open('xb') as stream: np.savez_compressed(stream,**arrays)
    return path
