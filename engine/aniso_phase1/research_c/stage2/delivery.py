"""Strict stage-2 consumer: parent verification and extension verification differ."""
import json
from pathlib import Path
import numpy as np
from ...research_d.frozen_inputs import load_frozen_inputs
from ...research_d.identity import sha
from ...research_d.common_state import CommonState
from .model import FrozenPotential


def verify_delivery(folder, repository, *, trusted_parent_data_root,
                    expected_manifest_sha256=None, require_dynamic=False, manifest_name="handoff-package.json"):
    folder, repo = Path(folder), Path(repository)
    if repo.resolve() != Path(__file__).resolve().parents[4]:
        raise ValueError('extension must be imported from the verified source copy')
    if manifest_name not in ('handoff-package.json','preliminary-handoff.json'):
        raise ValueError('unsupported manifest name')
    manifest = folder/manifest_name
    if expected_manifest_sha256 is not None and sha(manifest) != expected_manifest_sha256:
        raise ValueError('wrong stage2 manifest identity')
    data = json.loads(manifest.read_text())
    if data.get('schema_version') != 1 or data.get('producer') != 'research_c_stage2':
        raise ValueError('unsupported C stage2 schema')
    def member(root, rel):
        p = Path(rel)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('invalid manifest member path')
        target = (root/p).resolve()
        if not target.is_relative_to(root.resolve()) or not target.is_file():
            raise ValueError('untrusted or missing stage2 member')
        return target
    for name, expected in data['extension_source_sha256'].items():
        if sha(member(repo, name)) != expected:
            raise ValueError('extension source changed: '+name)
    required = {'acceptance.json','dynamic-protocol.json','baseline-check.json',
        'mass-audit.json','mass-nullspace-audit.json','mass-null-directions.npz',
        'initial-dynamic-state.npz','operators.npz','dynamic-contract.json'}
    if not required.issubset(data['artifacts_sha256']):
        raise ValueError('incomplete C stage2 evidence')
    for name, expected in data['artifacts_sha256'].items():
        if sha(member(folder, name)) != expected:
            raise ValueError('stage2 artifact changed: '+name)
    s, _, _, parent = load_frozen_inputs(data['parent_bundle_path'],repo,
        trusted_data_root=trusted_parent_data_root,expected_sha256=data['parent_bundle_sha256'])
    if s.signature != data['space_manifest_sha256']:
        raise ValueError('wrong stage2 space')
    acceptance = json.loads((folder/'acceptance.json').read_text())
    if require_dynamic and acceptance.get('dynamic_cycle') is not True:
        raise ValueError('C stage2 dynamic cycle is not certified: '+acceptance['status'])
    return s, dict(parent_verified=parent['passed'], extension_verified=True,
        dynamic_cycle=acceptance['dynamic_cycle'], status=acceptance['status'],
        source_count=len(data['extension_source_sha256']), artifact_count=len(data['artifacts_sha256']))


def recompute(space, state, points, mass, *, material_order=6):
    """Reconstruct full x/F/v/C/PK1 and total linear/angular momentum."""
    if not isinstance(state, CommonState):
        raise TypeError('explicit full CommonState required')
    space._check(state.q,(space.ndof,3),'full displacement')
    space._check(state.velocity,(space.ndof,3),'full velocity')
    mass=np.asarray(mass)
    if mass.shape != (space.ndof,space.ndof) or not np.isfinite(mass).all():
        raise ValueError('finite full scalar mass required')
    out=FrozenPotential(space,order=material_order).fields(state,points)
    dual=mass@state.velocity
    out['linear_momentum']=dual[:space.n].sum(axis=0)
    out['angular_momentum']=np.cross(space.reference+state.q,dual).sum(axis=0)
    out['kinetic_J']=.5*float(np.sum(state.velocity*dual))
    return out
