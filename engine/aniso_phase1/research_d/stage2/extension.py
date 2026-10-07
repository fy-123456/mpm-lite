"""Verify the complete parent and the independently pinned D extension."""
import json
from pathlib import Path
from .contracts import PARENT_SHA256,sha
from ..frozen_inputs import load_frozen_inputs

SOURCE_PREFIXES=('engine/aniso_phase1/research_d/stage2/','benchmarks/research_d/stage2/','tests/research_d/stage2/')

def member(root,name):
    root=Path(root).resolve();rel=Path(name);p=root/rel
    if rel.is_absolute() or '..' in rel.parts or not p.resolve().is_relative_to(root) or not p.is_file():
        raise ValueError('missing or external extension member: '+name)
    return p

def verify_extension(folder,repository,*,expected_sha256,require=(),trusted_parent_data_root=None):
    root,repo=Path(folder),Path(repository)
    manifest=root/'extension.json'
    if sha(manifest)!=expected_sha256:raise ValueError('wrong extension digest')
    data=json.loads(manifest.read_text())
    if data.get('schema_version')!=2 or data.get('parent_bundle_sha256')!=PARENT_SHA256:raise ValueError('unsupported parent/schema')
    if not data.get('extension_source_sha256') or not data.get('files'):raise ValueError('empty extension closure')
    for name,digest in data['extension_source_sha256'].items():
        if not name.startswith(SOURCE_PREFIXES) or sha(member(repo,name))!=digest:raise ValueError('changed extension source: '+name)
    for name,digest in data['files'].items():
        if sha(member(root,name))!=digest:raise ValueError('changed extension data: '+name)
    parent=member(repo,data['parent_bundle_path']+'/bundle.json').parent
    _,_,_,verified=load_frozen_inputs(parent,repo,expected_sha256=PARENT_SHA256,trusted_data_root=trusted_parent_data_root)
    for capability in require:
        evidence=data['capabilities'].get(capability,{})
        if evidence.get('passed') is not True:raise ValueError('unavailable capability: '+capability)
        record=json.loads(member(root,evidence['evidence']).read_text())
        if record.get('passed') is not True:raise ValueError('capability evidence did not pass')
    if any(data['capabilities'].get(k,{}).get('passed') for k in ('dynamic_cycle','coupled_physics','continuum_spatial_accuracy','production_default')):
        raise ValueError('static D extension cannot grant unrelated certification')
    return dict(passed=True,parent=verified,extension_sha256=expected_sha256,sources=len(data['extension_source_sha256']),files=len(data['files']),capabilities=data['capabilities'])
