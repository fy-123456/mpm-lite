"""Verify a B child delivery without extending its parent's capabilities."""
import json
from pathlib import Path
from .material import PARENT_SHA
from .operator import CommonMaterialOperator, FixedRule
from .material import MaterialSource
from ...research_d.frozen_inputs import load_frozen_inputs
from ...research_d.identity import sha


def _member(root,name,trust=None):
    rel=Path(name)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe child member path')
    p=root/rel;target=p.resolve()
    if not target.is_relative_to(root.resolve()) and (trust is None or not target.is_relative_to(Path(trust).resolve())):
        raise ValueError('external child artifact requires trusted data root')
    if not p.is_file():raise ValueError('missing child artifact: '+name)
    return p


def load_delivery(folder, repository, *, expected_sha256, trusted_data_root=None, field='F45', compressed=True, require_dynamic=False):
    root=Path(folder);repo=Path(repository)
    manifest=root/'handoff-package.json'
    if sha(manifest)!=expected_sha256:raise ValueError('B delivery identity changed')
    data=json.loads(manifest.read_text())
    if data['parent_bundle_sha256']!=PARENT_SHA:raise ValueError('wrong B parent')
    for name,expected in data['extension_source_sha256'].items():
        if sha(_member(repo,name))!=expected:raise ValueError('changed B extension source: '+name)
    for name,expected in data['artifacts_sha256'].items():
        if sha(_member(root,name,trusted_data_root))!=expected:raise ValueError('changed B evidence: '+name)
    if require_dynamic:raise ValueError('B static delivery does not certify a C dynamic cycle')
    if field not in data['fields']:raise ValueError('unknown material variant')
    entry=data['fields'][field]
    if not entry['certified']:raise ValueError('requested material variant is not certified')
    space,_,_,parent=load_frozen_inputs(repo/data['parent_relative_path'],repo,expected_sha256=PARENT_SHA,
                                      trusted_data_root=trusted_data_root)
    if space.signature!=data['space_manifest_sha256']:raise ValueError('wrong B space instance')
    order=entry['candidate_order'] if compressed else entry['full_order']
    source=MaterialSource(field)
    if source.signature!=entry['source_sha256']:raise ValueError('B material provenance mismatch')
    operator=CommonMaterialOperator(space,FixedRule.uniform(space,order),source,min_detF=data['min_detF'])
    return operator,dict(parent=parent,field=field,scope='bounded static states and directions',dynamic_certified=False)
