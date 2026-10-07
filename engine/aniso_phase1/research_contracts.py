"""Schema-1 handoff semantics for isolated v22 research directions.

D owns only this interface. A/B/C/E retain their space, quadrature, temporal
transaction and physical-law definitions. No caller may infer dynamic approval
from a static result. A package names sealed evidence by content, never latest.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Protocol
import numpy as np


@dataclass(frozen=True)
class HandoffMetadata:
    schema_version: int
    baseline_sha256: str
    producer: str
    code_sha256: dict[str,str]
    input_sha256: dict[str,str]
    units: dict[str,str]
    dtype: str
    device: str
    coordinate_system: str
    q_convention: str
    material_parameters: dict
    boundary_conditions: dict
    volume_convention: str
    random_seed: int
    capabilities: tuple[str,...]

    def validate(self):
        if self.schema_version!=1: raise ValueError('unsupported schema; explicit adapter required')
        if self.producer not in ('A','B','C','D','E'): raise ValueError('unknown producer')
        if self.dtype not in ('float64','float32'): raise ValueError('explicit numerical precision required')
        if self.q_convention not in ('displacement','total_position','velocity'):
            raise ValueError('explicit position/displacement/velocity convention required')
        if not all((self.units,self.device,self.coordinate_system,self.boundary_conditions,
                    self.volume_convention,self.code_sha256,self.input_sha256,self.material_parameters)):
            raise ValueError('incomplete physical or provenance metadata')
        if not isinstance(self.random_seed,int): raise ValueError('integer seed required')
        for value in [self.baseline_sha256,*self.code_sha256.values(),*self.input_sha256.values()]:
            if len(value)!=64 or any(c not in '0123456789abcdef' for c in value): raise ValueError('invalid SHA256')
        if 'dynamic' in self.capabilities and 'transaction' not in self.capabilities:
            raise ValueError('dynamic handoff must declare C-owned transaction interface')
        return self


class SpaceMap(Protocol):
    """All methods refer to the same frozen basis and constraint lifting."""
    def positions(self,q:np.ndarray)->np.ndarray: ...
    def deformation(self,q:np.ndarray)->np.ndarray: ...
    def direction(self,q:np.ndarray,dq:np.ndarray)->np.ndarray: ...
    def force_adjoint(self,q:np.ndarray,dual:np.ndarray)->np.ndarray: ...


class StateTransaction(Protocol):
    """Implemented and ordered by C; D never commits trial states inside Newton."""
    def trial(self,q:np.ndarray): ...
    def accept(self,trial): ...
    def rollback(self,trial): ...


@dataclass(frozen=True)
class SolverCallbacks:
    size: int
    residual: Callable
    exact_tangent: Callable
    preconditioner: Callable
    project: Callable
    identity_sha256: str
    tangent_kind: str = 'exact'
    preconditioner_fixed_spd: bool = True

    def validate(self):
        if self.size<1 or self.tangent_kind!='exact': raise ValueError('exact physical tangent required')
        if not all(callable(f) for f in (self.residual,self.exact_tangent,self.preconditioner,self.project)):
            raise ValueError('missing solver callback')
        if not self.preconditioner_fixed_spd: raise ValueError('PCG requires fixed SPD preconditioning; use another method')
        if len(self.identity_sha256)!=64: raise ValueError('frozen operator identity required')
        return self


def verify_files(root,manifest):
    root=Path(root).resolve()
    for relative,expected in manifest.items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root): raise ValueError('handoff paths must remain under repository root')
        with path.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected: raise ValueError(f'changed handoff artifact: {relative}')


def validate_package(package,root,baseline_sha256,*,require_dynamic=False):
    """Verify bytes AND a scoped acceptance result; format stubs cannot pass."""
    meta=HandoffMetadata(**package['metadata']).validate()
    if meta.baseline_sha256!=baseline_sha256: raise ValueError('different baseline requires explicit integration decision')
    verify_files(root,meta.code_sha256); verify_files(root,meta.input_sha256)
    evidence=package['acceptance']; verify_files(root,{evidence['path']:evidence['sha256']})
    record=json.loads((Path(root)/evidence['path']).read_text())
    required='dynamic' if require_dynamic else 'static'
    if record.get('producer')!=meta.producer or record.get('baseline_sha256')!=meta.baseline_sha256:
        raise ValueError('acceptance provenance differs from package')
    if record.get('input_sha256')!=meta.input_sha256 or record.get('code_sha256')!=meta.code_sha256:
        raise ValueError('acceptance does not bind exactly these inputs and implementation')
    if record.get('passed') is not True or required not in record.get('validated_capabilities',[]):
        raise ValueError(f'{required} physics acceptance missing or failed')
    if required not in meta.capabilities: raise ValueError('capability not declared')
    return meta
