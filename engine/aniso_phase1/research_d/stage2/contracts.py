"""Versioned, fail-closed extension identities; no inferred coordinate semantics."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import numpy as np

PARENT_SHA256 = '55682a7b8e90b62c1306818cdf9c1f060b4286a174da069ce2217aa5b53b3c7c'
SPACE_SHA256 = '7422b2b099127bebe9c144ea6cac94409bde861aadb94693b7fc650b397c0332'
CAPABILITIES = {'static_operator', 'bounded_material_reference', 'dynamic_cycle', 'cuda', 'coupled_physics'}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def array_digest(a):
    a=np.asarray(a,dtype='<f8',order='C')
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

@dataclass(frozen=True)
class OperatorContract:
    parent_bundle_sha256: str
    space_sha256: str
    material_rule_sha256: str
    mass_rule_sha256: str
    material_sha256: str
    boundary_sha256: str
    state_sha256: str
    extension_source_sha256: str
    q_shape: tuple
    full_shape: tuple
    device: str
    operator_kind: str = 'solid_exact_hessian'
    dtype: str = 'float64'
    coordinates: str = 'displacement; free carrier then local; xyz last'
    force: str = 'positive potential gradient'
    weights: str = 'unweighted adjoint; dV applied once'
    schema_version: int = 2

    def __post_init__(self):
        object.__setattr__(self,'q_shape',tuple(self.q_shape))
        object.__setattr__(self,'full_shape',tuple(self.full_shape))

    @property
    def signature(self): return digest(asdict(self))

    def validate(self, expected=None):
        for key,value in asdict(self).items():
            if key.endswith('sha256') and (not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('missing or invalid '+key)
        if self.schema_version!=2 or self.dtype!='float64' or not self.device:
            raise ValueError('unsupported schema/dtype/device')
        if self.operator_kind not in ('solid_exact_hessian','mixed_general'):
            raise ValueError('unknown operator kind')
        if self.coordinates!='displacement; free carrier then local; xyz last' or self.force!='positive potential gradient' or self.weights!='unweighted adjoint; dV applied once':
            raise ValueError('incompatible field semantics')
        if len(self.q_shape)!=2 or len(self.full_shape)!=2 or self.q_shape[1]!=self.full_shape[1] or not all(isinstance(i,int) and not isinstance(i,bool) and i>0 for i in (*self.q_shape,*self.full_shape)):
            raise ValueError('explicit finite coordinate shapes required')
        if self.q_shape[0]>self.full_shape[0]:raise ValueError('invalid free/full shapes')
        if self.operator_kind=='solid_exact_hessian' and self.q_shape[1]!=3:raise ValueError('solid requires xyz')
        if expected:
            for key,value in expected.items():
                if getattr(self,key)!=value:raise ValueError('contract mismatch: '+key)
        return self

    def check_field(self,value,kind):
        shapes={'free_displacement':self.q_shape,'full_displacement':self.full_shape,'lift':self.full_shape,'lift_velocity':self.full_shape,'full_velocity':self.full_shape}
        if kind not in shapes:raise ValueError('unknown coordinate meaning')
        a=np.asarray(value)
        if a.dtype!=np.float64 or a.shape!=shapes[kind] or not np.isfinite(a).all():raise ValueError('invalid '+kind)
        return a

def validate_handoff(contract, *, expected, capabilities, required, evidence, solver='general', spd_evidence=None, callbacks=None):
    contract.validate(expected)
    if solver not in ('general','pcg','svd'):raise ValueError('unknown solver capability')
    if set(capabilities)-CAPABILITIES or not set(required)<=set(capabilities):
        raise ValueError('missing or unknown capability')
    for name in required:
        record=evidence.get(name,{})
        if record.get('passed') is not True or record.get('contract_sha256')!=contract.signature:
            raise ValueError('capability lacks bound evidence: '+name)
    if solver=='pcg' and (contract.operator_kind!='solid_exact_hessian' or not spd_evidence or spd_evidence.get('contract_sha256')!=contract.signature or spd_evidence.get('positive_definite') is not True):
        raise ValueError('PCG needs state-bound SPD evidence; mixed blocks require a general solver')
    for name,callback in (callbacks or {}).items():
        if not callable(callback) or callback(contract) is not True:raise ValueError('handoff validation failed: '+name)
    return True

@dataclass(frozen=True)
class MixedBlockContract:
    """E owns an independent space and equations; no solid/SPD assumptions.

    Flat block order is displacement then pressure. Pressure is Pa, vector
    displacement m; the producer must identify the scaling of each residual.
    """
    parent_bundle_sha256:str
    space_sha256:str
    equation_sha256:str
    material_rule_sha256:str
    mass_rule_sha256:str
    boundary_sha256:str
    state_sha256:str
    extension_source_sha256:str
    displacement_size:int
    pressure_size:int
    residual_units:tuple
    device:str
    dimension:int=2
    dtype:str='float64'
    operator_kind:str='mixed_general'
    schema_version:int=2

    @property
    def signature(self):return digest(asdict(self))

    def validate(self,expected=None):
        for k,v in asdict(self).items():
            if k.endswith('sha256') and (not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)):
                raise ValueError('missing or invalid '+k)
        if self.schema_version!=2 or self.dtype!='float64' or self.operator_kind!='mixed_general' or self.dimension!=2 or not self.device:
            raise ValueError('unsupported mixed contract')
        if any(not isinstance(n,int) or isinstance(n,bool) or n<=0 for n in (self.displacement_size,self.pressure_size)) or self.displacement_size%2:
            raise ValueError('explicit 2D displacement and pressure block sizes required')
        if len(self.residual_units)!=2 or any(not isinstance(v,str) or not v for v in self.residual_units):
            raise ValueError('both mixed residual unit/scaling declarations are required')
        for k,v in (expected or {}).items():
            if getattr(self,k)!=v:raise ValueError('contract mismatch: '+k)
        return self
