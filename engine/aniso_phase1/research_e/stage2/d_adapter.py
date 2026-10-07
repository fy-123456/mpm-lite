"""Exact RT0 flux elimination for D's genuine two-block mixed contract.

This is an algebraic adapter, not a changed Darcy model. Reconstructed flux
must satisfy the original three-block equations. D's source is a separately
hashed read-only input; no D public file is edited by E.
"""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from .handoff import save,sha
from .transaction import digest_value
from .residual import block_residual


def consume(folder,metadata,M,v,contract_source):
    path=Path(contract_source)
    if not path.is_file():return dict(passed=False,status='not_run: D mixed contract unavailable',D_solver_certified=False)
    spec=importlib.util.spec_from_file_location('e_readonly_d_contract',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    if not hasattr(module,'MixedBlockContract'):
        return dict(passed=False,status='not_run: D has no two-block mixed contract',D_solver_certified=False,source_sha256=sha(path))
    nu=len(v['free_u']);nc=len(v['old_p']);nup=nu+nc
    H=M[nup:,nup:].tocsc();B=M[nu:nup,nup:]/metadata['dt'];factor=splu(H)
    invBt=factor.solve(B.T.toarray());invR=factor.solve(v['rhs'][nup:])
    reduced=M[:nup,:nup].toarray();reduced[nu:,nu:]+=metadata['dt']*(B@invBt)
    rhs=v['rhs'][:nup].copy();rhs[nu:]-=metadata['dt']*(B@invR)
    x=np.linalg.solve(reduced,rhs);flux=invR+invBt@x[nu:];full=np.r_[x,flux]
    residual=block_residual(M,v['rhs'],full,nu,nc)
    direction=v['direction'][:nup];full_direction=np.r_[direction,invBt@direction[nu:]]
    action_difference=float(np.max(np.abs((M@full_direction)[:nup]-reduced@direction)))
    contract=module.MixedBlockContract(parent_bundle_sha256=metadata['parent_bundle_sha256'],space_sha256=metadata['space_sha256'],
        equation_sha256=digest_value({'equations':metadata['equations'],'dt':metadata['dt'],'elimination':'q=H^-1(r+B.T p)'}),
        material_rule_sha256=digest_value({'Q2_material_order':3,'RT0_resistance_order':2}),
        mass_rule_sha256=digest_value({'inertia':'none; quasistatic','storage':'S*reference_cell_volume'}),
        boundary_sha256=metadata['boundary_sha256'],state_sha256=metadata['initial_state_sha256'],
        extension_source_sha256=metadata['extension_source_sha256'],displacement_size=nu,pressure_size=nc,
        residual_units=('mechanics N / 0.1 N','mass m^3 / 1 m^3'),device='cpu')
    contract.validate()
    expected=asdict(contract)
    evidence={'coupled_physics':{'passed':all(e['passed'] for k,e in metadata['physics_evidence'].items() if k.startswith(('E5','E6','E7','E8'))),
                                  'contract_sha256':contract.signature}}
    accepted=module.validate_handoff(contract,expected=expected,capabilities=['coupled_physics'],required=['coupled_physics'],evidence=evidence,
        solver='general',callbacks={'original_three_block_residual':lambda _:bool(residual['maximum']<1e-8 and action_difference<1e-10)})
    rejected={}
    for key,value in [('space_sha256','0'*64),('boundary_sha256','0'*64),('equation_sha256','0'*64),('residual_units',('cm','Pa'))]:
        try:contract.validate(dict(expected,**{key:value}));rejected[key]=False
        except ValueError:rejected[key]=True
    try:
        module.validate_handoff(contract,expected=expected,capabilities=[],required=[],evidence={},solver='pcg',
            spd_evidence={'contract_sha256':contract.signature,'positive_definite':True});rejected['false_SPD_PCG']=False
    except ValueError:rejected['false_SPD_PCG']=True
    if folder is not None:
        folder=Path(folder);folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'condensed.npz',matrix=reduced,rhs=rhs,solution=x,flux=flux,direction=direction,action=reduced@direction)
        save(folder/'D-mixed-contract.json',dict(contract=asdict(contract),signature=contract.signature,D_source_sha256=sha(path),evidence=evidence))
    return dict(passed=bool(accepted and all(rejected.values()) and residual['maximum']<1e-8 and action_difference<1e-10),
        D_contract_consumed=True,D_solver_certified=False,D_source_sha256=sha(path),contract_sha256=contract.signature,
        exact_elimination='RT0 q=H^-1(r+B.T p); reduced mass C+dt B H^-1 B.T; original equations and dt unchanged',
        original_residual_blocks=residual,action_absolute_difference=action_difference,
        full_solution_relative_difference=float(np.linalg.norm(full-v['solution'])/np.linalg.norm(v['solution'])),
        rejection_controls=rejected,status='actual D MixedBlockContract and validate_handoff in an independent source copy; E CPU direct solve, no D backend/GPU certification')
