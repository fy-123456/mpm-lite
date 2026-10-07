"""Private, fail-closed 2D mixed operator package. D remains the public owner."""
import hashlib
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from .residual import block_residual
from .transaction import solver_identity, digest_value


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def export(folder,solver,old,dt,load,boundary,parent_sha,extension_sha,protocol_sha,physics_evidence):
    folder=Path(folder);folder.mkdir()
    s,f=solver.solid,solver.flow
    M,rhs,free,_,_,_=solver._system(old,dt,load,boundary,0.)
    result=solver.step(old,dt,load,boundary)
    for name,a in dict(A=s.A,G=s.G,C=solver.C,B=f.B,H=f.H,mixed=M).items():sp.save_npz(folder/(name+'.npz'),a)
    x=np.r_[result.state.u[s.free],result.state.p,result.flux[free]]
    direction=np.random.default_rng(20260930).normal(size=M.shape[0])
    np.savez_compressed(folder/'vectors.npz',rhs=rhs,solution=x,direction=direction,action=M@direction,
        old_u=old.u,old_p=old.p,old_time=old.time,load=load,free_u=s.free,free_flux=free,
        fixed_u=s.fixed,centers=f.grid.centers,nodes=s.nodes,permeability=f.K,volumes=np.full(f.grid.nc,f.grid.volume))
    config,signature=solver_identity(solver)
    bc=[[a,b,k,v] for (a,b),(k,v) in sorted(boundary.items())]
    contract=dict(schema_version=2,parent_bundle_sha256=parent_sha,space_sha256=signature,
        extension_source_sha256=extension_sha,protocol_sha256=protocol_sha,physics_evidence=physics_evidence,
        config=config,boundary=bc,boundary_sha256=digest_value(bc),dt=dt,dtype='float64',device='cpu',
        units={'u':'m','p':'Pa','flux':'m^3/s','dt':'s'},unknowns=['u[free_u]','p[cell]','flux[free_flux]'],
        block_shapes={'u_full':[s.ndof],'u_free':[len(s.free)],'p':[f.grid.nc],'flux_full':[f.grid.nf],'flux_free':[len(free)]},
        block_units={'A':'N/m','G':'m^2','C':'m^3/Pa','B':'1','H':'Pa s/m^3'},
        equations=['A u - G.T p = load','G delta_u + C delta_p + dt B flux = dt source_volume','H flux - B.T p = r'],
        coordinates='independent 2D displacement, P0 pressure, integrated signed RT0 face flux',
        operator_kind='mixed_general',symmetric=False,positive_definite=False,pressure_gauge='pressure drainage boundary fixes gauge',
        admissible_solvers=['direct','GMRES'],validated_solvers=['direct','fixed_stress'],
        row_scales={'mechanics_N':.1,'mass_m3':1.,'darcy_Pa':.1},
        initial_state_sha256=digest_value(dict(u=old.u.tolist(),p=old.p.tolist(),time=old.time)),
        files={p.name:sha(p) for p in folder.iterdir()})
    save(folder/'operator-package.json',contract)
    return contract,sha(folder/'operator-package.json')


def load(folder, *, expected_manifest_sha256, expected):
    folder=Path(folder)
    if sha(folder/'operator-package.json')!=expected_manifest_sha256:raise ValueError('mixed package identity mismatch')
    m=json.loads((folder/'operator-package.json').read_text())
    for k in ('parent_bundle_sha256','space_sha256','boundary_sha256','dt','units','extension_source_sha256','protocol_sha256'):
        if k not in expected or m.get(k)!=expected[k]:raise ValueError('missing/wrong expected '+k)
    if (m.get('schema_version')!=2 or m.get('dtype')!='float64' or m.get('device')!='cpu'
        or m.get('operator_kind')!='mixed_general' or m.get('positive_definite') is not False
        or m.get('symmetric') is not False or 'PCG' in m.get('admissible_solvers',[])
        or m.get('unknowns')!=['u[free_u]','p[cell]','flux[free_flux]']):raise ValueError('incompatible mixed solver/coordinate semantics')
    if digest_value(m['config'])!=m['space_sha256'] or digest_value(m['boundary'])!=m['boundary_sha256']:
        raise ValueError('unbound E configuration or boundary')
    required={'A.npz','G.npz','C.npz','B.npz','H.npz','mixed.npz','vectors.npz'}
    if set(m['files'])!=required:raise ValueError('incomplete mixed package')
    for name,digest in m['files'].items():
        path=folder/name
        if Path(name).name!=name or not path.resolve().is_relative_to(folder.resolve()) or sha(path)!=digest:
            raise ValueError('missing, changed or external mixed package member')
    matrices={k:sp.load_npz(folder/(k+'.npz')) for k in ('A','G','C','B','H','mixed')}
    if any(not np.isfinite(a.data).all() for a in matrices.values()):raise ValueError('nonfinite operator')
    with np.load(folder/'vectors.npz',allow_pickle=False) as z:v={k:z[k].copy() for k in z.files}
    if any(not np.isfinite(a).all() for a in v.values()):raise ValueError('nonfinite vectors')
    iu,iq=v['free_u'],v['free_flux'];nu=len(iu);nc=len(v['old_p']);A,G,C,B,H=(matrices[k] for k in ('A','G','C','B','H'))
    rebuilt=sp.bmat([[A[iu][:,iu],-G[:,iu].T,None],[G[:,iu],C,m['dt']*B[:,iq]],[None,-B[:,iq].T,H[iq][:,iq]]],format='csc')
    delta=rebuilt-matrices['mixed']
    if np.max(np.abs(delta.data),initial=0.)>1e-12:raise ValueError('mixed block disagrees with declared blocks')
    if digest_value(dict(u=v['old_u'].tolist(),p=v['old_p'].tolist(),time=float(v['old_time'])))!=m['initial_state_sha256']:
        raise ValueError('initial state identity mismatch')
    residual=block_residual(rebuilt,v['rhs'],v['solution'],nu,nc)
    if residual['maximum']>1e-8 or not np.allclose(rebuilt@v['direction'],v['action'],rtol=1e-12,atol=1e-12):
        raise ValueError('reference residual/action mismatch')
    return m,rebuilt,v,residual
