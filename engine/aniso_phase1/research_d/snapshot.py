"""Canonical physical-node adapter of the existing sparse implicit operator.

Sparse allocation order and arbitrary bases of repeated constraint eigenvalues
are implementation details: freeze the same physical state and compare in one
Cartesian ordering before assessing CPU/GPU equivalence.
"""
import numpy as np
from engine.sp_grid import B
from .identity import digest


def canonical_variational_check(probe,F,seed=42):
    p=probe;s=p.s;p.set_deformation(F)
    mapping=s.ndof2bijk[:p.n].numpy();block=s.block_xyz_by_id.numpy()[mapping[:,0]]
    local=mapping[:,1]
    coords=B*block+np.column_stack((local//(B*B),(local//B)%B,local%B))
    order=np.lexsort(coords.T[::-1]);canonical_nodes=coords[order]
    if len(np.unique(canonical_nodes,axis=0))!=p.n: raise ValueError('duplicate physical node in sparse map')
    def to_slot(a):
        out=np.empty((p.n,3));out[order]=np.asarray(a).reshape(p.n,3);return out
    def to_canonical(a):return np.asarray(a).reshape(p.n,3)[order].ravel()
    rng=np.random.default_rng(seed)
    initial=rng.normal(size=(p.n,3))*.01
    v=p.project(to_slot(initial));direction=p.project(to_slot(rng.normal(size=(p.n,3))))
    direction/=np.linalg.norm(direction)
    residual=p.residual(v);analytic=float(np.sum(residual*direction));action=p.tangent(direction)
    gradients={};tangents={}
    for eps in (1e-3,1e-4,1e-5,1e-6):
        plus=p.residual(v+eps*direction);ep=s.incremental_potential()
        minus=p.residual(v-eps*direction);em=s.incremental_potential()
        gradients[str(eps)]=abs((ep-em)/(2*eps)-analytic)/max(abs(analytic),1e-12)
        tangents[str(eps)]=float(np.linalg.norm((plus-minus)/(2*eps)-action)/max(np.linalg.norm(action),1e-12))
    p.residual(v)
    I=np.eye(3*p.n)
    Q=np.column_stack([to_canonical(p.project(to_slot(e))) for e in I])
    val,V=np.linalg.eigh((Q+Q.T)/2);Z=V[:,val>.5]
    masses=s.grid_m[:int(s.bcn)].numpy()
    m=np.array([masses[b,l//(B*B),(l//B)%B,l%B] for b,l in mapping])[order]
    full_mass=np.diag(np.repeat(m,3));L=np.linalg.cholesky(Z.T@full_mass@Z)
    matrices={};result=dict(potential_fd_errors=gradients,tangent_fd_errors=tangents,
        free_dofs=Z.shape[1],active_blocks=int(s.bcn),
        canonical_input_sha256=digest([F,initial,canonical_nodes]),
        dof_order='physical (x,y,z) node lexicographic, Cartesian components inner')
    for modified,name in ((False,'exact'),(True,'modified')):
        J=np.column_stack([to_canonical(p.tangent(to_slot(e),modified)) for e in I])
        reduced=Z.T@J@Z;symmetric=(reduced+reduced.T)/2;eig=np.linalg.eigvalsh(symmetric)
        normalized=np.linalg.solve(L,np.linalg.solve(L,symmetric).T).T
        scaled=np.linalg.eigvalsh((normalized+normalized.T)/2)
        result[name]=dict(symmetry_relative=float(np.linalg.norm(J-J.T)/np.linalg.norm(J)),
            min_eigenvalue=float(eig[0]),max_eigenvalue=float(eig[-1]),negative_eigenvalues=int(np.sum(eig<0)),
            mass_scaled_min=float(scaled[0]),mass_scaled_max=float(scaled[-1]))
        matrices[name]=J
    result['tangent_modification_relative']=float(np.linalg.norm(matrices['modified']-matrices['exact'])/np.linalg.norm(matrices['exact']))
    matrices.update(mass=Q@full_mass@Q,nodes=canonical_nodes,velocity=to_canonical(v),projection=Q)
    p.base=v
    return result,matrices
