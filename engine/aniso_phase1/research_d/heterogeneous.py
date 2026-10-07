"""Piecewise-constant material slabs assembled from the unchanged Qp energy.

This explicitly assembled control is deliberately separate from the constant-H
matrix-free GPU tensor backend. No average tangent replaces the physical K.
"""
import numpy as np
import scipy.sparse as sp
from benchmarks.aniso_local_q3 import assemble
from .tensor import FrozenTensor,readonly
from .identity import digest


class FrozenHeterogeneous(FrozenTensor):
    def __init__(self,edges,degree,tangents,name='heterogeneous'):
        H=np.asarray(tangents,dtype=float)
        if H.shape!=(len(edges[0])-1,9,9): raise ValueError('one exact tangent per x slab required')
        lengths=np.diff(edges[0]); avg=np.einsum('s,sij->ij',lengths/lengths.sum(),H)
        super().__init__(edges,degree,avg,name=name)
        rows=[]; cols=[]; values=[]; n=int(np.prod(self.shape))
        for slab,h in enumerate(H):
            _,K=assemble([self.edges[0][slab:slab+2],*self.edges[1:]],degree,h)
            ids=np.arange(n).reshape(self.shape)[slab*degree:(slab+1)*degree+1].ravel()
            dofs=np.concatenate([ids+a*n for a in range(3)]); coo=K.tocoo()
            rows.append(dofs[coo.row]); cols.append(dofs[coo.col]); values.append(coo.data)
        self.full_matrix=sp.csr_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(3*n,3*n))
        self.matrix=self.full_matrix[self.free][:,self.free].tocsr()
        self.rhs=readonly(-(self.full_matrix@self.lift)[self.free])
        self.identity.update(material_distribution=H.tolist(),preconditioner='volume-average tensor only; exact heterogeneous physical K',
                             state_sha256=digest([self.lift,self.rhs]),representation='assembled heterogeneous CSR')
        self.key=digest(self.identity)
    def apply(self,v):return self.matrix@v
    def assembled(self):return self.matrix.copy()
    def metrics(self,x):
        u=self.lift.copy();u[self.free]=x;f=self.full_matrix@u
        reaction=float(f.reshape(3,*self.shape)[0,-1].sum()); energy=float(.5*u@f)
        return dict(reaction_N=reaction,energy_J=energy,true_residual=float(np.linalg.norm(f[self.free])),
                    constraint_residual=0.,work_identity_absolute_J=abs(energy-.5*self.displacement*reaction),
                    finite=bool(np.isfinite(u).all() and np.isfinite(f).all()))
