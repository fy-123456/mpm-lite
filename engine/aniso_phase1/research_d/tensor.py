"""Frozen adapters of v22 TensorElastic/BoxElastic, plus an assembled oracle."""
from __future__ import annotations
import itertools
import numpy as np
from benchmarks.aniso_local_q3 import assemble
from engine.aniso_phase1.high_order_space import BoxElastic
from engine.aniso_phase1.beam_reference import reference_hessian
from .identity import BASELINE, digest


def readonly(value):
    a = np.array(value, dtype=np.float64, copy=True); a.setflags(write=False); return a


class FrozenTensor:
    def __init__(self, edges, degree=2, H=None,
                 faces=((True,True),(False,False),(False,False)), name='F45', displacement=.005):
        if len(edges) != 3 or any(np.ndim(e) != 1 or len(e) < 2 or
                not np.isfinite(e).all() or np.any(np.diff(e) <= 0) for e in edges):
            raise ValueError('three strictly increasing finite axes required')
        if not isinstance(degree,int) or degree < 1: raise ValueError('positive integer degree required')
        if len(faces) != 3 or any(len(f) != 2 or any(type(v) is not bool for v in f) for f in faces):
            raise ValueError('explicit physical/artificial Dirichlet faces required')
        H = reference_hessian(kf=200.,direction=(2**-.5,2**-.5,0.)) if H is None else H
        H = np.asarray(H).reshape(9,9)
        if not np.isfinite(H).all() or not np.allclose(H,H.T,rtol=1e-10,atol=1e-12):
            raise ValueError('finite symmetric material tangent required')
        self.edges, self.H = tuple(readonly(e) for e in edges), readonly(H)
        self.degree, self.faces, self.name = degree, tuple(tuple(f) for f in faces), name
        self.op = BoxElastic(self.edges,degree,self.H,self.faces)
        self.shape, self.free_shape = self.op.shape, self.op.free_shape
        if min(self.free_shape) < 1: raise ValueError('empty free subspace')
        self.size = 3*int(np.prod(self.free_shape))
        mask = np.zeros(self.shape,dtype=bool); mask[self.op.slices] = True
        self.free = np.flatnonzero(np.tile(mask.ravel(),3))
        self.nodes = np.array(list(itertools.product(*[a[0] for a in self.op.axes])))
        self.lift = np.zeros((3,*self.shape)); self.lift[0,-1] = displacement
        # Nonzero lifting is only the standard two-grip static experiment.
        self.displacement = float(displacement)
        if displacement and self.faces != ((True,True),(False,False),(False,False)):
            raise ValueError('local correction problems must have zero lifting')
        self.lift = readonly(self.lift.ravel())
        self.rhs = readonly(-self.op.apply(self.lift)[self.free])
        self.identity = dict(schema_version=1,baseline_sha256=BASELINE,name=name,
            edges=[e.tolist() for e in self.edges],degree=degree,material_tangent=self.H.tolist(),
            dirichlet_faces=self.faces,displacement_m=displacement,dtype='float64',
            coordinates='Cartesian reference meters',q_convention='displacement; x=X+u; F=I+grad(u)',
            dof_order='component,x,y,z; C order; free subspace only',
            volume='positive reference volume; physical free span, rigid grips restored separately',
            mass_included=False,stiffness_shift=0.,state_sha256=digest([self.lift,self.rhs]))
        self.key = digest(self.identity)

    def apply(self,v): return self.op.free_apply(v)
    def precondition(self,v): return self.op.precondition(v)

    def assembled(self):
        nodes, K = assemble(self.edges,self.degree,self.H)
        if not np.array_equal(nodes,self.nodes): raise RuntimeError('oracle node ordering mismatch')
        return K[self.free][:,self.free].tocsr()

    def field(self,x):
        u = self.lift.copy(); u[self.free] = x
        return u.reshape(3,-1).T

    def metrics(self,x):
        u = self.lift.copy(); u[self.free] = x
        f = self.op.apply(u); n = int(np.prod(self.shape))
        reaction = float(f.reshape(3,*self.shape)[0,-1].sum())
        energy = float(.5*u@f)
        return dict(reaction_N=reaction,energy_J=energy,
            true_residual=float(np.linalg.norm(f[self.free])),
            constraint_residual=float(np.max(abs((u-self.lift)[np.setdiff1d(np.arange(3*n),self.free)]))),
            work_identity_absolute_J=abs(energy-.5*self.displacement*reaction),
            finite=bool(np.isfinite(u).all() and np.isfinite(f).all()))


def solver_callbacks(problem):
    """Static integration adapter: q is the free displacement, lift is fixed."""
    from engine.aniso_phase1.research_contracts import SolverCallbacks
    return SolverCallbacks(problem.size,lambda q:problem.apply(q)-problem.rhs,
        lambda q,dq:problem.apply(dq),problem.precondition,lambda v:np.asarray(v).copy(),problem.key).validate()
