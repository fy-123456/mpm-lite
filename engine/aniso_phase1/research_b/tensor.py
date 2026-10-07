"""Streaming adapter for the actual v22 overlapping Q4 space.

Tensor Gauss samples belong to disjoint material cells, never to overlapping
kinematic patches. A constant direction is a degenerate conditional moment
rule. Sparse basis evaluation avoids the multi-GB dense [A,Z] reconstruction.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
import scipy.sparse as sp
from ..tensor_reference import apply_axis, coordinates
from ..tensor_metrics import sampling, quadrature_axis, evaluate_gradient
from ..high_order_space import transpose_gradient
from ..consistent_transfer import material_response
from ..history_increment import material_tangent
from .rules import readonly, digest_arrays


@dataclass(frozen=True)
class TensorRule:
    orders: tuple
    def __post_init__(self):
        if not self.orders or any(int(n) != n or n < 1 for n in self.orders):
            raise ValueError('positive integer Gauss orders required')
        object.__setattr__(self, 'orders', tuple(int(n) for n in self.orders))

    @property
    def signature(self):
        return hashlib.sha256(repr(('cell-Gauss-v1', self.orders)).encode()).hexdigest()

    @classmethod
    def uniform(cls, edges, order):
        return cls((order,)*(len(edges[0])-1))

    def fallback(self, full, partitions):
        if len(self.orders) != len(full.orders):
            raise ValueError('partition mismatch')
        result = list(self.orders)
        for k in partitions:
            if k < 0 or k >= len(result):
                raise ValueError('unknown material slab')
            result[k] = full.orders[k]
        return TensorRule(tuple(result))


class V22Space:
    def __init__(self, root):
        from benchmarks.aniso_v20_common import controlled_case
        base = root/'docs/results/lite-aniso-mainline'
        folder = base/'v22/multiscale/space/q4'
        state, energy, _, _, _ = controlled_case()
        with np.load(folder/'round6.npz') as z:
            self.edges = tuple(readonly(z[f'axis{k}']) for k in range(3))
            self.p = int(z['degree'])
            self.q = np.vstack((z['y'], z['local_coefficients']))
            saved = z['u']
        with np.load(base/'v19/space/reconstruction32.npz') as z:
            oldedges = [z[f'axis{k}'] for k in range(3)]
            self.oldA = readonly(z['A'])
        self.oldshape = tuple(2*(len(e)-1)+1 for e in oldedges)
        self.shape = tuple(self.p*(len(e)-1)+1 for e in self.edges)
        self.prolong = tuple(sampling(e, 2, x) for e, x in zip(oldedges, coordinates(self.edges, self.p)))
        self.raw = sp.load_npz(folder/'basis-raw.npz')
        with np.load(folder/'basis-transform.npz') as z:
            self.transform = readonly(z['transform'])
        self.n = len(state.Y)
        self.reference = np.vstack((state.Y, np.zeros((self.transform.shape[1], 3))))
        self.ndof = len(self.reference)
        self.Ks, self.params, self.A = energy.Ks, energy.params, readonly(energy.A[0])
        self.reconstruction_error = float(np.linalg.norm(self.nodes(self.q)-saved)/np.linalg.norm(saved))
        if self.reconstruction_error > 1e-8:
            raise ValueError('v22 round6 coefficients/basis reconstruction mismatch')
        self.signature = digest_arrays(self.q, self.transform, self.A, *self.edges)
        self.boundary = dict(left_x=.25, right_x=.75, right_displacement=.005)

    def nodes(self, q):
        q = np.asarray(q)
        if q.shape != (self.ndof, 3) or not np.isfinite(q).all():
            raise ValueError('finite v22 displacement coefficients required')
        u = (self.oldA @ q[:self.n]).reshape(*self.oldshape, 3)
        for k, B in enumerate(self.prolong):
            u = apply_axis(B, u, k)
        return u.reshape(-1, 3) + self.raw @ (self.transform @ q[self.n:])

    def adjoint(self, force):
        f = np.asarray(force).reshape(*self.shape, 3)
        for k in (2, 1, 0):
            f = apply_axis(self.prolong[k].T, f, k)
        return np.vstack((self.oldA.T @ f.reshape(-1, 3), self.transform.T @ (self.raw.T @ force)))

    def original_potential(self):
        """Execute unchanged HighOrderPotential for B1 wiring verification."""
        from ..high_order_space import HighOrderPotential
        space = self
        class Map:
            def __init__(self, transpose=False):
                self.transpose = transpose
            def __matmul__(self, q):
                return space.adjoint(q) if self.transpose else space.nodes(q)
            @property
            def T(self):
                return Map(not self.transpose)
        pot = HighOrderPotential.__new__(HighOrderPotential)
        pot.edges, pot.p, pot.T, pot.n = self.edges, self.p, Map(), self.n
        pot.Ks, pot.params, pot.fiber_tensor = self.Ks, self.params, self.A
        return pot


class TensorMaterialOperator:
    def __init__(self, space, rule):
        if len(rule.orders) != len(space.edges[0])-1:
            raise ValueError('rule must cover every disjoint material slab')
        self.space, self.rule = space, rule
        self.signature = hashlib.sha256((space.signature+rule.signature).encode()).hexdigest()

    def evaluate(self, q, direction=None):
        s = self.space
        u = s.nodes(q)
        du = None if direction is None else s.nodes(direction)
        force = np.zeros_like(u)
        action = np.zeros_like(u) if direction is not None else None
        U, samples, minimum = 0., 0, float('inf')
        # Per-x-cell constant and global-linear weak stress probes; retaining
        # each thin grip slab prevents cancellation between difficult regions.
        weak, slabs, tangents = [], [], []
        for cell, order in enumerate(self.rule.orders):
            axes = [s.edges[0][cell:cell+2], s.edges[1], s.edges[2]]
            qs = [quadrature_axis(e, order) for e in axes]
            points, weights = [a[0] for a in qs], [a[1] for a in qs]
            V = weights[0][:, None, None]*weights[1][None, :, None]*weights[2][None, None, :]
            F = np.eye(3) + evaluate_gradient((s.edges, s.p, u), points)
            flat = F.reshape(-1, 3, 3)
            det = np.linalg.det(flat)
            if not np.isfinite(det).all() or np.any(det <= 0):
                raise ValueError('non-positive or non-finite material Jacobian')
            minimum = min(minimum, float(det.min()))
            A = np.broadcast_to(s.A, flat.shape)
            psi, P = material_response(flat, A, s.params)
            if not np.isfinite(psi).all() or not np.isfinite(P).all():
                raise ValueError('non-finite material response')
            energy = float(V.ravel() @ psi)
            U += energy
            samples += len(flat)
            slabs.append(energy)
            force += transpose_gradient(P.reshape(F.shape), s.edges, s.p, points, weights)
            X = np.array(np.meshgrid(*points, indexing='ij')).reshape(3, -1).T
            probes = np.column_stack((np.ones(len(X)), X))
            weak.append(np.einsum('p,pk,pij->kij', V.ravel(), probes, P))
            if direction is not None:
                dF = evaluate_gradient((s.edges, s.p, du), points).reshape(-1, 3, 3)
                dP = material_tangent(flat, A, dF, s.params)
                action += transpose_gradient(dP.reshape(F.shape), s.edges, s.p, points, weights)
                tangents.append(float(np.einsum('p,pij,pij->', V.ravel(), dF, dP)))
        material_force = s.adjoint(force)
        y = s.reference[:s.n] + q[:s.n]
        stabilization = .5*float(np.sum(y*(s.Ks @ y)))
        total_force = material_force.copy()
        total_force[:s.n] += s.Ks @ y
        result = dict(U=U+stabilization, material_U=U, stabilization_U=stabilization,
                      force=total_force, material_force=material_force,
                      weak_moments=np.array(weak), slab_energy=np.array(slabs),
                      min_detF=minimum, material_calls=samples,
                      rule_signature=self.rule.signature)
        if direction is not None:
            result['material_tangent_action'] = s.adjoint(action)
            result['tangent_action'] = result['material_tangent_action'].copy()
            result['tangent_action'][:s.n] += s.Ks @ direction[:s.n]
            result['slab_tangent_work'] = np.array(tangents)
        if not all(np.isfinite(a).all() for a in (result['force'], U, result.get('tangent_action', 0.))):
            raise ValueError('non-finite integrated material response')
        return result
