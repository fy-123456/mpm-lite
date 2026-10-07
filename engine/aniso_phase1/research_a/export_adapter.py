"""A-owned fixed spatial contract; q denotes displacement, never position.

No history commit, time evolution, or change of basis is performed here.
"""
import numpy as np
from engine.aniso_phase1.high_order_space import HighOrderPotential, transpose_gradient
from engine.aniso_phase1.tensor_metrics import evaluate_gradient, sampling
from engine.aniso_phase1.tensor_reference import apply_axis


class FixedSpace:
    def __init__(self, problem, W):
        self.problem = problem
        self.W = np.array(W, copy=True)
        self.W.setflags(write=False)
        self.Q, self.lift = problem.Q, problem.lift
        self.nf = self.Q.shape[1]
        self.shape = (self.nf+self.W.shape[1], 3)
        self.potential = HighOrderPotential(problem.edges, problem.degree, problem.A, self.W,
                                            problem.Ks, problem.params, problem.fiber_tensor)

    def _check(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != self.shape or not np.isfinite(q).all():
            raise ValueError(f"Expected finite displacement coordinates {self.shape}")
        return q

    def carrier_displacement(self, q):
        q = self._check(q)
        return self.lift+self.Q@q[:self.nf]

    def total_coefficients(self, q):
        q = self._check(q)
        return np.vstack((self.problem.carrier_X+self.carrier_displacement(q), q[self.nf:]))

    def direction_coefficients(self, v):
        v = self._check(v)
        return np.vstack((self.Q@v[:self.nf], v[self.nf:]))

    def nodal_displacement(self, q):
        q = self._check(q)
        return self.problem.A@self.carrier_displacement(q)+self.W@q[self.nf:]

    def _sample(self, nodal, points):
        p = self.problem
        shape = tuple(p.degree*(len(e)-1)+1 for e in p.edges)
        field = nodal.reshape(*shape, 3)
        for k in range(3):
            field = apply_axis(sampling(p.edges[k], p.degree, points[k]), field, k)
        return field, evaluate_gradient((p.edges, p.degree, nodal), points)

    def evaluate(self, q, points):
        u, grad_u = self._sample(self.nodal_displacement(q), points)
        X = np.stack(np.meshgrid(*points, indexing="ij"), axis=-1)
        return X+u, np.eye(3)+grad_u

    def jvp(self, direction, points):
        v = self._check(direction)
        return self._sample(self.problem.A@(self.Q@v[:self.nf])+self.W@v[self.nf:], points)

    def vjp(self, position_dual, gradient_dual, points):
        """Euclidean adjoint; caller incorporates quadrature volume in duals."""
        p = self.problem
        shape = tuple(len(x) for x in points)
        fx, fF = np.asarray(position_dual), np.asarray(gradient_dual)
        if fx.shape != (*shape, 3) or fF.shape != (*shape, 3, 3):
            raise ValueError("Duals must match the Cartesian product probe grid")
        if not np.isfinite(fx).all() or not np.isfinite(fF).all():
            raise ValueError("Non-finite duals")
        out = fx
        for k in (2, 1, 0):
            out = apply_axis(sampling(p.edges[k], p.degree, points[k]).T, out, k)
        nodal = out.reshape(-1, 3)+transpose_gradient(fF, p.edges, p.degree, points, [np.ones(len(x)) for x in points])
        return np.vstack((self.Q.T@(p.A.T@nodal), self.W.T@nodal))

    def response(self, q, direction=None, order=None):
        result = self.potential.evaluate(self.total_coefficients(q),
                  None if direction is None else self.direction_coefficients(direction), order)
        nc = len(self.problem.carrier_X)
        out = dict(energy_J=result["U"], force=np.vstack((self.Q.T@result["force"][:nc], result["force"][nc:])))
        if direction is not None:
            out["tangent_action"] = np.vstack((self.Q.T@result["tangent_action"][:nc], result["tangent_action"][nc:]))
        return out
