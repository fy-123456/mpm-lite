"""Consistent reference-point inertia on the frozen A basis.

The scalar mass is T.T (Mx kron My kron Mz) T, applied without a dense
global basis. All carrier/local cross terms and constrained rows survive.
No APIC micro-inertia, lumping, stabilization mass, or time integrator.
"""
import numpy as np
import scipy.sparse as sp
from ..tensor_metrics import quadrature_axis, sampling
from ..tensor_reference import apply_axis


class PointInertia:
    def __init__(self, space, *, density=1., order=None):
        self.space = space
        self.density = float(density)
        self.order = space.p + 1 if order is None else order
        if not np.isfinite(self.density) or self.density <= 0:
            raise ValueError("Positive finite reference density required")
        if isinstance(self.order, bool) or int(self.order) != self.order or self.order < 1:
            raise ValueError("Positive integer mass quadrature order required")
        self.axes = []
        for edges in space.edges:
            x, w = quadrature_axis(edges, int(self.order))
            B = sampling(edges, space.p, x)
            self.axes.append((B.T @ sp.diags(w) @ B).tocsr())
        self.shape = tuple(space.p*(len(e)-1)+1 for e in space.edges)
        self.volume = float(np.prod([e[-1]-e[0] for e in space.edges]))

    def apply(self, full_velocity):
        v = np.asarray(full_velocity, dtype=float)
        if v.shape != (self.space.ndof, 3) or not np.isfinite(v).all():
            raise ValueError("Finite full velocity coefficients required")
        out = self.space.nodes(v).reshape(*self.shape, 3)
        for k, M in enumerate(self.axes):
            out = apply_axis(M, out, k)
        return self.density*self.space.adjoint(out.reshape(-1, 3))

    def energy(self, full_velocity):
        v = np.asarray(full_velocity, dtype=float)
        return .5*float(np.sum(v*self.apply(v)))

    def contract(self):
        return dict(model="fixed_reference_continuum_point_inertia",
                    density_kg_m3=self.density, reference_volume_m3=self.volume,
                    total_mass_kg=self.volume*self.density, quadrature_order=int(self.order),
                    formula="M=T^T (Mx tensor My tensor Mz) T",
                    full_carrier_local_cross_terms=True, mass_lumping=False,
                    independent_APIC_micro_inertia=False, geometry_dependent_mass=False,
                    material_quadrature_independent=True,
                    exact_polynomial_condition="per-cell Qp basis: order >= p+1 integrates N_i*N_j",
                    motion_solver_implemented=False)
