"""Fixed material cubature on CommonSpace with explicit dynamic lifts."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from ...tensor_metrics import quadrature_axis, sampling
from ...tensor_reference import apply_axis
from ..rules import readonly
from .material import ConstitutiveState, MaterialSource, digest


@dataclass(frozen=True)
class FixedRule:
    orders: tuple

    def __post_init__(self):
        if not self.orders or any(isinstance(o, (bool, np.bool_)) or int(o) != o or not 1 <= o <= 12 for o in self.orders):
            raise ValueError('Gauss orders must be integers in [1,12]')
        object.__setattr__(self, 'orders', tuple(int(o) for o in self.orders))

    @classmethod
    def uniform(cls, space, order):
        return cls((order,)*(len(space.edges[0])-1))

    @property
    def signature(self):
        return digest(dict(schema='B-fixed-cell-Gauss-v1', orders=self.orders))

    def fallback(self, full, slabs):
        if len(self.orders) != len(full.orders):
            raise ValueError('fallback partition mismatch')
        orders = list(self.orders)
        for cell in slabs:
            if isinstance(cell, bool) or not isinstance(cell, (int, np.integer)) or not 0 <= cell < len(orders):
                raise ValueError('unknown fallback slab')
            orders[cell] = full.orders[cell]
        return FixedRule(tuple(orders))


def _real(value, shape, name):
    a = np.asarray(value)
    if a.dtype.kind not in 'fiu' or a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f'{name} requires finite real shape {shape}')
    return np.asarray(a, dtype=np.float64)


class CommonMaterialOperator:
    """Pure evaluation; forces are positive energy gradients.

    The operator holds quadrature/basis caches only. Rules and material sources
    are frozen. It owns no time, mass, evolving material history or acceptance.
    """
    def __init__(self, space, rule, source=None, *, boundary_protocol='explicit-full-lift-v1', min_detF=.15):
        started = perf_counter()
        source = source or MaterialSource(space_sha256=space.signature)
        if source.space_sha256 != space.signature:
            raise ValueError('material source names another space')
        if len(rule.orders) != len(space.edges[0])-1:
            raise ValueError('rule must cover every disjoint x slab')
        if not isinstance(boundary_protocol, str) or not boundary_protocol or not np.isfinite(min_detF) or min_detF <= 0:
            raise ValueError('explicit boundary identity and positive detF domain required')
        self.space, self._rule, self._source = space, rule, source
        self.boundary_protocol, self.min_detF = boundary_protocol, float(min_detF)
        self._identity = dict(space=space.signature, source=source.signature, rule=rule.signature,
                              boundary=boundary_protocol, dtype='float64', device='cpu', min_detF=self.min_detF)
        self.signature = digest(self._identity)
        self._slabs = []
        self.point_count = 0
        for cell, order in enumerate(rule.orders):
            lo, hi = space.edges[0][cell:cell+2]
            edges_x = sorted([lo, hi]+[b for b in source.breaks if lo < b < hi])
            # Every original slab is integrated once. Physical jumps are split
            # inside it, never smeared or averaged into a direction tensor.
            qs = [quadrature_axis(e, order) for e in (edges_x, *space.edges[1:])]
            points, weights = [q[0] for q in qs], [q[1] for q in qs]
            X = np.stack(np.meshgrid(*points, indexing='ij'), axis=-1).reshape(-1, 3)
            V = (weights[0][:, None, None]*weights[1][None, :, None]*weights[2][None, None, :])
            B = tuple(tuple(sampling(e, space.p, x, deriv) for deriv in (False, True)) for e, x in zip(space.edges, points))
            self._slabs.append((B, readonly(X), readonly(V), tuple(map(len, points))))
            self.point_count += len(X)
        self.construction_seconds = perf_counter()-started

    @property
    def rule(self):
        return self._rule

    @property
    def source(self):
        return self._source

    def check_identity(self, *, source_sha256=None, space_sha256=None, cache_sha256=None):
        current = dict(space=self.space.signature, source=self.source.signature, rule=self.rule.signature,
                       boundary=self.boundary_protocol, dtype='float64', device='cpu', min_detF=self.min_detF)
        if digest(current) != self.signature:
            raise ValueError('operator identity changed after cache construction')
        for actual, expected in ((self.source.signature, source_sha256), (self.space.signature, space_sha256), (self.signature, cache_sha256)):
            if expected is not None and actual != expected:
                raise ValueError('source, space or cache identity mismatch')

    @staticmethod
    def _gradient(u, B):
        result = []
        for j in range(3):
            a = u
            for k in range(3):
                a = apply_axis(B[k][k == j], a, k)
            result.append(a)
        return np.stack(result, axis=-1)

    @staticmethod
    def _adjoint(dual, B, V, shape):
        force = np.zeros((*shape, 3))
        for j in range(3):
            a = dual[..., j]*V[..., None]
            for k in (2, 1, 0):
                a = apply_axis(B[k][k == j].T, a, k)
            force += a
        return force

    def evaluate_free(self, q, *, lift, directions=None):
        """Lift is mandatory and never inserted into tangent directions."""
        if lift is None:
            raise ValueError('explicit full displacement lift is required')
        s = self.space
        q = _real(q, s.q_shape, 'free displacement')
        lift = _real(lift, (s.ndof, 3), 'full lift')
        full = s.expand(q, lift)
        dirs = None if directions is None else {k: s.direction_coefficients(_real(v, s.q_shape, 'free direction')) for k, v in directions.items()}
        return self.evaluate_full(full, directions=dirs)

    def evaluate_full(self, displacement, *, directions=None):
        self.check_identity()
        s = self.space
        q = _real(displacement, (s.ndof, 3), 'full displacement')
        directions = {} if directions is None else directions
        if not isinstance(directions, dict) or any(not isinstance(k, str) or not k for k in directions):
            raise ValueError('named full tangent directions required')
        dirs = {k: _real(v, (s.ndof, 3), 'full direction') for k, v in directions.items()}
        times = dict(mapping=0., basis=0., material=0., assembly=0., tangent=0.)
        t = perf_counter()
        u = s.nodes(q).reshape(*s.shape, 3)
        du = {k: s.nodes(d).reshape(*s.shape, 3) for k, d in dirs.items()}
        times['mapping'] += perf_counter()-t
        force = np.zeros_like(u)
        actions = {k: np.zeros_like(u) for k in dirs}
        weak, energies, tangent_work = [], [], {k: [] for k in dirs}
        minimum = float('inf')
        for B, X, V, pshape in self._slabs:
            t = perf_counter(); F = np.eye(3)+self._gradient(u, B); times['basis'] += perf_counter()-t
            t = perf_counter()
            constitutive = ConstitutiveState(F.reshape(-1, 3, 3), X, self.source, min_detF=self.min_detF, tangent=bool(dirs))
            times['material'] += perf_counter()-t
            minimum = min(minimum, constitutive.minimum)
            energies.append(float(V.ravel() @ constitutive.energy))
            probes = np.column_stack((np.ones(len(X)), X))
            weak.append(np.einsum('p,pk,pij->kij', V.ravel(), probes, constitutive.P))
            t = perf_counter(); force += self._adjoint(constitutive.P.reshape(*pshape, 3, 3), B, V, s.shape); times['assembly'] += perf_counter()-t
            for name in dirs:
                t = perf_counter()
                dF = self._gradient(du[name], B).reshape(-1, 3, 3)
                dP = constitutive.action(dF)
                actions[name] += self._adjoint(dP.reshape(*pshape, 3, 3), B, V, s.shape)
                tangent_work[name].append(float(np.einsum('p,pij,pij->', V.ravel(), dF, dP)))
                times['tangent'] += perf_counter()-t
        t = perf_counter()
        material_force = s.adjoint(force.reshape(-1, 3))
        y = s.reference[:s.n]+q[:s.n]
        stabilization_force = np.zeros_like(q); stabilization_force[:s.n] = s.Ks @ y
        stabilization = .5*float(np.sum(y*stabilization_force[:s.n]))
        total_force = material_force+stabilization_force
        tangent_results = {}
        for name, a in actions.items():
            material_action = s.adjoint(a.reshape(-1, 3))
            stabilization_action = np.zeros_like(q); stabilization_action[:s.n] = s.Ks @ dirs[name][:s.n]
            full_action = material_action+stabilization_action
            tangent_results[name] = dict(full=full_action, free=s.restrict(full_action), material=material_action,
                                         stabilization=stabilization_action, slab_work=np.array(tangent_work[name]))
        times['assembly'] += perf_counter()-t
        result = dict(energy_J=float(sum(energies))+stabilization, material_energy_J=float(sum(energies)),
                      stabilization_energy_J=stabilization, full_force=total_force, free_force=s.restrict(total_force),
                      material_force=material_force, stabilization_force=stabilization_force,
                      weak_moments=np.array(weak), slab_energy=np.array(energies), tangents=tangent_results,
                      min_detF=minimum, point_count=self.point_count, source_sha256=self.source.signature,
                      rule_sha256=self.rule.signature, cache_sha256=self.signature, timing_seconds=times)
        if not np.isfinite(total_force).all() or not np.isfinite(result['energy_J']) or any(not np.isfinite(v['full']).all() for v in tangent_results.values()):
            raise ValueError('non-finite integrated response')
        return result
