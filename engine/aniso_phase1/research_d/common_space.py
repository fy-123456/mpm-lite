"""Frozen A-package adapter for common B/C/D inputs.

``q`` contains free displacement coordinates. B's tensor operator instead
accepts *full displacement coefficients*: use ``expand(q)`` for states and
``direction_coefficients(dq)`` for directions. Neither ``nodes`` nor its
adjoint inserts a lift or a reference position. No external archive is read.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..high_order_space import transpose_gradient
from ..tensor_metrics import evaluate_gradient, sampling
from ..tensor_reference import apply_axis, coordinates
from ..types import AnisotropicMaterialParams
from ..research_b.rules import readonly


_MEMBERS = frozenset(("geometry.npz", "carrier-basis.npz", "basis-raw.npz",
                      "basis-transform.npz", "test-vectors.npz"))


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _axes(values):
    result = tuple(readonly(v) for v in values)
    if len(result) != 3 or any(v.ndim != 1 or len(v) < 2 or
                             not np.isfinite(v).all() or np.any(np.diff(v) <= 0)
                             for v in result):
        raise ValueError("Three strictly increasing finite geometry axes required")
    return result


def _kron(matrices):
    return sp.kron(sp.kron(matrices[0], matrices[1], format="csr"), matrices[2], format="csr")


class CommonSpace:
    """Sparse/axis-factored reconstruction of one immutable A space package.

    ``signature`` is the SHA256 of the exact package manifest bytes. The
    manifest hashes all five package members and is checked before loading.
    A caller can pin ``expected_manifest_sha256`` when reopening a freeze.
    Existing package-member symlinks are followed and their target bytes are
    checked; relocating a package therefore does not change its identity.
    """

    def __init__(self, folder, *, expected_manifest_sha256=None):
        self.folder = Path(folder)
        manifest = self.folder / "space-package.json"
        self.signature = _sha(manifest)
        if expected_manifest_sha256 is not None and self.signature != expected_manifest_sha256:
            raise ValueError("Package manifest digest mismatch")
        self.metadata = json.loads(manifest.read_text())
        m = self.metadata
        if m.get("schema_version") != 1 or m.get("producer") != "research_a" or m.get("dtype") != "float64" or m.get("device") != "cpu":
            raise ValueError("Unsupported A package schema, producer, dtype, or device")
        if m.get("displacement_convention") != "x=X+u(q); F=I+grad(u)":
            raise ValueError("Unsupported displacement convention")
        if m.get("dof_order") != "rows: free carriers in original order, then local scalar basis; columns: x,y,z":
            raise ValueError("Unsupported degree-of-freedom order")
        if set(m.get("files", {})) != _MEMBERS:
            raise ValueError("Package must bind exactly the five required local members")
        for name, expected in m["files"].items():
            if Path(name).name != name or _sha(self.folder / name) != expected:
                raise ValueError(f"Package digest mismatch: {name}")
        with np.load(self.folder / "geometry.npz", allow_pickle=False) as g:
            self.edges = _axes([g[f"axis{k}"] for k in range(3)])
            self.p = int(g["degree"])
            self.carrier_X = readonly(g["carrier_X"])
            self.Ks, self.A = readonly(g["Ks"]), readonly(g["fiber_tensor"])
        if self.p < 1 or self.carrier_X.ndim != 2 or self.carrier_X.shape[1] != 3:
            raise ValueError("Invalid geometry or degree")
        self.n = len(self.carrier_X)
        self.shape = tuple(self.p*(len(e)-1)+1 for e in self.edges)
        with np.load(self.folder / "carrier-basis.npz", allow_pickle=False) as z:
            self.oldedges = _axes([z[f"axis{k}"] for k in range(3)])
            self.oldA = readonly(z["A"])
        self.oldshape = tuple(2*(len(e)-1)+1 for e in self.oldedges)
        if self.oldA.shape != (int(np.prod(self.oldshape)), self.n):
            raise ValueError("Carrier basis dimensions do not match bundled geometry")
        self.prolong = tuple(sampling(e, 2, x) for e, x in zip(self.oldedges, coordinates(self.edges, self.p)))
        self.raw = sp.load_npz(self.folder / "basis-raw.npz").tocsr()
        with np.load(self.folder / "basis-transform.npz", allow_pickle=False) as z:
            self.transform = readonly(z["transform"])
        if self.transform.ndim != 2 or self.raw.shape != (int(np.prod(self.shape)), self.transform.shape[0]):
            raise ValueError("Local basis dimensions do not match bundled geometry")
        if self.Ks.shape != (self.n, self.n) or self.A.shape != (3, 3):
            raise ValueError("Invalid stabilization or material tensor shape")
        if not all(np.isfinite(a).all() for a in (self.oldA, self.carrier_X, self.Ks, self.A, self.raw.data, self.transform)):
            raise ValueError("Non-finite package geometry, basis, or material tensor")
        # Immutable owners prevent accidental in-place changes within a solve.
        self.raw.data = readonly(self.raw.data)
        self.raw.indices = readonly(self.raw.indices, self.raw.indices.dtype)
        self.raw.indptr = readonly(self.raw.indptr, self.raw.indptr.dtype)
        self.ndof = self.n + self.transform.shape[1]
        self.reference = readonly(np.vstack((self.carrier_X, np.zeros((self.ndof-self.n, 3)))))
        material = m["material"]
        theta = np.deg2rad(material["fiber_angle_degrees"])
        self.params = AnisotropicMaterialParams(material["mu"], material["lam"], material["k_f"], [np.cos(theta), np.sin(theta), 0.])
        if not np.allclose(self.A, self.params.A0, atol=1e-12, rtol=1e-12):
            raise ValueError("Material angle and bundled fiber tensor disagree")
        bc = m["boundary_conditions"]
        box = np.asarray(bc["box"], dtype=float)
        if box.shape != (3, 2) or not np.array_equal(box, [[e[0], e[-1]] for e in self.edges]):
            raise ValueError("Boundary box and geometry disagree")
        left, right = np.asarray(bc["grips"], dtype=float)
        if not np.isfinite([left, right]).all() or not left < right:
            raise ValueError("Invalid rigid grip locations")
        free_carriers = np.flatnonzero((self.carrier_X[:, 0] > left) & (self.carrier_X[:, 0] < right))
        self.free_scalar_ids = readonly(np.r_[free_carriers, np.arange(self.n, self.ndof)], int)
        self.fixed_scalar_ids = readonly(np.setdiff1d(np.arange(self.ndof), self.free_scalar_ids), int)
        self.nfree_carrier = len(free_carriers)
        self.q_shape = (len(self.free_scalar_ids), 3)
        if list(self.q_shape) != m["q_shape"]:
            raise ValueError("Package free-coordinate dimensions disagree")
        lift = np.zeros((self.ndof, 3))
        lift[np.flatnonzero(self.carrier_X[:, 0] <= left)] = bc["left_displacement"]
        lift[np.flatnonzero(self.carrier_X[:, 0] >= right)] = bc["right_displacement"]
        if not np.isfinite(lift).all():
            raise ValueError("Non-finite prescribed displacement")
        self.lift = readonly(lift)
        self.boundary = dict(left_x=float(left), right_x=float(right),
                             left_displacement=list(bc["left_displacement"]),
                             right_displacement=list(bc["right_displacement"]))
        with np.load(self.folder / "test-vectors.npz", allow_pickle=False) as z:
            self.test_vectors = {k: readonly(z[k]) for k in z.files}
        self.q0 = readonly(self._check(self.test_vectors["q"], self.q_shape, "package q0"))

    @staticmethod
    def _check(value, shape, name):
        a = np.asarray(value, dtype=np.float64)
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f"{name} must have finite shape {shape}")
        return a

    def expand(self, q, lift=None):
        """Insert free displacement q and a prescribed full/carrier lift."""
        q = self._check(q, self.q_shape, "free displacement")
        if lift is None:
            out = self.lift.copy()
        else:
            lift = np.asarray(lift, dtype=np.float64)
            if lift.shape == (self.n, 3):
                lift = np.vstack((lift, np.zeros((self.ndof-self.n, 3))))
            out = self._check(lift, (self.ndof, 3), "lift").copy()
            if np.any(out[self.free_scalar_ids] != 0):
                raise ValueError("A prescribed lift must vanish on all free coordinates")
        out[self.free_scalar_ids] = q
        return out

    def direction_coefficients(self, dq):
        """Derivative of expand; no boundary lift or reference position."""
        return self.expand(dq, np.zeros((self.ndof, 3)))

    def restrict(self, full_force):
        return self._check(full_force, (self.ndof, 3), "full force")[self.free_scalar_ids].copy()

    def nodes(self, full_displacement):
        """Full displacement coefficients -> high-order nodal displacement."""
        q = self._check(full_displacement, (self.ndof, 3), "full displacement")
        u = (self.oldA @ q[:self.n]).reshape(*self.oldshape, 3)
        for k, B in enumerate(self.prolong):
            u = apply_axis(B, u, k)
        return u.reshape(-1, 3) + self.raw @ (self.transform @ q[self.n:])

    def adjoint(self, nodal_force):
        """Euclidean transpose of nodes, retaining constrained reactions."""
        force = self._check(nodal_force, (int(np.prod(self.shape)), 3), "nodal force")
        f = force.reshape(*self.shape, 3)
        for k in (2, 1, 0):
            f = apply_axis(self.prolong[k].T, f, k)
        return np.vstack((self.oldA.T @ f.reshape(-1, 3), self.transform.T @ (self.raw.T @ force)))

    def _points(self, points):
        if len(points) != 3:
            raise ValueError("Three Cartesian probe axes required")
        axes = tuple(np.asarray(x, dtype=np.float64) for x in points)
        if any(x.ndim != 1 or not len(x) or not np.isfinite(x).all() or
               np.any(x < e[0]) or np.any(x > e[-1]) for x, e in zip(axes, self.edges)):
            raise ValueError("Finite probe axes inside the reference box required")
        return axes

    def scalar_basis_at(self, point_axes, *, free=False):
        """Sample scalar displacement basis; caller chooses bounded probe slabs.

        Output is (number of Cartesian probes, number of full/free scalar DOF).
        This never constructs a dense high-order nodal basis. It is suitable
        for streaming a consistent mass Gram matrix over quadrature slabs.
        """
        points = self._points(point_axes)
        S = tuple(sampling(e, self.p, x) for e, x in zip(self.edges, points))
        carrier = _kron(tuple(s @ p for s, p in zip(S, self.prolong))) @ self.oldA
        local = (_kron(S) @ self.raw) @ self.transform
        full = np.column_stack((carrier, local))
        return full[:, self.free_scalar_ids] if free else full

    def _sample(self, nodal, points):
        out = nodal.reshape(*self.shape, 3)
        for k, (e, x) in enumerate(zip(self.edges, points)):
            out = apply_axis(sampling(e, self.p, x), out, k)
        return out, evaluate_gradient((self.edges, self.p, nodal), points)

    def evaluate(self, q, points):
        points = self._points(points)
        u, grad_u = self._sample(self.nodes(self.expand(q)), points)
        X = np.stack(np.meshgrid(*points, indexing="ij"), axis=-1)
        return X+u, np.eye(3)+grad_u

    def jvp(self, dq, points):
        points = self._points(points)
        return self._sample(self.nodes(self.direction_coefficients(dq)), points)

    def position_vjp(self, dual, points):
        points = self._points(points)
        out = self._check(dual, tuple(map(len, points))+(3,), "position dual")
        for k in (2, 1, 0):
            out = apply_axis(sampling(self.edges[k], self.p, points[k]).T, out, k)
        return self.restrict(self.adjoint(out.reshape(-1, 3)))

    def gradient_vjp(self, dual, points):
        points = self._points(points)
        dual = self._check(dual, tuple(map(len, points))+(3, 3), "gradient dual")
        force = transpose_gradient(dual, self.edges, self.p, points, [np.ones(len(x)) for x in points])
        return self.restrict(self.adjoint(force))

    def vjp(self, position_dual, gradient_dual, points):
        """Unweighted adjoint; physical quadrature weights belong in duals."""
        return self.position_vjp(position_dual, points) + self.gradient_vjp(gradient_dual, points)

    def response(self, q, direction=None, *, order=None):
        """Original potential using B's full-coefficient streaming operator."""
        from ..research_b.tensor import TensorMaterialOperator, TensorRule
        rule = TensorRule.uniform(self.edges, self.p+1 if order is None else order)
        result = TensorMaterialOperator(self, rule).evaluate(self.expand(q), None if direction is None else self.direction_coefficients(direction))
        out = dict(result, energy_J=result["U"], full_force=result["force"], force=self.restrict(result["force"]))
        if direction is not None:
            out["full_tangent_action"] = result["tangent_action"]
            out["tangent_action"] = self.restrict(result["tangent_action"])
        return out

    def maps_info(self):
        return dict(schema_version=1, package_manifest_sha256=self.signature,
                    full_scalar_dofs=self.ndof, carrier_scalar_dofs=self.n,
                    local_scalar_dofs=self.ndof-self.n, free_carrier_scalar_dofs=self.nfree_carrier,
                    free_scalar_ids=self.free_scalar_ids.tolist(), fixed_scalar_ids=self.fixed_scalar_ids.tolist(),
                    q_shape=list(self.q_shape), full_shape=[self.ndof, 3], nodal_shape=list(self.shape),
                    flatten_order="C order; scalar basis first, xyz component last",
                    full_coordinates="displacement coefficients [carrier u; local alpha]",
                    free_coordinates="free carrier displacement, then all local alpha",
                    affine_lift="package rigid-grip displacement; derivative has zero lift",
                    deformation_gradient="I + grad(u); position X + u; one identity only",
                    force_convention="positive energy gradient; mechanical internal force is its negative",
                    adjoint_convention="Euclidean; insert physical quadrature weights in duals",
                    basis_and_support_fixed=True, boundary=self.boundary)
