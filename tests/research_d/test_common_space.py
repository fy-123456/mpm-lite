"""Small independent A/B consistency checks for the frozen package adapter."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp

from engine.aniso_phase1.research_a.baseline_adapter import Problem
from engine.aniso_phase1.research_a.export_adapter import FixedSpace
from engine.aniso_phase1.research_d.common_space import CommonSpace
from engine.aniso_phase1.tensor_reference import coordinates, interpolate
from engine.aniso_phase1.types import AnisotropicMaterialParams


def small_package(folder):
    """A's actual FixedSpace/export schema on an entirely synthetic tiny grid."""
    edges = [np.array([.125, .25, .5, .75, .875]), np.array([.375, .625]), np.array([.375, .625])]
    degree = 3
    carrier = np.stack(np.meshgrid(*coordinates(edges, 2), indexing="ij"), axis=-1).reshape(-1, 3)
    oldA = np.eye(len(carrier))
    A = interpolate(edges, 2, oldA, edges, degree)
    nodes = np.stack(np.meshgrid(*coordinates(edges, degree), indexing="ij"), axis=-1).reshape(-1, 3)
    raw = np.zeros((len(nodes), 2))
    interior = np.flatnonzero((nodes[:, 0] > .25) & (nodes[:, 0] < .75))
    raw[interior[3], 0], raw[interior[-4], 1] = .3, .7
    transform = np.array([[1., .2], [-.1, .8]])
    W = raw @ transform
    params = AnisotropicMaterialParams(10., 20., 200., np.array([1., 1., 0.]))
    Ks = .1*np.eye(len(carrier))
    problem = Problem(edges, degree, A, carrier, Ks, params, params.A0, np.zeros((len(nodes), 3)))
    source = FixedSpace(problem, W)
    q = np.random.default_rng(2).normal(scale=1e-5, size=source.shape)
    direction = np.random.default_rng(3).normal(size=source.shape)
    direction /= np.linalg.norm(direction)
    points = [np.array([.27, .42, .64, .73]), np.array([.39, .57]), np.array([.4, .61])]
    x, F = source.evaluate(q, points)
    dx, dF = source.jvp(direction, points)
    response = source.response(q, direction)
    np.savez(folder/"geometry.npz", degree=degree, carrier_X=carrier, Ks=Ks,
             fiber_tensor=params.A0, **{f"axis{k}": a for k, a in enumerate(edges)})
    np.savez(folder/"carrier-basis.npz", A=oldA, **{f"axis{k}": a for k, a in enumerate(edges)})
    sp.save_npz(folder/"basis-raw.npz", sp.csr_matrix(raw))
    np.savez(folder/"basis-transform.npz", transform=transform)
    np.savez(folder/"test-vectors.npz", q=q, direction=direction, x=x, F=F, dx=dx, dF=dF,
             **response, **{f"points{k}": p for k, p in enumerate(points)})
    metadata = dict(schema_version=1, producer="research_a", dtype="float64", device="cpu",
                    displacement_convention="x=X+u(q); F=I+grad(u)",
                    dof_order="rows: free carriers in original order, then local scalar basis; columns: x,y,z",
                    material=dict(mu=10., lam=20., k_f=200., fiber_angle_degrees=45.),
                    boundary_conditions=dict(box=[[a[0], a[-1]] for a in edges], grips=[.25, .75],
                                             left_displacement=[0., 0., 0.], right_displacement=[.005, 0., 0.]),
                    q_shape=list(source.shape),
                    files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.npz"))})
    (folder/"space-package.json").write_text(json.dumps(metadata, indent=2))
    return source, q, direction, points


class CommonSpaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.source, self.q, self.direction, self.points = small_package(self.folder)
        self.space = CommonSpace(self.folder)

    def test_independent_A_maps_and_B_material_response(self):
        s = self.space
        for actual, expected in zip(s.evaluate(self.q, self.points), self.source.evaluate(self.q, self.points)):
            np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
        for actual, expected in zip(s.jvp(self.direction, self.points), self.source.jvp(self.direction, self.points)):
            np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-12)
        actual, expected = s.response(self.q, self.direction), self.source.response(self.q, self.direction)
        for key in ("energy_J", "force", "tangent_action"):
            np.testing.assert_allclose(actual[key], expected[key], atol=2e-9, rtol=2e-9)
        np.testing.assert_array_equal(s.q0, self.q)

    def test_lift_direction_and_full_reaction_semantics(self):
        s = self.space
        full, direction = s.expand(self.q), s.direction_coefficients(self.direction)
        np.testing.assert_array_equal(full[s.free_scalar_ids], self.q)
        np.testing.assert_array_equal(direction[s.fixed_scalar_ids], 0.)
        np.testing.assert_array_equal(full[s.fixed_scalar_ids], s.lift[s.fixed_scalar_ids])
        np.testing.assert_allclose(s.expand(self.q+self.direction)-full, direction, atol=1e-16)
        np.testing.assert_array_equal(s.restrict(full), self.q)
        np.testing.assert_array_equal(s.expand(self.q, np.zeros((s.n, 3)))[s.fixed_scalar_ids], 0.)
        bad_lift = s.lift.copy()
        bad_lift[s.free_scalar_ids[0], 0] = 1.
        with self.assertRaises(ValueError):
            s.expand(self.q, bad_lift)
        self.assertEqual(s.response(self.q)["full_force"].shape, (s.ndof, 3))

    def test_separate_position_gradient_and_nodal_adjoints(self):
        s = self.space
        rng = np.random.default_rng(19)
        dx, dF = s.jvp(self.direction, self.points)
        fx, fF = rng.normal(size=dx.shape), rng.normal(size=dF.shape)
        self.assertAlmostEqual(float(np.sum(dx*fx)), float(np.sum(self.direction*s.position_vjp(fx, self.points))), places=11)
        self.assertAlmostEqual(float(np.sum(dF*fF)), float(np.sum(self.direction*s.gradient_vjp(fF, self.points))), places=10)
        nodal = rng.normal(size=(int(np.prod(s.shape)), 3))
        full = rng.normal(size=(s.ndof, 3))
        self.assertAlmostEqual(float(np.sum(s.nodes(full)*nodal)), float(np.sum(full*s.adjoint(nodal))), places=10)
        np.testing.assert_allclose(s.vjp(fx, fF, self.points), s.position_vjp(fx, self.points)+s.gradient_vjp(fF, self.points))

    def test_scalar_sampling_and_energy_directional_derivative(self):
        s = self.space
        scalar = s.scalar_basis_at(self.points)
        X = np.stack(np.meshgrid(*self.points, indexing="ij"), axis=-1)
        x, _ = s.evaluate(self.q, self.points)
        np.testing.assert_allclose(scalar @ s.expand(self.q), (x-X).reshape(-1, 3), atol=2e-15)
        np.testing.assert_array_equal(s.scalar_basis_at(self.points, free=True), scalar[:, s.free_scalar_ids])
        eps = 1e-6
        plus, minus = s.response(self.q+eps*self.direction), s.response(self.q-eps*self.direction)
        base = s.response(self.q, self.direction)
        self.assertAlmostEqual((plus["energy_J"]-minus["energy_J"])/(2*eps), float(np.sum(base["force"]*self.direction)), places=8)
        np.testing.assert_allclose((plus["force"]-minus["force"])/(2*eps), base["tangent_action"], atol=1e-7, rtol=2e-6)

    def test_digest_pin_corruption_and_immutable_arrays(self):
        CommonSpace(self.folder, expected_manifest_sha256=self.space.signature)
        with self.assertRaisesRegex(ValueError, "manifest digest"):
            CommonSpace(self.folder, expected_manifest_sha256="0"*64)
        for arr in (self.space.oldA, self.space.transform, self.space.edges[0], self.space.q0, self.space.raw.data):
            with self.assertRaises(ValueError):
                arr.setflags(write=True)
        path = self.folder/"geometry.npz"
        path.write_bytes(path.read_bytes()+b"tampered")
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            CommonSpace(self.folder)

    def test_invalid_shape_probe_and_missing_manifest_member(self):
        s = self.space
        with self.assertRaises(ValueError):
            s.nodes(self.q)
        with self.assertRaises(ValueError):
            s.expand(np.full(s.q_shape, np.nan))
        with self.assertRaises(ValueError):
            s.scalar_basis_at([[-1.], [.5], [.5]])
        manifest = self.folder/"space-package.json"
        metadata = json.loads(manifest.read_text())
        del metadata["files"]["basis-raw.npz"]
        manifest.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "five required"):
            CommonSpace(self.folder)


if __name__ == "__main__":
    unittest.main()
