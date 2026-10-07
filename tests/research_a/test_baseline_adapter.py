import tempfile
from pathlib import Path
import unittest
import numpy as np
from benchmarks.research_a.protocol import write_json, run_path
from tests.research_a.helpers import small_problem
from engine.aniso_phase1.research_a.support_family import support_family
from engine.aniso_phase1.high_order_space import local_box
from benchmarks.aniso_v21_common import hessian
from benchmarks.aniso_local_q3 import assemble


class BaselineAdapterTests(unittest.TestCase):
    def test_reduced_equilibrium_and_original_constraints(self):
        p = small_problem()
        field, records = p.equilibrate(np.empty((p.n, 0)))
        self.assertTrue(all(r["static_passed"] and r["zero_modes"] == 0 and r["negative_modes"] == 0 for r in records.values()))
        self.assertLess(records["F45"]["independent_energy_error_J"], 1e-10)
        self.assertLess(records["F45"]["independent_reaction_error_N"], 1e-8)
        X = p.nodes()
        np.testing.assert_allclose(field["u"][X[:, 0] <= .25], 0., atol=1e-9)
        np.testing.assert_allclose(field["u"][X[:, 0] >= .75], np.tile([.005, 0., 0.], (np.sum(X[:, 0] >= .75), 1)), atol=1e-9)

    def test_supports_match_principal_operator_and_natural_faces(self):
        edges = [np.array([.125, .25, .375, .5, .625, .75, .875]),
                 np.array([.375, .4375, .5, .5625, .625]), np.array([.375, .5, .625])]
        X, K = assemble(edges, 2, hessian("F45"))
        rng = np.random.default_rng(12)
        for name in ("v22-original", "v22-overlap", "wide-overlap", "fiber-rect"):
            for definition in (support_family(name)[0], support_family(name)[-1]):
                idx, op, meta = local_box(edges, 2, definition, hessian("F45"))
                ids = np.concatenate([idx+a*len(X) for a in range(3)])
                v = rng.normal(size=len(ids))
                np.testing.assert_allclose(op.free_apply(v), K[ids][:, ids]@v, rtol=2e-7, atol=2e-8)
                self.assertEqual(meta["dirichlet_faces"][0], (True, True))

    def test_append_only_results_and_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "record.json"
            write_json(target, {"passed": True})
            with self.assertRaises(FileExistsError):
                write_json(target, {"passed": False})
            self.assertIn("true", target.read_text())
        for name in ("../v22", "/tmp/escape", ".", ""):
            with self.assertRaises(ValueError):
                run_path(name)
