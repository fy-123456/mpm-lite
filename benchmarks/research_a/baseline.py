"""A2: re-equilibrate the actual archived 144-function Q4 overlap space."""
import argparse
from time import monotonic
import resource
import numpy as np
from .protocol import BASE, require_frozen, run_path, write_json, sha, storage_guard
from engine.aniso_phase1.research_a.baseline_adapter import Problem, load_basis


def reproduce(run):
    protocol = require_frozen(run)
    folder = run / "baseline"
    folder.mkdir(exist_ok=False)
    start = monotonic()
    problem = Problem.from_archive()
    source = BASE / "v22/multiscale/space/q4"
    W = load_basis(source)
    print("BASELINE loaded", problem.n, W.shape, flush=True)
    field, materials = problem.equilibrate(W)
    old = __import__("json").loads((source / "round6.json").read_text())["materials"]["F45"]
    differences = {k: abs(materials["F45"][k]-old[k])/max(abs(old[k]), 1e-30) for k in ("energy_J", "reaction_N")}
    with np.load(source / "round6.npz", allow_pickle=False) as data:
        differences["displacement_relative"] = float(np.linalg.norm(field["u"]-data["u"])/np.linalg.norm(data["u"]))
        differences["displacement_max_absolute_m"] = float(np.max(np.abs(field["u"]-data["u"])))
    invariants = problem.invariant_errors(W)
    passed = (max(differences[k] for k in ("energy_J", "reaction_N", "displacement_relative")) < protocol["reproduction_tolerances"]["relative"]
              and invariants["quadratic_polynomial_error"] < 1e-8 and invariants["local_fixed_grip_value"] < 1e-10)
    record = dict(completed=True, passed=passed, reproduction="same archived basis, newly assembled and equilibrated",
        source_sha256=sha(source / "round6.npz"), scalar_dofs=144, carrier_free_vector_dofs=225,
        added_vector_dofs=432, differences=differences, invariants=invariants, materials=materials,
        seconds=monotonic()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        not_new_basis_generation=True, spatial_accuracy_certified=False)
    storage_guard(run)
    write_json(folder / "reproduction.json", record)
    print("BASELINE", record, flush=True)
    if not passed:
        raise RuntimeError("Archived candidate reproduction failed")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    reproduce(run_path(parser.parse_args().run_id))
