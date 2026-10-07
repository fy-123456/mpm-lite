"""Replay bundled mappings, energy, force, and tangent after independent load."""
import argparse
import numpy as np
from .protocol import candidate_path, run_path, require_frozen, write_json
from engine.aniso_phase1.research_a.package_loader import load_package


def validate(run, candidate):
    require_frozen(run)
    folder = candidate_path(run, candidate) / "space-package"
    space, metadata = load_package(folder)
    with np.load(folder / "test-vectors.npz", allow_pickle=False) as z:
        points = [z[f"points{k}"] for k in range(3)]
        x, F = space.evaluate(z["q"], points)
        dx, dF = space.jvp(z["direction"], points)
        response = space.response(z["q"], z["direction"])
        relative = lambda a, b: float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12))
        errors = dict(x=float(np.max(np.abs(x-z["x"]))), F=float(np.max(np.abs(F-z["F"]))),
                      dx=float(np.max(np.abs(dx-z["dx"]))), dF=float(np.max(np.abs(dF-z["dF"]))),
                      energy=relative(response["energy_J"], z["energy_J"]),
                      force=relative(response["force"], z["force"]),
                      tangent=relative(response["tangent_action"], z["tangent_action"]))
    passed = max(errors.values()) < 1e-5
    result = dict(completed=True, passed=passed, errors=errors, shape=list(space.shape),
                  schema_version=metadata["schema_version"], tested_without_B_C_D_E=True)
    write_json(folder.parent / "package-validation.json", result)
    if not passed:
        raise RuntimeError("Round-trip package validation failed")
    print("PACKAGE", result, flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--candidate", required=True)
    a = p.parse_args()
    validate(run_path(a.run_id), a.candidate)
