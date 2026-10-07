"""Reconstruct an A package without consulting another research namespace."""
import hashlib
import json
from pathlib import Path
import numpy as np
from engine.aniso_phase1.types import AnisotropicMaterialParams
from engine.aniso_phase1.tensor_reference import interpolate
from .baseline_adapter import Problem, load_basis
from .export_adapter import FixedSpace


def load_package(folder):
    folder = Path(folder)
    metadata = json.loads((folder / "space-package.json").read_text())
    if metadata["schema_version"] != 1 or metadata["dtype"] != "float64" or metadata["device"] != "cpu":
        raise ValueError("Unsupported space package schema/dtype/device")
    for name, expected in metadata["files"].items():
        if Path(name).name != name:
            raise ValueError("Package members must be local basenames")
        with (folder / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Package digest mismatch: {name}")
    with np.load(folder / "geometry.npz", allow_pickle=False) as g:
        degree = int(g["degree"])
        edges = [g[f"axis{k}"].copy() for k in range(3)]
        carrier, Ks, fiber = g["carrier_X"].copy(), g["Ks"].copy(), g["fiber_tensor"].copy()
    with np.load(folder / "carrier-basis.npz", allow_pickle=False) as z:
        A = interpolate([z[f"axis{k}"] for k in range(3)], 2, z["A"], edges, degree)
    mat = metadata["material"]
    theta = np.deg2rad(mat["fiber_angle_degrees"])
    params = AnisotropicMaterialParams(mat["mu"], mat["lam"], mat["k_f"], [np.cos(theta), np.sin(theta), 0.])
    problem = Problem(edges, degree, A, carrier, Ks, params, fiber, np.zeros((len(A), 3)))
    result = FixedSpace(problem, load_basis(folder))
    if list(result.shape) != metadata["q_shape"]:
        raise ValueError("Package dimension mismatch")
    return result, metadata
