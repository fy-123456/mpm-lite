"""Material-point directional and loading/unloading diagnostics."""

from __future__ import annotations

import argparse
import json

import numpy as np

from engine.aniso_phase1 import AnisotropicMaterialParams, energy, pk1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stretch", type=float, default=1.10)
    parser.add_argument("--shear", type=float, default=0.25)
    args = parser.parse_args()
    if args.stretch <= 0.0 or args.shear < 0.0:
        raise SystemExit("stretch must be positive and shear must be non-negative")

    F_uniaxial = np.diag([args.stretch, 1.0, 1.0])
    directional = []
    for degrees in (0.0, 45.0, 90.0):
        angle = np.deg2rad(degrees)
        params = AnisotropicMaterialParams(2.0, 3.0, 40.0, [np.cos(angle), np.sin(angle), 0.0])
        P = pk1(F_uniaxial, params.A0, params)
        directional.append({
            "fiber_angle_deg": degrees,
            "engineering_strain": args.stretch - 1.0,
            "P11": float(P[0, 0]),
            "force_per_unit_area": float(P[0, 0]),
            "energy": float(energy(F_uniaxial, params.A0, params)),
        })

    params = AnisotropicMaterialParams(4.0, 6.0, 18.0, [1.0, 0.0, 0.0])
    F_shear = np.array([[1.0, args.shear, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    P_shear = pk1(F_shear, params.A0, params)
    print(json.dumps({
        "uniaxial": directional,
        "simple_shear": {
            "gamma": args.shear,
            "energy": float(energy(F_shear, params.A0, params)),
            "P12": float(P_shear[0, 1]),
        },
        "unload": {
            "energy": float(energy(np.eye(3), params.A0, params)),
            "P_frobenius": float(np.linalg.norm(pk1(np.eye(3), params.A0, params))),
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
