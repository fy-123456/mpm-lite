"""Read-only adaptation of the v22 space and original quadratic potential."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from benchmarks.aniso_v21_common import controlled_case, hessian
from engine.aniso_phase1.tensor_reference import coordinates, interpolate
from engine.aniso_phase1.high_order_space import BoxElastic, scalar_grams
from engine.aniso_phase1.stress_local_space import reduced_solve

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "docs/results/lite-aniso-mainline"


def read_field(path, degree=None):
    with np.load(path, allow_pickle=False) as data:
        return ([data[f"axis{k}"].copy() for k in range(3)],
                int(data["degree"]) if "degree" in data else degree, data["u"].copy())


@dataclass
class Problem:
    edges: list
    degree: int
    A: np.ndarray
    carrier_X: np.ndarray
    Ks: np.ndarray
    params: object
    fiber_tensor: np.ndarray
    initial_u: np.ndarray

    @classmethod
    def from_archive(cls, degree=4):
        if degree not in (2, 3, 4):
            raise ValueError("Supported degrees are Q2/Q3/Q4")
        state, energy, *_ = controlled_case()
        with np.load(BASE / "v11-reference-q3/cases/local2.npz", allow_pickle=False) as data:
            edges = [data[f"axis{k}"].copy() for k in range(3)]
        with np.load(BASE / "v19/space/reconstruction32.npz", allow_pickle=False) as data:
            old = [data[f"axis{k}"].copy() for k in range(3)]
            A = interpolate(old, 2, data["A"], edges, degree)
        with np.load(BASE / "v20/space/F45-r32-e0.npz", allow_pickle=False) as data:
            initial = interpolate(old, 2, data["u"], edges, degree)
        return cls(edges, degree, A, state.Y.copy(), energy.Ks.copy(), energy.params, energy.A[0].copy(), initial)

    @property
    def n(self):
        return self.A.shape[0]

    @property
    def free(self):
        return (self.carrier_X[:, 0] > .25) & (self.carrier_X[:, 0] < .75)

    @property
    def Q(self):
        return np.eye(len(self.carrier_X))[:, self.free]

    @property
    def lift(self):
        out = np.zeros_like(self.carrier_X)
        out[self.carrier_X[:, 0] >= .75, 0] = .005
        return out

    def nodes(self):
        return np.array(np.meshgrid(*coordinates(self.edges, self.degree), indexing="ij")).reshape(3, -1).T

    def invariant_errors(self, W):
        x = self.carrier_X
        poly = lambda z: np.column_stack((np.ones(len(z)), z, z*z, z[:, 0]*z[:, 1], z[:, 0]*z[:, 2], z[:, 1]*z[:, 2]))
        X = self.nodes()
        fixed = (X[:, 0] <= .25) | (X[:, 0] >= .75)
        return dict(quadratic_polynomial_error=float(np.max(np.abs(self.A@poly(x)-poly(X)))),
                    local_fixed_grip_value=float(np.max(np.abs(W[fixed]), initial=0.)),
                    orthogonality_error=float(la.norm(W.T@W-np.eye(W.shape[1]))))

    def equilibrate(self, W, labels=("ISO", "F0", "F45", "F90")):
        Q, lift = self.Q, self.lift
        nf = Q.shape[1]
        G = scalar_grams(self.edges, self.degree, np.column_stack((self.A@Q, W, self.A@lift[:, 0])))
        patch = Q.T@self.Ks@Q
        patchlift = Q.T@self.Ks@lift
        constant = .5*float(np.sum(lift*(self.Ks@lift)))
        gram = W.T@W
        records = {}
        field = None
        for label in labels:
            coef, _, result, _ = reduced_solve(G, hessian(label), patch, patchlift, constant,
                                              nf, list(range(W.shape[1])), gram)
            if not result["static_passed"] or result["negative_modes"] or result["zero_modes"]:
                raise RuntimeError(f"Unstable unconstrained space: {label}: {result}")
            if result["work_identity_relative"] > 1e-6 or result["free_residual"] > 1e-7:
                raise RuntimeError(f"Equilibrium failed: {label}: {result}")
            records[label] = result
            if label == "F45":
                y = lift+Q@coef[:nf]
                alpha = coef[nf:]
                field = dict(y=y, local_coefficients=alpha, u=self.A@y+W@alpha)
        if field is None:
            raise ValueError("F45 must be included")
        op = BoxElastic(self.edges, self.degree, hessian("F45"))
        force = op.apply(field["u"].T.ravel()).reshape(3, -1).T
        energy = .5*float(np.sum(field["u"]*force)+np.sum(field["y"]*(self.Ks@field["y"])))
        reaction = float((self.A.T@force+self.Ks@field["y"])[self.carrier_X[:, 0] >= .75, 0].sum())
        records["F45"].update(independent_energy_J=energy, independent_reaction_N=reaction,
            independent_energy_error_J=abs(energy-records["F45"]["energy_J"]),
            independent_reaction_error_N=abs(reaction-records["F45"]["reaction_N"]))
        if not np.isfinite(field["u"]).all() or abs(energy-records["F45"]["energy_J"]) > 1e-10 or abs(reaction-records["F45"]["reaction_N"]) > 1e-8:
            raise RuntimeError("Independent potential/reaction check failed")
        return field, records


def load_basis(folder):
    folder = Path(folder)
    raw = sp.load_npz(folder / "basis-raw.npz")
    with np.load(folder / "basis-transform.npz", allow_pickle=False) as data:
        transform = data["transform"].copy()
    if raw.shape[1] != transform.shape[0] or not np.isfinite(raw.data).all() or not np.isfinite(transform).all():
        raise ValueError("Invalid basis archive")
    return raw@transform
