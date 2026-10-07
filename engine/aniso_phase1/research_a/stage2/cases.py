"""Explicit material, geometry and grip identities shared by all A operators."""
from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np
from engine.aniso_phase1.beam_reference import reference_hessian
from engine.aniso_phase1.types import AnisotropicMaterialParams
from engine.aniso_phase1.research_a.support_family import support_family as parent_supports

BASE_BOX = ((.125, .875), (.375, .625), (.375, .625))


@dataclass(frozen=True)
class CaseSpec:
    name: str = "F45"
    angle: float = 45.
    k_f: float = 200.
    mu: float = 10.
    lam: float = 20.
    box: tuple = BASE_BOX
    grips: tuple = (.25, .75)
    loading: str = "extension"
    displacement: float = .005
    rotation: float = .02
    rotation_center: tuple = (.75, .5, .5)

    def __post_init__(self):
        box = np.asarray(self.box, dtype=float)
        if box.shape != (3, 2) or not np.isfinite(box).all() or np.any(np.diff(box) <= 0):
            raise ValueError("Invalid physical box")
        if tuple(self.grips) != (.25, .75) or tuple(box[0]) != BASE_BOX[0]:
            raise ValueError("This version supports the registered x geometry and grips only")
        if self.loading not in ("extension", "shear", "bending"):
            raise ValueError("Unknown loading; never fall back to F45")
        if not np.isfinite([self.angle, self.displacement, self.rotation]).all():
            raise ValueError("Nonfinite loading or angle")
        if self.mu <= 0 or self.lam < 0 or self.k_f < 0:
            raise ValueError("Nonpositive elastic stability parameters")
        self.params  # Validate finite material parameters.

    @property
    def fiber(self):
        a = np.deg2rad(self.angle)
        return np.array([np.cos(a), np.sin(a), 0.])

    @property
    def params(self):
        return AnisotropicMaterialParams(self.mu, self.lam, self.k_f, self.fiber)

    @property
    def H(self):
        return reference_hessian(self.mu, self.lam, self.k_f, self.fiber)

    @property
    def volume(self):
        return float(np.prod(np.diff(np.asarray(self.box), axis=1)))

    @property
    def signature(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    @property
    def affine_scale(self):
        return np.diff(self.box, axis=1).ravel()/np.diff(BASE_BOX, axis=1).ravel()

    def map_points(self, X):
        return (np.asarray(X)-np.asarray(BASE_BOX)[:, 0])*self.affine_scale+np.asarray(self.box)[:, 0]

    def map_edges(self, edges):
        return [(np.asarray(e)-BASE_BOX[k][0])*self.affine_scale[k]+self.box[k][0]
                for k, e in enumerate(edges)]

    def boundary(self, X):
        X = np.asarray(X)
        u = np.zeros_like(X, dtype=float)
        right = X[:, 0] >= self.grips[1]-1e-12
        if self.loading == "bending":
            c, s = np.cos(self.rotation), np.sin(self.rotation)
            R = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
            u[right] = (X[right]-self.rotation_center)@(R-np.eye(3)).T
        else:
            u[right, int(self.loading == "shear")] = self.displacement
        return u

    def as_dict(self):
        return dict(**asdict(self), case_sha256=self.signature, volume_m3=self.volume,
                    linear_model="tangent at identity of Hencky plus bilateral quadratic fiber",
                    nonlinear_model="Hencky plus bilateral quadratic fiber",
                    stress="PK1", units=dict(length="m", force="N", stress="Pa", energy="J"))


def registered_cases():
    cases = [CaseSpec()]
    cases += [CaseSpec(name=f"angle{a}", angle=float(a)) for a in (15, 30, 60, 75)]
    cases += [CaseSpec(name="soft-fiber", k_f=50.), CaseSpec(name="stiff-fiber", k_f=500.),
              CaseSpec(name="tall-shear", box=((.125, .875), (.3125, .6875), (.375, .625)), loading="shear"),
              CaseSpec(name="bending", loading="bending")]
    return {c.name: c for c in cases}


def supports(case, family):
    """Axis-aligned physical supports, including a direction-dependent envelope."""
    result = parent_supports("v22-overlap" if family == "fiber-rect" else family)
    for i, item in enumerate(result):
        for key in ("center", "lo", "hi"):
            item[key] = case.map_points(np.asarray(item[key])).tolist()
        if family == "fiber-rect" and i < len(result)-3:
            c = np.asarray(item["center"])
            # This is an axis-aligned envelope, never a rotated FE grid.
            radius = (.0625+.0625*np.abs(case.fiber)/max(abs(case.fiber)))*case.affine_scale
            item["lo"] = np.maximum(c-radius, [.25, case.box[1][0], case.box[2][0]]).tolist()
            item["hi"] = np.minimum(c+radius, [.75, case.box[1][1], case.box[2][1]]).tolist()
        if np.any(np.asarray(item["hi"]) <= item["lo"]):
            raise ValueError("Empty support")
    return result
