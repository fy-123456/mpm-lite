"""Frozen whole-family snapshots; no use of the v22 final spatial reference."""
import numpy as np
from engine.aniso_phase1.research_b.rules import MaterialRule
from engine.aniso_phase1.research_b.operator import LinearMaterialOperator
from engine.aniso_phase1.types import AnisotropicMaterialParams

PARAMS = AnisotropicMaterialParams(10., 20., 200.)
SPACE_ID = 'B-v1-quadratic-displacement-9scalar-point-major-SI'


def gradient_map(X):
    x, y, z = (X-np.array([.5, .5, .5])).T
    D = np.zeros((len(X), 9, 3))
    D[:, :3] = np.eye(3)
    D[:, 3, 0], D[:, 4, 1], D[:, 5, 2] = x, y, z
    D[:, 6, 0], D[:, 6, 1] = y, x
    D[:, 7, 0], D[:, 7, 2] = z, x
    D[:, 8, 1], D[:, 8, 2] = z, y
    return D


def source_rule(angle=45., mixture=False, order=4):
    edges = [np.linspace(.125, .875, 7), np.linspace(.375, .625, 3), np.linspace(.375, .625, 3)]
    z, w = np.polynomial.legendre.leggauss(order)
    points, weights, partitions = [], [], []
    for cell, ijk in enumerate(np.ndindex(6, 2, 2)):
        lo = np.array([edges[k][j] for k, j in enumerate(ijk)])
        hi = np.array([edges[k][j+1] for k, j in enumerate(ijk)])
        X = np.array(np.meshgrid(*[lo[k]+(hi[k]-lo[k])*(z+1)/2 for k in range(3)], indexing='ij')).reshape(3, -1).T
        W = np.prod(np.array(np.meshgrid(w, w, w, indexing='ij')), axis=0).ravel()*np.prod(hi-lo)/8
        points.extend(X)
        weights.extend(W)
        partitions.extend([cell]*len(W))
    X, W, part = np.array(points), np.array(weights), np.array(partitions)
    theta = np.full(len(X), np.deg2rad(angle))
    if mixture:
        # Resolved reference-material interface; never average across it.
        theta[X[:, 1] > .5] += np.pi/2
    a = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(len(X))))
    return MaterialRule.from_directions(X, W, a, part)


def operator(rule):
    return LinearMaterialOperator(rule, gradient_map, PARAMS, SPACE_ID, 9)


def directions():
    rng = np.random.default_rng(220930)
    random = rng.normal(size=(9, 3)); random /= np.linalg.norm(random)
    affine = np.zeros((9, 3)); affine[0, 0] = 1
    bending = np.zeros((9, 3)); bending[3, 1] = 1; bending[6, 0] = -1
    soft = np.zeros((9, 3)); soft[2, 2] = 1
    stress = np.zeros((9, 3)); stress[0, 1] = 1; stress[1, 0] = 1
    return dict(random=random, affine=affine, bending=bending, soft=soft, stress=stress)


def states(split):
    # Entire load and direction families belong to one split. The hidden
    # evaluator supplies its own unseen parameters after candidate freeze.
    families = {'train': [('axial45', 45., False, .03, 0.), ('shear0', 0., False, .015, .05)],
                'development': [('bend30', 30., False, .04, .12), ('mixed60', 60., True, .025, .08)]}
    for family, angle, mixture, amplitude, bend in families[split]:
        for stage, scale in [('load', .5), ('hold', 1.), ('unload', .35), ('end_hold', 0.)]:
            q = np.zeros((9, 3)); q[0, 0] = amplitude*scale
            q[1, 1], q[2, 2] = -.2*amplitude*scale, -.15*amplitude*scale
            q[3, 1], q[6, 0], q[0, 1] = bend*scale, -bend*scale, .2*amplitude*scale
            yield dict(id=f'{family}/{stage}', family=family, angle=angle, mixture=mixture, q=q)


def errors(candidate, reference, scale=1.):
    # Dimensioned absolute floors fixed before any result: U=1e-8 J,
    # coefficient force=1e-5 (SI generalized force), weak moment=1e-6,
    # tangent action=1e-4 for unit coefficient direction. Raw differences kept.
    out = {}
    for k, floor in [('U', 1e-8), ('force', 1e-5), ('weak_moments', 1e-6), ('tangent_action', 1e-4)]:
        if k not in candidate or k not in reference:
            continue
        absolute = float(np.linalg.norm(np.asarray(candidate[k])-reference[k]))
        denominator = max(float(np.linalg.norm(reference[k])), floor)
        out[k] = dict(relative=absolute/denominator, absolute=absolute, denominator=denominator,
                      passed=bool(absolute <= scale*({'tangent_action': .02}.get(k, .01))*denominator))
    return out
