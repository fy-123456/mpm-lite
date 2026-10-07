"""Globally polynomial compatible histories for a separate initial-state control.

Polynomial coefficients are history only, never additional motion unknowns.
The old piecewise history remains an independently reproducible experiment.
"""
import itertools

import numpy as np
from numpy.polynomial import polynomial as poly

from .history_increment import HistoryField, frozen


class PolynomialHistoryBasis:
    kind = 'global-polynomial-v1'

    def __init__(self, degrees=(5, 3, 2), origin=(.25, .5, .5), scale=(.5, .0625, .0625)):
        self.degrees = tuple(int(d) for d in degrees)
        self.origin, self.scale = frozen(origin), frozen(scale)
        if min(self.degrees) < 1 or np.any(self.scale <= 0):
            raise ValueError('positive degrees and coordinate scales required')
        self.powers = np.array(list(itertools.product(*[range(d+1) for d in self.degrees])))
        # Distinct, deterministic identity for HistoryField's term aggregation.
        self.X = frozen(-1-self.powers)

    def evaluate(self, X, coefficients):
        s = ((np.asarray(X)-self.origin)/self.scale).T
        c = np.asarray(coefficients).reshape(tuple(d+1 for d in self.degrees)+(3,))
        x = np.column_stack([poly.polyval3d(*s, c[..., i]) for i in range(3)])
        F = np.empty((len(X), 3, 3))
        for d in range(3):
            dc = poly.polyder(c, axis=d)/self.scale[d]
            F[:, :, d] = np.column_stack([poly.polyval3d(*s, dc[..., i]) for i in range(3)])
        return x, F

    def same_basis(self, other):
        return (type(other) is type(self) and self.degrees == other.degrees
                and np.array_equal(self.origin, other.origin)
                and np.array_equal(self.scale, other.scale))


def fit_smooth_history(field, X, weights, degrees=(5, 3, 2)):
    """Volume-L2 fit displacement, constrained to zero on the left clamp.

    g(X)=X+s_x * polynomial(s_x,s_y,s_z). No F is fitted independently.
    """
    basis = PolynomialHistoryBasis(degrees)
    s = (X-basis.origin)/basis.scale
    active = basis.powers[:, 0] > 0
    powers = basis.powers[active]
    design = np.prod(s[:, None, :]**powers[None, :, :], axis=2)
    weight = np.sqrt(weights/np.sum(weights))
    displacement = field.evaluate(X)[0]-X
    fit, _, rank, singular = np.linalg.lstsq(design*weight[:, None], displacement*weight[:, None], rcond=None)
    if rank != len(powers):
        raise ValueError('polynomial fit is rank deficient')
    c = np.zeros((len(basis.powers), 3))
    c[active] = fit
    for i in range(3):
        c[np.all(basis.powers == 0, axis=1), i] += basis.origin[i]
        power = np.zeros(3, dtype=int); power[i] = 1
        c[np.all(basis.powers == power, axis=1), i] += basis.scale[i]
    return HistoryField(((basis, frozen(c)),)), dict(
        degrees=list(degrees), scalar_fit_rank=int(rank),
        design_condition=float(singular[0]/singular[-1]),
        fit='volume-L2 displacement; exact left-face clamp; analytic gradient')
