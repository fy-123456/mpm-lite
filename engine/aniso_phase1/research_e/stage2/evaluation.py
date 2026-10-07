"""Same-location stress evaluation; no changes to the frozen assembler.

2D outputs are full 3x3 plane-strain tensors (including sigma_zz). P0
pressure is reconstructed as constant inside each cell. All weights are dV.
"""
import numpy as np
from ..flow import quadrature
from ..poro import q2_basis
from ..material import linear_stress


def stress_from_gradient(gradient, pressure, material, alpha=1.):
    gradient = np.asarray(gradient, dtype=float)
    if gradient.shape[-2:] == (2, 2):
        embedded = np.zeros(gradient.shape[:-2] + (3, 3))
        embedded[..., :2, :2] = gradient
    else:
        embedded = gradient
    effective = linear_stress(embedded, material)
    total = effective - alpha*np.asarray(pressure)[..., None, None]*np.eye(embedded.shape[-1])
    return effective, total


class StressEvaluator:
    def __init__(self, solid, order=4):
        self.solid = solid
        g = solid.grid
        rule = list(quadrature(g.dim, order))
        self.local_points = np.array([r[0] for r in rule])
        self.weights = np.array([r[1] for r in rule])
        self.gradients = np.array([q2_basis(s, g.h)[1] for s in self.local_points])
        self.points = (g.indices[:, None, :]+self.local_points)*g.h
        self.dV = np.broadcast_to(g.volume*self.weights, (g.nc, len(rule))).copy()
        self.center_gradient = q2_basis(np.full(g.dim, .5), g.h)[1]

    def evaluate(self, u, p):
        s = self.solid
        u, p = np.asarray(u), np.asarray(p)
        if u.shape != (s.ndof,) or p.shape != (s.grid.nc,) or not np.isfinite(np.r_[u, p]).all():
            raise ValueError('finite displacement and P0 pressure in declared E space required')
        local = u[s.cell_dofs].reshape(s.grid.nc, -1, s.grid.dim)
        grad = np.einsum('cni,qnj->cqij', local, self.gradients)
        center = np.einsum('cni,nj->cij', local, self.center_gradient)
        eff, total = stress_from_gradient(grad, p[:, None], s.material, s.alpha)
        ce, ct = stress_from_gradient(center, p, s.material, s.alpha)
        return dict(gradient=grad, effective_full_field=eff, total_full_field=total,
                    effective_cell_mean=np.einsum('q,cqij->cij', self.weights, eff),
                    total_cell_mean=np.einsum('q,cqij->cij', self.weights, total),
                    effective_center=ce, total_center=ct)

    def reference(self, gradient, pressure):
        flat = self.points.reshape(-1, self.solid.grid.dim)
        grad = np.array([gradient(x) for x in flat]).reshape(self.points.shape[:-1]+(self.solid.grid.dim,)*2)
        p = np.array([pressure(x) for x in flat]).reshape(self.points.shape[:-1])
        e, t = stress_from_gradient(grad, p, self.solid.material, self.solid.alpha)
        return dict(effective_full_field=e, total_full_field=t,
                    effective_cell_mean=np.einsum('q,cqij->cij', self.weights, e),
                    total_cell_mean=np.einsum('q,cqij->cij', self.weights, t))


def partitions(grid):
    edge = np.any((grid.centers < .26) | (grid.centers > .74), axis=1)
    return {'global': np.ones(grid.nc, bool), 'boundary': edge, 'interior': ~edge}


def weighted_error(value, reference, weights, absolute_scale=.1, near_zero=1e-8):
    value, reference, weights = np.asarray(value), np.asarray(reference), np.asarray(weights)
    if value.shape != reference.shape or value.shape[:weights.ndim] != weights.shape or weights.size == 0:
        raise ValueError('same locations, nonempty regions and matching weights required')
    if not np.isfinite(value).all() or not np.isfinite(reference).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('finite fields and positive weights required')
    axes = tuple(range(weights.ndim, value.ndim))
    err2, ref2 = (value-reference)**2, reference**2
    if axes:
        err2, ref2 = err2.sum(axis=axes), ref2.sum(axis=axes)
    volume = float(weights.sum())
    rms = float(np.sqrt(np.sum(weights*err2)/volume))
    ref_rms = float(np.sqrt(np.sum(weights*ref2)/volume))
    return dict(error=rms/(absolute_scale if ref_rms <= near_zero else ref_rms),
                absolute_rms=rms, reference_rms=ref_rms, volume=volume,
                normalization='fixed_absolute_scale' if ref_rms <= near_zero else 'relative_L2')


def stress_errors(evaluator, numerical, reference):
    result = {}
    for region, mask in partitions(evaluator.solid.grid).items():
        result[region] = {}
        for kind in ('effective', 'total'):
            for metric in ('cell_mean', 'full_field'):
                key = kind+'_'+metric
                weights = evaluator.dV[mask] if metric == 'full_field' else np.full(mask.sum(), evaluator.solid.grid.volume)
                result[region][key] = weighted_error(numerical[key][mask], reference[key][mask], weights)
    return result


def stress_passed(metrics, threshold=.04):
    """Fail closed: every named region, stress type, mean and full field is required."""
    try:
        for region in ('global', 'boundary', 'interior'):
            for kind in ('effective', 'total'):
                for metric in ('cell_mean', 'full_field'):
                    item = metrics[region][kind+'_'+metric]
                    if not (np.isfinite(item['error']) and 0 <= item['error'] <= threshold and item['volume'] > 0):
                        return False
    except (KeyError, TypeError, ValueError):
        return False
    return True


def physical_diagnostics(evaluator, values):
    F = np.eye(evaluator.solid.grid.dim)+values['gradient']
    J = np.linalg.det(F)
    porosity = 1-.65/J
    return dict(min_J=float(J.min()), max_J=float(J.max()),
                porosity_min=float(porosity.min()), porosity_max=float(porosity.max()),
                finite=bool(all(np.isfinite(x).all() for x in values.values())),
                passed=bool(np.isfinite(J).all() and J.min()>0 and porosity.min()>0 and porosity.max()<1))
