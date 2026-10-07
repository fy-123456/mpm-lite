"""True residual blocks use fixed physical row scales, never solver status alone."""
import numpy as np

SCALES = {'mechanics': .1, 'mass': 1., 'darcy': .1}


def block_residual(matrix, rhs, solution, nu, np_):
    residual = matrix @ solution-rhs
    blocks = {'mechanics': residual[:nu], 'mass': residual[nu:nu+np_], 'darcy': residual[nu+np_:]}
    result = {name: float(np.max(np.abs(value), initial=0.)/SCALES[name]) for name, value in blocks.items()}
    result['maximum'] = max(result.values())
    return result


def step_residual(solver, old, result, dt, load, boundary, source=0.):
    M, rhs, free, _, _, _ = solver._system(old, dt, load, boundary, source)
    x = np.r_[result.state.u[solver.solid.free], result.state.p, result.flux[free]]
    return block_residual(M, rhs, x, len(solver.solid.free), len(old.p))


def step_ledger(solver, old, result, dt, source=0.):
    g = solver.flow.grid
    src = g.average(source) if callable(source) else np.broadcast_to(source, (g.nc,))
    local = solver.solid.G @ (result.state.u-old.u)+solver.C @ (result.state.p-old.p)+dt*solver.flow.B @ result.flux-dt*g.volume*src
    return local


def valid_ledger(metrics, residual, diagnostics, limit=1e-6):
    return bool(all(np.isfinite(list(metrics.values()))) and residual['maximum'] <= 1e-8 and diagnostics['passed']
                and max(metrics['mass_defect'], metrics['local_mass_defect'], abs(metrics['energy_residual']), metrics['momentum_balance']) <= limit
                and metrics['physical_dissipation'] >= -1e-10 and metrics['numerical_loss'] >= -1e-10)
