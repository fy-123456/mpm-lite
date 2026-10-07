"""Residual-mode acceptance, fixed-time dt study, and independent quadrature study."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigvalsh
from scipy.spatial.transform import Rotation

from engine.aniso_phase1.consistent_transfer import MaterialQ1, bent_nodes, material_response
from engine.aniso_phase1.history_increment import HistoryState
from engine.aniso_phase1.residual_enrichment import ResidualEnrichedQ1
from engine.aniso_phase1.template_remap import axes_for, common_rule, weighted_relative
from .aniso_history_increment import guard, history_error, relative


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def setup(grid, rule='old3'):
    source = MaterialQ1(grid, ppc_axis=3, order=int(rule[-1]))
    axes = axes_for(source, 'identity')
    axes[0][len(axes[0])//2] += .2*source.h
    if rule.startswith('union'):
        X, weight = common_rule(source, SimpleNamespace(axes=axes), order=int(rule[-1]))
        source.quadrature = source.sample(X, weight)
    state = HistoryState.from_nodal(source, bent_nodes(source))
    return source, state, axes


def derivatives(m, s, dt=.0005):
    rng = np.random.default_rng(194)
    v, p, q, pred = [m.R @ (rng.normal(size=(m.R.shape[1], 3))*.01) for _ in range(4)]
    _, r = m.potential(v, s, pred, dt)
    Jp, Jq = m.tangent(v, s, p, dt), m.tangent(v, s, q, dt)
    work = float(np.sum(p*r))
    records = []
    for eps in (1e-2, 1e-3, 1e-4, 1e-5):
        ep, rp = m.potential(v+eps*p, s, pred, dt)
        em, rm = m.potential(v-eps*p, s, pred, dt)
        records.append(dict(eps=eps, potential_relative=abs((ep-em)/(2*eps)-work)/max(abs(work), 1e-30),
                            tangent_relative=relative((rp-rm)/(2*eps), Jp)))
    symmetry = float(abs(np.sum(q*Jp)-np.sum(p*Jq))/(np.linalg.norm(q)*np.linalg.norm(Jp)))
    return dict(differences=records, symmetry_relative=symmetry)


def rotation(source, state, axes, enrich):
    m = ResidualEnrichedQ1(source, state, axes, enrich=enrich, clamped=False)
    U = m.elastic(state)[0]
    psi, P = material_response(state.Fq, state.quadrature.A, m.params)
    records = []
    for axis, angle in [([0., 0., 1.], 45.), ([1., 2., 3.], 90.)]:
        axis = np.array(axis)/np.linalg.norm(axis)
        Q = Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix()
        target = (state.xp-.5) @ (Q-np.eye(3)).T
        fit = m.project(target)
        target_grad = (Q-np.eye(3)) @ state.Fq
        fitted_F = state.Fq+m.gradient(m.quadrature, fit)
        psi1, P1 = material_response(fitted_F, state.quadrature.A, m.params)
        records.append(dict(angle=angle, axis=axis.tolist(),
                            displacement_relative=weighted_relative(m.particles.N @ fit-target, target, state.mass),
                            gradient_relative=weighted_relative(fitted_F-Q @ state.Fq, target_grad, state.quadrature.weight),
                            energy_relative=float((state.quadrature.weight @ psi1-U)/U),
                            stress_covariance_relative=relative(P1, Q @ P)))
    omega = np.array([[0, -1., 0], [1., 0, 0], [0, 0, 0]])
    infinitesimal = m.project((state.xp-.5) @ omega.T)
    _, force = m.elastic(state)
    return dict(rank=m.rank_info, finite_rotations=records,
                infinitesimal_rotation_virtual_work=float(np.sum(force*infinitesimal)))


def spectrum(m, state):
    R = sp.kron(m.R, np.eye(3), format='csr')
    K = (R.T @ m.stiffness(state) @ R).toarray()
    e = eigvalsh(K)
    threshold = max(np.max(abs(e)), 1.)*1e-9
    # M is only a metric here: K z=lambda M z, never K+M.
    generalized = eigvalsh(K, sp.kron(m.Mr, np.eye(3)).toarray())
    return dict(negative_modes=int(np.sum(e < -threshold)), zero_modes=int(np.sum(abs(e) <= threshold)),
                threshold=float(threshold), lowest_raw=e[:8].tolist(), largest_raw=float(e[-1]),
                lowest_mass_normalized=generalized[:8].tolist(),
                symmetry_relative=float(np.linalg.norm(K-K.T)/np.linalg.norm(K)))


def common_work(source, state, axes):
    old = ResidualEnrichedQ1(source, state, enrich=False)
    new = ResidualEnrichedQ1(source, state, axes)
    modes = []
    for m in (old, new):
        X = m.X
        t = X[:, 0]-.25
        a = np.stack([t[:, None]*[1., .2, 0.], t[:, None]*[0., 1., 0.],
                      (t*(X[:, 2]-.5))[:, None]*[0., 1., 0.]])
        a = np.pad(a, ((0, 0), (0, m.rank), (0, 0)))
        modes.append(a)
    work, stiffness = [], []
    for m, a in zip((old, new), modes):
        work.append(a.reshape(3, -1) @ m.elastic(state)[1].ravel())
        stiffness.append(a.reshape(3, -1) @ m.stiffness(state) @ a.reshape(3, -1).T)
    return dict(virtual_work_relative=relative(work[1], work[0]), stiffness_relative=relative(stiffness[1], stiffness[0]))


def advanced_state_audit(grid):
    """Audit the full nine-mode space encountered at the actual switch time."""
    source, state, axes = setup(grid)
    vp = np.zeros_like(state.xp)
    for _ in range(2):
        guard()
        state, vp, _ = ResidualEnrichedQ1(source, state, enrich=False).step(state, vp, .0005)
    m = ResidualEnrichedQ1(source, state, axes)
    return dict(rank=m.rank_info, derivatives=derivatives(m, state), spectrum=spectrum(m, state),
                rotation=rotation(source, state, axes, True))


def comparison(state, velocity, control, control_velocity):
    return dict(position_rms=float(np.sqrt(np.average(np.sum((state.xp-control.xp)**2, axis=1), weights=state.mass))),
                F_relative=relative(state.Fp, control.Fp),
                F_minus_identity_relative=relative(state.Fp-np.eye(3), control.Fp-np.eye(3)),
                velocity_relative=relative(velocity, control_velocity))


def release(grid, rule, dt, identity=False):
    source, initial, axes = setup(grid, rule)
    nsteps = round(.004/dt)
    switch_step = round(.001/dt)
    if abs(nsteps*dt-.004) > 1e-12 or abs(switch_step*dt-.001) > 1e-12:
        raise ValueError('dt must exactly divide duration and switch time')
    names = ['control', 'shifted', 'enriched']+(['identity'] if identity else [])
    paths, finals = {}, {}
    for name in names:
        state, vp = initial, np.zeros_like(initial.xp)
        rows = []
        start = time.perf_counter()
        for step in range(nsteps):
            guard()
            m = ResidualEnrichedQ1(source, state, axes if step >= switch_step and name in ('shifted', 'enriched') else None,
                                   enrich=name in ('enriched', 'identity'))
            state, vp, info = m.step(state, vp, dt)
            rows.append(dict(step=step+1, time=(step+1)*dt, **info))
        paths[name] = dict(rows=rows, history_error=history_error(state), elapsed_seconds=time.perf_counter()-start)
        finals[name] = (state, vp)
        print(f'grid {grid} {rule} dt={dt:g} {name}: completed {nsteps} steps', flush=True)
    control, cv = finals['control']
    for name in names[1:]:
        paths[name]['comparison'] = comparison(*finals[name], control, cv)
        paths[name]['comparison']['mechanical_relative'] = paths[name]['rows'][-1]['mechanical']/paths['control']['rows'][-1]['mechanical']-1
    return dict(grid=grid, rule=rule, dt=dt, switch_time=.001, duration=.004,
                initial_energy=ResidualEnrichedQ1(source, initial).elastic(initial)[0], paths=paths), finals


def integration_audit(grid):
    data, arrays = {}, {}
    for rule in ('old3', 'old5', 'union3', 'union5', 'union7'):
        guard()
        source, state, axes = setup(grid, rule)
        m = ResidualEnrichedQ1(source, state, axes)
        rng = np.random.default_rng(121)
        du = m.R @ (rng.normal(size=(m.R.shape[1], 3))*1e-5)
        energy, force = m.elastic(state, du)
        R = sp.kron(m.R, np.eye(3))
        K = (R.T @ m.stiffness(state) @ R).toarray()
        arrays[rule] = energy, m.R.T @ force, K
        data[rule] = dict(samples=len(state.Fq), scalar_modes=m.rank, trial_energy=energy)
    ref = arrays['union7']
    for rule, values in arrays.items():
        data[rule].update(energy_relative=relative(values[0], ref[0]), force_relative=relative(values[1], ref[1]),
                          stiffness_relative=relative(values[2], ref[2]))
    return data


def release_gate(record):
    return all(
        all(row['min_det'] > 0 and row['scaled_residual_inf'] < 1e-9 and row['clamp_speed'] < 1e-9
            and abs(row['energy_budget_residual']) < 1e-12 for row in p['rows'])
        and all(b['mechanical'] <= a['mechanical']+1e-10 for a, b in zip(p['rows'], p['rows'][1:]))
        and max(p['history_error'].values()) < 1e-9 for p in record['paths'].values())


def run(grid, out):
    result = dict(grid=grid, storage_start=guard())
    source, state, axes = setup(grid)
    m = ResidualEnrichedQ1(source, state, axes)
    result['rank'] = m.rank_info
    result['snapshot_energy_delta'] = m.elastic(state)[0]-ResidualEnrichedQ1(source, state, enrich=False).elastic(state)[0]
    result['derivatives'] = derivatives(m, state)
    result['common_work'] = common_work(source, state, axes)
    result['rotation'] = {name: rotation(source, state, axes, enrich) for name, enrich in [('shifted', False), ('enriched', True)]}
    print(f'grid {grid}: rotation and derivatives completed', flush=True)
    result['spectra'] = {}
    for state_name, s in [('bent', state), ('rest', HistoryState.from_nodal(source, source.X))]:
        for name in ('control', 'shifted', 'enriched'):
            result['spectra'][state_name+'_'+name] = spectrum(ResidualEnrichedQ1(source, s, axes if name != 'control' else None,
                                                                              enrich=name == 'enriched'), s)
    result['advanced_state'] = advanced_state_audit(grid)
    print(f'grid {grid}: spectra completed', flush=True)
    save(out/f'g{grid}.json', result)
    base, base_final = release(grid, 'old3', .0005, identity=True)
    result['primary_release'] = base
    save(out/f'g{grid}.json', result)
    result['integration'] = integration_audit(grid)
    print(f'grid {grid}: static integration comparison completed', flush=True)
    save(out/f'g{grid}.json', result)
    # Fixed physical horizon/switch time. Only dt changes in this family.
    dt_records = [base]
    dt_finals = [base_final]
    for dt in (.00025, .000125):
        record, final = release(grid, 'old3', dt)
        dt_records.append(record); dt_finals.append(final)
    for record, final in zip(dt_records, dt_finals):
        record['dt_comparison_to_smallest'] = {name: comparison(*final[name], *dt_finals[-1][name])
                                                for name in ('control', 'shifted', 'enriched')}
    result['dt_sensitivity'] = dt_records
    save(out/f'g{grid}.json', result)
    # Rules are chosen BEFORE each independent run and fixed for its duration.
    # Never change sites inside an ongoing history trajectory.
    quad_records, quad_finals = [base], [base_final]
    for rule in ('union3', 'union5'):
        record, final = release(grid, rule, .0005)
        quad_records.append(record); quad_finals.append(final)
    for record, final in zip(quad_records, quad_finals):
        record['quadrature_comparison_to_union5'] = {name: comparison(*final[name], *quad_finals[-1][name])
                                                     for name in ('control', 'shifted', 'enriched')}
    result['quadrature_sensitivity'] = quad_records
    d = result['derivatives']
    result['gates'] = dict(snapshot=abs(result['snapshot_energy_delta']) < 1e-15,
                          derivatives=min(v['potential_relative'] for v in d['differences']) < 1e-6
                                      and min(v['tangent_relative'] for v in d['differences']) < 1e-6
                                      and d['symmetry_relative'] < 1e-12,
                          common_work=max(result['common_work'].values()) < 1e-10,
                          finite_rotations=all(abs(r['energy_relative']) < 1e-8 and r['gradient_relative'] < 1e-9
                                               for r in result['rotation']['enriched']['finite_rotations']),
                          static_rank=all(s['negative_modes'] == s['zero_modes'] == 0 for s in result['spectra'].values()),
                          identity_control=max(base['paths']['identity']['comparison'][k] for k in ('position_rms', 'F_relative', 'velocity_relative')) < 1e-7,
                          releases=all(release_gate(r) for r in dt_records+quad_records),
                          union_integration=result['integration']['union5']['stiffness_relative'] < 1e-7,
                          advanced_state=advanced_gate(result['advanced_state']))
    result['storage_end'] = guard()
    save(out/f'g{grid}.json', result)
    return result


def advanced_gate(record):
    d = record['derivatives']
    return (min(v['potential_relative'] for v in d['differences']) < 1e-6
            and min(v['tangent_relative'] for v in d['differences']) < 1e-6
            and d['symmetry_relative'] < 1e-12
            and record['spectrum']['negative_modes'] == record['spectrum']['zero_modes'] == 0
            and all(abs(r['energy_relative']) < 1e-8 and r['gradient_relative'] < 1e-9
                    for r in record['rotation']['finite_rotations']))


def plot(out, results):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpm-lite-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r in results:
        grid = r['grid']
        for name, style in [('shifted', '--'), ('enriched', '-')]:
            paths = r['primary_release']['paths']
            ax[0].plot([0]+[1000*v['time'] for v in paths[name]['rows']],
                       [1]+[v['mechanical']/r['primary_release']['initial_energy'] for v in paths[name]['rows']],
                       style, label=f'g{grid} {name}')
            ax[1].plot([v['dt']*1000 for v in r['dt_sensitivity']],
                       [v['paths'][name]['comparison']['velocity_relative']*100 for v in r['dt_sensitivity']],
                       style+'o', label=f'g{grid} {name}')
        ax[2].plot(['old3', 'old5', 'union3', 'union5'],
                   [r['integration'][rule]['stiffness_relative'] for rule in ('old3', 'old5', 'union3', 'union5')],
                   'o-', label=f'g{grid}')
    ax[0].set(title='Single-switch release', xlabel='Time (ms)', ylabel='Mechanical energy / initial')
    ax[1].set(title='Fixed physical horizon: 4 ms', xlabel='dt (ms)', ylabel='Velocity difference from no-switch (%)')
    ax[2].set(title='Elastic stiffness quadrature sensitivity', ylabel='Relative difference from union7', yscale='log')
    for a in ax:
        a.grid(alpha=.2); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'summary.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grids', nargs='+', type=int, default=[17, 33])
    parser.add_argument('--output', type=Path, default=Path('docs/results/residual-enrichment/v1'))
    args = parser.parse_args()
    guard(); args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for grid in args.grids:
        results.append(run(grid, args.output))
        save(args.output/'results.json', results)
        print(json.dumps(dict(grid=grid, gates=results[-1]['gates'])), flush=True)
    plot(args.output, results)
    if any(not passed for r in results for passed in r['gates'].values()):
        raise SystemExit('acceptance gate failed; see results.json')


if __name__ == '__main__':
    main()
