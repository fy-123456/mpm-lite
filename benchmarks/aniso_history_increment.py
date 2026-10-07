"""Reproducible single-switch acceptance audit; no production solver changes."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigvalsh

from engine.aniso_phase1.consistent_transfer import MaterialQ1, bent_nodes, material_response
from engine.aniso_phase1.template_remap import RemappedQ1, axes_for, weighted_relative
from engine.aniso_phase1.history_increment import HistoryState, HistoryIncrementalQ1
from utils.resource_guard import inspect_storage, prepare_warp_cache


DATA_ROOT = os.environ.get('MPM_LITE_DATA_ROOT', '/mnt/sdd/yin/mpm-lite-data')


def guard():
    prepare_warp_cache('/tmp/mpm-lite-warp-cache', DATA_ROOT)
    return asdict(inspect_storage(DATA_ROOT))


def relative(a, b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(b), 1e-30))


def derivative_audit(model, state, dt):
    rng = np.random.default_rng(329)
    v, predictor, p, q = [rng.normal(size=model.X.shape)*.05 for _ in range(4)]
    for a in (v, predictor, p, q):
        a[model.fixed] = 0
    saved = [a.copy() for a in (state.xp, state.Fp, state.xq, state.Fq)]
    _, r = model.potential(v, state, predictor, dt)
    Jp = model.tangent(v, state, p, dt)
    Jq = model.tangent(v, state, q, dt)
    exact_work = float(np.sum(r*p))
    rows = []
    for eps in (1e-2, 1e-3, 1e-4, 1e-5):
        Ep, rp = model.potential(v+eps*p, state, predictor, dt)
        Em, rm = model.potential(v-eps*p, state, predictor, dt)
        work = (Ep-Em)/(2*eps)
        rows.append(dict(eps=eps, potential_relative=abs(work-exact_work)/max(abs(exact_work), 1e-30),
                         residual_relative=relative((rp-rm)/(2*eps), Jp)))
    symmetry = float(abs(np.sum(q*Jp)-np.sum(p*Jq))/max(np.linalg.norm(q)*np.linalg.norm(Jp), 1e-30))
    for before, after in zip(saved, (state.xp, state.Fp, state.xq, state.Fq)):
        np.testing.assert_array_equal(before, after)
    return dict(differences=rows, tangent_symmetry_relative=symmetry, trials_preserve_history=True)


def common_modes(old, new, state):
    def modes(X):
        t = X[:, 0]-.25
        return np.stack([t[:, None]*[1., .2, 0.], t[:, None]*[0., 1., 0.],
                         (t*(X[:, 2]-.5))[:, None]*[0., 1., 0.],
                         (t*(X[:, 1]-.5)*(X[:, 2]-.5))[:, None]*[1., .3, -.2]])
    v0, v1 = modes(old.X), modes(new.X)
    _, f0 = old.elastic(state)
    _, f1 = new.elastic(state)
    K0, K1 = old.stiffness(state), new.stiffness(state)
    a, b = v0.reshape(4, -1), v1.reshape(4, -1)
    work0, work1 = a @ f0.ravel(), b @ f1.ravel()
    stiffness0, stiffness1 = a @ K0 @ a.T, b @ K1 @ b.T
    return dict(names=['axial_mixed', 'shear', 'bilinear', 'trilinear'],
                displacement_max_absolute=float(max(np.max(abs(old.quadrature.N @ p-new.quadrature.N @ q)) for p, q in zip(v0, v1))),
                gradient_max_absolute=float(max(np.max(abs(old.gradient(old.quadrature, p)-new.gradient(new.quadrature, q))) for p, q in zip(v0, v1))),
                virtual_work_before=work0.tolist(), virtual_work_after=work1.tolist(),
                virtual_work_relative=relative(work1, work0), stiffness_relative=relative(stiffness1, stiffness0),
                stiffness_before=stiffness0.tolist(), stiffness_after=stiffness1.tolist())


def spectrum(model, state):
    K = model.stiffness(state)
    free = np.flatnonzero(np.repeat(model.free, 3))
    matrix = K[free][:, free].toarray()
    symmetry = float(np.linalg.norm(matrix-matrix.T)/np.linalg.norm(matrix))
    eigen = eigvalsh(matrix)
    threshold = 1e-9*max(abs(eigen).max(), 1.)
    return dict(lowest_eigenvalues=eigen[:8].tolist(), max_eigenvalue=float(eigen[-1]),
                negative_modes=int(np.sum(eigen < -threshold)), near_zero_modes=int(np.sum(abs(eigen) <= threshold)),
                threshold=float(threshold), symmetry_relative=symmetry)


def rotation_audit(old, new, state):
    angle = np.pi/4
    Q = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    psi, P = material_response(state.Fq, state.quadrature.A, old.params)
    rotated_psi, rotated_P = material_response(Q @ state.Fq, state.quadrature.A, old.params)
    out = dict(angle=angle, clamp='removed for this kinematic diagnostic',
               objectivity_energy_relative=relative(rotated_psi, psi), stress_covariance_relative=relative(rotated_P, Q @ P))
    target_p = (state.xp-.5) @ (Q-np.eye(3)).T
    target_q = (state.xq-.5) @ (Q-np.eye(3)).T
    target_dF = (Q-np.eye(3)) @ state.Fq
    for name, model in [('old', old), ('shifted', new)]:
        fit = model.project(target_p)
        dF = model.gradient(model.quadrature, fit)
        U = model.elastic(state, fit)[0]
        out[name] = dict(displacement_relative=weighted_relative(model.particles.N @ fit-target_p, target_p, state.mass),
                         material_displacement_relative=weighted_relative(model.quadrature.N @ fit-target_q, target_q, state.quadrature.weight),
                         gradient_relative=weighted_relative(dF-target_dF, target_dF, state.quadrature.weight),
                         fitted_rotation_energy_relative=(U-float(state.quadrature.weight @ psi))/float(state.quadrature.weight @ psi))
    # Translations are represented on both templates regardless of old shape.
    translation = np.tile([.03, -.02, .01], (len(new.X), 1))
    out['translation_energy_delta'] = new.elastic(state, translation)[0]-new.elastic(state)[0]
    out['rigid_rotation_increment_supported'] = out['shifted']['gradient_relative'] < 1e-10
    return out


def history_error(state):
    out = {}
    for tag, sites, x, F in [('particle', state.particles, state.xp, state.Fp),
                             ('material', state.quadrature, state.xq, state.Fq)]:
        xf, Ff = state.field.evaluate(sites.X)
        out[tag+'_position_absolute'] = float(np.max(abs(xf-x)))
        out[tag+'_F_absolute'] = float(np.max(abs(Ff-F)))
    return out


def release(source, initial, axes, steps, dt, switch_step):
    paths, final = {}, {}
    for name in ('control', 'identity', 'shifted'):
        model = HistoryIncrementalQ1(source, initial)
        state = initial
        vp = np.zeros_like(state.xp)
        rows = []
        start = time.perf_counter()
        for step in range(steps):
            guard()
            if step == switch_step and name != 'control':
                before = state
                model = HistoryIncrementalQ1(source, state, axes if name == 'shifted' else None)
                assert state is before  # switching changes no physical state
            state, vp, info = model.step(state, vp, dt)
            rows.append(dict(step=step+1, time=(step+1)*dt, **info))
        paths[name] = dict(rows=rows, elapsed_seconds=time.perf_counter()-start, history_error=history_error(state))
        final[name] = (state, vp)
    control, cv = final['control']
    for name in ('identity', 'shifted'):
        state, v = final[name]
        paths[name]['final_comparison'] = dict(position_rms=float(np.sqrt(np.average(np.sum((state.xp-control.xp)**2, axis=1), weights=state.mass))),
                                               F_relative=relative(state.Fp, control.Fp),
                                               F_minus_identity_relative=relative(state.Fp-np.eye(3), control.Fp-np.eye(3)),
                                               velocity_relative=relative(v, cv),
                                               mechanical_relative=(paths[name]['rows'][-1]['mechanical']-paths['control']['rows'][-1]['mechanical'])/paths['control']['rows'][-1]['mechanical'])
    return paths


def run(grid, ppc, order, steps, dt, switch_step):
    storage = guard()
    source = MaterialQ1(grid=grid, ppc_axis=ppc, order=order)
    x = bent_nodes(source)
    state = HistoryState.from_nodal(source, x)
    axes = axes_for(source, 'identity')
    # A single interior x knot moves once, with outer boundary untouched.
    # For tensor-product Q1 this moves the corresponding cross-section nodes.
    knot = len(axes[0])//2
    axes[0][knot] += .2*source.h
    old = HistoryIncrementalQ1(source, state)
    new = HistoryIncrementalQ1(source, state, axes)
    try:
        RemappedQ1(source, axes).recover(state.xp, state.Fp)
        strict = dict(accepted=True)
    except ValueError as exc:
        strict = dict(accepted=False, reason=str(exc))
    snapshot = dict(energy_before=old.elastic(state)[0], energy_after=new.elastic(state)[0])
    # Re-evaluate retained fields; compare all quantities instead of inferring
    # conservation only from equal integrated energy.
    for tag, sites, position, F in [('particle', state.particles, state.xp, state.Fp),
                                    ('material', state.quadrature, state.xq, state.Fq)]:
        xr, Fr = state.field.evaluate(sites.X)
        _, P0 = material_response(F, sites.A, source.params)
        _, P1 = material_response(Fr, sites.A, source.params)
        stress0 = P0 @ F.transpose(0, 2, 1)/np.linalg.det(F)[:, None, None]
        stress1 = P1 @ Fr.transpose(0, 2, 1)/np.linalg.det(Fr)[:, None, None]
        C0 = F @ sites.A @ F.transpose(0, 2, 1)
        C1 = Fr @ sites.A @ Fr.transpose(0, 2, 1)
        C0 /= np.trace(C0, axis1=1, axis2=2)[:, None, None]
        C1 /= np.trace(C1, axis1=1, axis2=2)[:, None, None]
        snapshot[tag] = dict(position_max_absolute=float(np.max(abs(xr-position))),
                             F_max_absolute=float(np.max(abs(Fr-F))), stress_relative=relative(stress1, stress0),
                             current_direction_relative=relative(C1, C0))
    print(f'grid {grid}: snapshot and derivatives', flush=True)
    result = dict(grid=grid, ppc_axis=ppc, quadrature_order=order, dt=dt, steps=steps, switch_step=switch_step,
                  nodes=len(new.X), particles=len(state.xp), material_samples=len(state.xq),
                  shift_knot=knot, shift_distance=.2*source.h, storage_start=storage,
                  strict_control=strict, snapshot=snapshot, derivatives=derivative_audit(new, state, dt),
                  common_modes=common_modes(old, new, state), rotation=rotation_audit(old, new, state))
    print(f'grid {grid}: mass-free spectra', flush=True)
    rest = HistoryState.from_nodal(source, source.X)
    result['spectra'] = dict(bent_old=spectrum(old, state), bent_shifted=spectrum(new, state),
                            rest_old=spectrum(HistoryIncrementalQ1(source, rest), rest),
                            rest_shifted=spectrum(HistoryIncrementalQ1(source, rest, axes), rest))
    result['min_mass_eigenvalue'] = float(new.mass_eigenvalues[0])
    print(f'grid {grid}: release paths', flush=True)
    result['release'] = release(source, state, axes, steps, dt, switch_step)
    # Explicit numerical gates; rotation expressibility has its own failed gate.
    result['gates'] = dict(
        strict_control_rejects=not strict['accepted'],
        snapshot=all(max(snapshot[tag].values()) < 1e-12 for tag in ('particle', 'material')),
        derivatives=min(r['potential_relative'] for r in result['derivatives']['differences']) < 1e-6
                    and min(r['residual_relative'] for r in result['derivatives']['differences']) < 1e-6
                    and result['derivatives']['tangent_symmetry_relative'] < 1e-12,
        common_modes=max(result['common_modes'][k] for k in ('virtual_work_relative', 'stiffness_relative')) < 1e-10,
        mass_free_rank=all(r['negative_modes'] == r['near_zero_modes'] == 0 for r in result['spectra'].values()),
        identity_release=max(result['release']['identity']['final_comparison'][k] for k in ('F_relative', 'velocity_relative')) < 1e-7,
        release=all(all(r['min_det'] > 0 and r['scaled_residual_inf'] < 1e-7 and r['clamp_speed'] == 0
                        and abs(r['energy_budget_residual']) < 1e-12 for r in path['rows'])
                    and all(b['mechanical'] <= a['mechanical']+1e-10 for a, b in zip(path['rows'], path['rows'][1:]))
                    and max(path['history_error'].values()) < 1e-11 for path in result['release'].values()),
        exact_rigid_rotation_space=result['rotation']['rigid_rotation_increment_supported'])
    result['storage_end'] = guard()
    return result


def plot_results(output, results):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpm-lite-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for result in results:
        grid = result['grid']
        for name, style in [('control', '-'), ('shifted', '--')]:
            rows = result['release'][name]['rows']
            axes[0].plot([0]+[1000*r['time'] for r in rows],
                         [1]+[r['mechanical']/result['snapshot']['energy_before'] for r in rows],
                         style, label=f'g{grid} {name}')
        for state, style in [('rest', '-'), ('bent', '--')]:
            spectrum0 = np.array(result['spectra'][state+'_old']['lowest_eigenvalues'])
            spectrum1 = np.array(result['spectra'][state+'_shifted']['lowest_eigenvalues'])
            axes[1].plot(np.arange(1, 9), spectrum1/spectrum0, style, label=f'g{grid} {state}')
    axes[0].set(title='Release after one switch', xlabel='Time (ms)', ylabel='Mechanical energy / initial')
    axes[1].set(title='Clamped elastic spectrum (no mass)', xlabel='Eigenvalue index', ylabel='Shifted / old eigenvalue')
    axes[1].axhline(1., color='gray', linewidth=.7)
    x = np.arange(len(results))
    for offset, key, label in [(-.18, 'gradient_relative', 'Rotation gradient error'),
                               (.18, 'fitted_rotation_energy_relative', 'Rotation energy error')]:
        axes[2].bar(x+offset, [100*r['rotation']['shifted'][key] for r in results], width=.36, label=label)
    axes[2].set_xticks(x, [f"grid {r['grid']}" for r in results])
    axes[2].set(title='45-degree rotation: unresolved increment', ylabel='Percent', yscale='log')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'summary.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grids', type=int, nargs='+', default=[17, 33])
    parser.add_argument('--ppc', type=int, default=3)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--dt', type=float, default=.0005)
    parser.add_argument('--switch-step', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('docs/results/history-increment/v1'))
    args = parser.parse_args()
    if not 0 <= args.switch_step < args.steps:
        parser.error('switch-step must precede the final step')
    guard()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for grid in args.grids:
        results.append(run(grid, args.ppc, args.order, args.steps, args.dt, args.switch_step))
        (args.output/'results.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(grid=grid, gates=results[-1]['gates'])), flush=True)
    plot_results(args.output, results)
    if any(not passed for r in results for gate, passed in r['gates'].items() if gate != 'exact_rigid_rotation_space'):
        raise SystemExit('numerical acceptance gate failed; see results.json')


if __name__ == '__main__':
    main()
