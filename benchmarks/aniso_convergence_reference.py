"""Reliable-quadrature time convergence and a common-state spatial reference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from engine.aniso_phase1.consistent_transfer import bent_nodes
from engine.aniso_phase1.consistent_transfer import material_response
from engine.aniso_phase1.history_increment import HistoryField, ReferenceBasis, frozen, material_tangent
from engine.aniso_phase1.convergence_reference import (
    FrozenResidualQ1, geometry, knots, union_knots, tensor_rule, sites,
    state_from_field, save_fields, load_fields, compare_fields, rms, directions, sum_fields)
from .aniso_history_increment import guard, history_error


PROTOCOL = 'common-switch-state-fixed-space-v1'


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def fingerprint(fields):
    h = hashlib.sha256()
    for name in sorted(fields):
        for b, c in fields[name].terms:
            if hasattr(b, 'kind'):
                h.update(b.kind.encode()); h.update(b.origin.tobytes()); h.update(b.scale.tobytes())
            h.update(b.X.tobytes()); h.update(c.tobytes())
    return h.hexdigest()


def shared_axes():
    return union_knots(*[knots(geometry(g)) for g in (17, 33, 49, 65)],
                       knots(geometry(17), True), knots(geometry(33), True))


def model_for(grid, case, initial, particles, order=5):
    source = geometry(grid)
    axes = knots(source, case != 'control')
    qaxes = union_knots(knots(geometry(17)), knots(source), axes)
    qsites = sites(*tensor_rule(qaxes, order))
    state = state_from_field(initial['position'], particles, qsites)
    model = FrozenResidualQ1(source, state, axes, enrich=case == 'enriched')
    vp = initial['velocity'].evaluate(particles.X)[0]
    return model, state, vp


def run_path(out, name, grid, case, initial, particles, dt, duration, order=5):
    guard()
    meta_path, field_path = out/f'{name}.json', out/f'{name}.npz'
    config = dict(protocol=PROTOCOL, grid=grid, case=case, dt=dt, duration=duration,
                  order=order, initial_fingerprint=fingerprint(initial), mass_rule='shared-union-gauss2')
    if meta_path.exists() and field_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['config'] == config and all(meta.get('checks', {'missing': False}).values()):
            print(f'{name}: cached', flush=True)
            return meta, load_fields(field_path)
    start = time.perf_counter()
    print(f'{name}: constructing frozen space', flush=True)
    m, state, vp = model_for(grid, case, initial, particles, order)
    setup_seconds = time.perf_counter()-start
    vprojected = m.particles.N @ m.project(vp)
    projection_relative = rms(vprojected-vp, state.mass)/max(rms(vp, state.mass), 1e-30)
    initial_energy = m.elastic(state)[0]+.5*float(np.sum(state.mass[:, None]*vp**2))
    steps = round(duration/dt)
    if abs(steps*dt-duration) > 1e-12:
        raise ValueError('duration must be an integer number of steps')
    rows = []
    for step in range(steps):
        guard()
        state, vp, info = m.step(state, vp, dt)
        rows.append(dict(step=step+1, time=(step+1)*dt, **info))
        if (step+1) % 8 == 0 or step+1 == steps:
            print(f'{name}: {step+1}/{steps}, iterations={info["iterations"]}, residual={info["scaled_residual_inf"]:.2g}', flush=True)
    fields = dict(position=state.field, velocity=m.last_velocity)
    herror = history_error(state)
    checks = dict(
        solve=all(r['scaled_residual_inf'] <= 1.01e-9 for r in rows),
        admissible=all(r['min_det'] > 0 for r in rows),
        clamp=all(r['clamp_speed'] < 1e-9 for r in rows),
        budget=all(abs(r['energy_budget_residual']) < 1e-12 for r in rows),
        history=max(herror.values()) < 1e-8,
        dissipation=rows[0]['mechanical'] <= initial_energy+1e-10 and
                    all(b['mechanical'] <= a['mechanical']+1e-10 for a, b in zip(rows, rows[1:])))
    meta = dict(config=config, rows=rows, rank=m.rank_info, mass_points=len(state.mass),
                material_points=len(state.Fq), initial_mechanical=initial_energy,
                initial_projection_relative=projection_relative, history_error=herror,
                setup_seconds=setup_seconds, elapsed_seconds=time.perf_counter()-start, checks=checks,
                storage_end=guard())
    save_fields(field_path, **fields)
    save(meta_path, meta)
    if not all(checks.values()):
        raise RuntimeError(f'{name}: numerical acceptance failed: {checks}')
    return meta, fields


def prepare_snapshot(out, particles):
    source = geometry(17)
    initial = dict(position=HistoryField(((ReferenceBasis(source), frozen(bent_nodes(source))),)),
                   velocity=HistoryField(((ReferenceBasis(source), frozen(np.zeros_like(source.X))),)))
    return run_path(out, 'initial-switch-state', 17, 'control', initial, particles, .00003125, .001)


def mass_difference(a, b, particles):
    va, _ = a['velocity'].evaluate(particles.X)
    vb, _ = b['velocity'].evaluate(particles.X)
    absolute = rms(va-vb, particles.weight)
    return dict(velocity_rms=absolute, velocity_relative=absolute/max(rms(vb, particles.weight), 1e-30))


def spatial_report(a, b, probes):
    return compare_fields(a['position'], a['velocity'], b['position'], b['velocity'],
                          *probes, geometry(17).params)


def endpoint_quadrature(grid, case, initial, final, particles, cache_path=None):
    """Recheck integration at the evolved endpoint, in the same frozen basis."""
    guard()
    signature = dict(grid=grid, case=case, initial=fingerprint(initial), final=fingerprint(final))
    if cache_path is not None and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached['signature'] == signature:
            return cached['metrics']
    m, _, _ = model_for(grid, case, initial, particles)
    source = geometry(grid)
    axes = union_knots(knots(geometry(17)), knots(source), knots(source, case != 'control'))
    p = m.R @ (np.random.default_rng(981).normal(size=(m.R.shape[1], 3))*.01)
    values = {}
    for order in (5, 7):
        X, weights = tensor_rule(axes, order)
        A = directions(X)
        mapping = m.map_points(X, weights, A)
        _, F = final['position'].evaluate(X)
        psi, P = material_response(F, A, m.params)
        force = sum(D.T @ (weights[:, None]*P[:, :, d]) for d, D in enumerate(mapping.D))
        dP = material_tangent(F, A, m.gradient(mapping, p), m.params)
        Kp = sum(D.T @ (weights[:, None]*dP[:, :, d]) for d, D in enumerate(mapping.D))
        values[order] = (float(weights @ psi), m.R.T @ force, m.R.T @ Kp)
    result = {name: float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))
              for name, a, b in zip(('energy_relative', 'force_relative', 'tangent_action_relative'), values[5], values[7])}
    result['passed_1e-6'] = max(result.values()) < 1e-6
    if cache_path is not None:
        save(cache_path, dict(signature=signature, metrics=result))
    return result


def residual_error_decomposition(grid, initial, enriched, control, reference, particles, probes):
    """Endpoint diagnostic, not a replacement trajectory or a causal ablation.

    Project only the accumulated displacement onto clamped new Q1, preserving
    the common old history. Split squared F error into cross and extra terms.
    """
    m, _, _ = model_for(grid, 'shifted', initial, particles)
    increments = enriched['position'].evaluate(particles.X)[0]-initial['position'].evaluate(particles.X)[0]
    fit = m.project(increments)
    projected = sum_fields(initial['position'], m.coefficient_field(fit))
    X, weights = probes
    Fe = enriched['position'].evaluate(X)[1]
    Fp = projected.evaluate(X)[1]
    G = Fe-Fp
    masks = dict(all=np.ones(len(X), dtype=bool), clamp=X[:, 0] < .3125,
                 switch=(X[:, 0] >= .4375) & (X[:, 0] <= .5625))
    masks['bulk'] = ~(masks['clamp'] | masks['switch'])
    out = {}
    for name, target in [('reference', reference), ('control', control)]:
        Ft = target['position'].evaluate(X)[1]
        D = Fp-Ft
        rows = {}
        for region, mask in masks.items():
            w = weights[mask]/weights[mask].sum()
            cross = float(2*np.sum(w*np.einsum('qij,qij->q', D[mask], G[mask])))
            extra = float(np.sum(w*np.sum(G[mask]**2, axis=(1, 2))))
            before = rms(D[mask], weights[mask])**2
            after = rms(Fe[mask]-Ft[mask], weights[mask])**2
            rows[region] = dict(projected_error_rms=float(np.sqrt(before)), enriched_error_rms=float(np.sqrt(after)),
                                extra_gradient_rms=float(np.sqrt(extra)), cross_term=cross, extra_squared=extra,
                                squared_error_change=after-before, identity_residual=after-before-cross-extra)
        out[name] = rows
    return out


def run(out, grids, dts, reference_grids, reference_dt):
    out.mkdir(parents=True, exist_ok=True)
    ax = shared_axes()
    particles = sites(*tensor_rule(ax, 2))
    # Independent common probes, distinct from all run-specific energy rules.
    probes = tensor_rule(ax, 3)
    report = dict(protocol=PROTOCOL, storage_start=guard(), mass_points=len(particles.X),
                  probe_points=len(probes[0]), initial_time=.001, final_time=.004,
                  region_bounds=dict(clamp=[.25, .3125], switch=[.4375, .5625]))
    initial_meta, initial = prepare_snapshot(out, particles)
    report['initial_state'] = initial_meta
    all_runs, final_fields = {}, {}
    report['time_convergence'] = {}
    for grid in grids:
        for case in ('control', 'shifted', 'enriched'):
            keys = []
            for dt in dts:
                name = f'g{grid}-{case}-dt{dt:.9f}'
                all_runs[name], final_fields[name] = run_path(out, name, grid, case, initial, particles, dt, .003)
                keys.append(name)
            pairs = []
            for a, b in zip(keys, keys[1:]):
                pair = dict(coarse=a, fine=b, dt=all_runs[a]['config']['dt'],
                            **mass_difference(final_fields[a], final_fields[b], particles))
                pairs.append(pair)
            for i in range(len(pairs)-1):
                pairs[i]['observed_order'] = float(np.log2(pairs[i]['velocity_rms']/pairs[i+1]['velocity_rms']))
            report['time_convergence'][f'g{grid}-{case}'] = dict(pairs=pairs,
                adjacent_differences_decrease=all(b['velocity_rms'] < a['velocity_rms'] for a, b in zip(pairs, pairs[1:])))
            report['runs'] = all_runs
            save(out/'results.json', report)
    reference_keys = []
    for grid in reference_grids:
        name = f'g{grid}-reference-dt{reference_dt:.9f}'
        all_runs[name], final_fields[name] = run_path(out, name, grid, 'control', initial, particles, reference_dt, .003)
        reference_keys.append(name)
    fine_grid = reference_grids[-1]
    time_key = f'g{fine_grid}-reference-dt{2*reference_dt:.9f}'
    all_runs[time_key], final_fields[time_key] = run_path(out, time_key, fine_grid, 'control', initial, particles, 2*reference_dt, .003)
    ref = final_fields[reference_keys[-1]]
    report['reference'] = dict(key=reference_keys[-1],
        spatial_resolution_check=spatial_report(final_fields[reference_keys[-2]], ref, probes),
        temporal_resolution_check=spatial_report(final_fields[time_key], ref, probes))
    report['spatial_comparison'] = {}
    report['template_differences'] = {}
    report['residual_error_decomposition'] = {}
    report['reference_ranking_sensitivity'] = {}
    for grid in grids:
        dt = dts[-1]
        control = final_fields[f'g{grid}-control-dt{dt:.9f}']
        for case in ('control', 'shifted', 'enriched'):
            name = f'g{grid}-{case}-dt{dt:.9f}'
            report['spatial_comparison'][name] = spatial_report(final_fields[name], ref, probes)
            if case != 'control':
                report['template_differences'][name] = spatial_report(final_fields[name], control, probes)
        report['residual_error_decomposition'][f'g{grid}'] = residual_error_decomposition(grid, initial,
            final_fields[f'g{grid}-enriched-dt{dt:.9f}'], control, ref, particles, probes)
        rankings = {}
        for refkey in (reference_keys[-2], time_key, reference_keys[-1]):
            scores = {}
            for case in ('shifted', 'enriched'):
                name = f'g{grid}-{case}-dt{dt:.9f}'
                comparison = (report['spatial_comparison'][name] if refkey == reference_keys[-1]
                              else spatial_report(final_fields[name], final_fields[refkey], probes))
                scores[case] = {region: {k: comparison[region][k+'_rms'] for k in ('x', 'F', 'P', 'v')}
                                for region in ('all', 'clamp', 'switch', 'bulk')}
            rankings[refkey] = scores
        report['reference_ranking_sensitivity'][f'g{grid}'] = rankings
    report['endpoint_quadrature'] = {}
    for grid in grids:
        name = f'g{grid}-enriched-dt{dts[-1]:.9f}'
        report['endpoint_quadrature'][name] = endpoint_quadrature(grid, 'enriched', initial, final_fields[name], particles,
                                                                out/f'{name}-quadrature-check.json')
    report['endpoint_quadrature'][reference_keys[-1]] = endpoint_quadrature(fine_grid, 'control', initial, ref, particles,
                                                                          out/f'{reference_keys[-1]}-quadrature-check.json')
    # Probe convergence is separate from material integration convergence.
    name = f'g{grids[-1]}-enriched-dt{dts[-1]:.9f}'
    fine_probe_report = spatial_report(final_fields[name], ref, tensor_rule(ax, 5))
    coarse_probe_report = report['spatial_comparison'][name]
    report['probe_check'] = {key: abs(coarse_probe_report['all'][key+'_rms']/fine_probe_report['all'][key+'_rms']-1)
                             for key in ('x', 'F', 'P', 'v')}
    report['numerical_checks'] = dict(
        trajectories=all(all(r['checks'].values()) for r in all_runs.values()),
        endpoint_quadrature=all(r['passed_1e-6'] for r in report['endpoint_quadrature'].values()),
        common_probes=max(report['probe_check'].values()) < 1e-6,
        decomposition=all(abs(row['identity_residual']) < 1e-14
                          for item in report['residual_error_decomposition'].values()
                          for target in item.values() for row in target.values()))
    report['runs'] = all_runs
    report['storage_end'] = guard()
    save(out/'results.json', report)
    return report


def plot(out, report):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpm-lite-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, row in report['time_convergence'].items():
        pairs = row['pairs']
        axes[0].loglog([p['dt'] for p in pairs], [p['velocity_relative'] for p in pairs], 'o-', label=name)
    for name, row in report['spatial_comparison'].items():
        label = name.split('-dt')[0]
        axes[1].plot(row['profile']['x'], row['profile']['F_rms'], label=label)
        axes[2].plot(row['profile']['x'], row['profile']['P_rms'], label=label)
    axes[0].set(title='Fixed-space time convergence', xlabel='dt (s)', ylabel='Adjacent velocity difference (mass norm)')
    axes[1].set(title='F error against finer reference', xlabel='Material X coordinate', ylabel='Local RMS difference')
    axes[2].set(title='PK1 error against finer reference', xlabel='Material X coordinate', ylabel='Local RMS difference')
    for ax in axes[1:]:
        ax.axvspan(.25, .3125, alpha=.08, color='red')
        ax.axvspan(.4375, .5625, alpha=.08, color='blue')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out/'summary.png', dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('docs/results/convergence-reference/v1'))
    parser.add_argument('--grids', nargs='+', type=int, default=[17, 33])
    parser.add_argument('--dts', nargs='+', type=float, default=[.00025, .000125, .0000625, .00003125])
    parser.add_argument('--reference-grids', nargs='+', type=int, default=[49, 65])
    parser.add_argument('--reference-dt', type=float, default=.000015625)
    args = parser.parse_args()
    if len(args.dts) < 3 or any(abs(a/2-b) > 1e-12 for a, b in zip(args.dts, args.dts[1:])):
        parser.error('use at least three successive dt halvings')
    if len(args.reference_grids) < 2:
        parser.error('at least two reference spatial resolutions are required')
    report = run(args.output, args.grids, args.dts, args.reference_grids, args.reference_dt)
    plot(args.output, report)
    print(json.dumps({k: v['adjacent_differences_decrease'] for k, v in report['time_convergence'].items()}), flush=True)
    if not all(report['numerical_checks'].values()):
        raise SystemExit(f"Numerical checks failed: {report['numerical_checks']}")


if __name__ == '__main__':
    main()
