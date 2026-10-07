"""Static Q1/accurately integrated MLS/finer Q1 comparison at identical loading."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from engine.aniso_phase1.aligned_quadrature import LOW, HIGH, box_rule, assemble
from engine.aniso_phase1.mechanics_comparison import Q1Static, MLSStatic, solve_metrics, mode_audit, field_errors, physical_modes
from engine.aniso_phase1.consistent_transfer import MaterialQ1
from engine.aniso_phase1.trace_probe import stiffness
from benchmarks.aniso_quadrature_validation import setup
from benchmarks.aniso_consistent_transfer import guard


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def prototype_check(grid, q1):
    """Verify the candidate actually matches the recently added prototype."""
    model = MaterialQ1(grid, field='uniform')
    np.testing.assert_allclose(model.X, q1.nodes, atol=1e-14)
    moments = np.zeros((len(model.quadrature.X), 9, 9)); moments[:, 0, 0] = 1
    K = stiffness(model.quadrature.D, model.quadrature.weight, moments)
    operator = float(sparse.linalg.norm(K-q1.K)/sparse.linalg.norm(q1.K))
    N, D = q1.read(model.quadrature.X, True)
    basis = float(sparse.linalg.norm(N-model.quadrature.N))
    gradient = max(float(sparse.linalg.norm(a-b)) for a, b in zip(D, model.quadrature.D))
    return dict(matrix_relative=operator, value_absolute=basis, gradient_absolute=gradient)


def plot(out, rows, reference, convergence):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpm-lite-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for kind in ('q1', 'mls'):
        subset = [r for r in rows if r['space']==kind]
        axes[0].plot([r['grid'] for r in subset], [abs(r['tip_displacement']) for r in subset], 'o-', label=kind)
        axes[1].plot([r['grid'] for r in subset], [100*abs(r['tip_relative_to_reference']) for r in subset], 'o-', label=kind)
        for r in subset:
            axes[2].plot(list(r['mode_audit']['modes']), [v['stiffness_ratio'] for v in r['mode_audit']['modes'].values()], 'o-', label=r['name'])
    axes[0].axhline(abs(reference['tip_displacement']), ls='--', color='gray', label='Q1 fine reference')
    axes[0].set(xlabel='Grid', ylabel='Area-mean tip displacement magnitude')
    axes[1].set(xlabel='Grid', ylabel='Tip difference from reference (%)')
    axes[2].axhline(1, color='gray', ls='--'); axes[2].set(ylabel='Common-mode stiffness / analytic value')
    for ax in axes:
        ax.grid(alpha=.3); ax.legend(fontsize=7)
    axes[2].tick_params(axis='x', rotation=25)
    fig.suptitle(f"Static comparison; reference refinement changes tip by {100*abs(convergence):.3f}%")
    fig.tight_layout(); fig.savefig(out/'summary.png', dpi=160); plt.close(fig)


def run(args):
    storage = guard(); args.out.mkdir(parents=True, exist_ok=True)
    reference = Q1Static(args.reference_grid)
    ur, rr = solve_metrics(reference)
    rr.update(grid=args.reference_grid, space='q1', name=f'q1-g{args.reference_grid}')
    save(args.out/'reference.json', rr)
    guard(); fine = Q1Static(args.check_grid)
    uf, rf = solve_metrics(fine)
    refinement = rr['tip_displacement']/rf['tip_displacement']-1
    reference_fields=field_errors(reference, ur, fine, uf, order=2)
    save(args.out/'reference-check.json', dict(grid=args.check_grid, **rf, reference_relative_to_check=refinement,
                                             reference_field_comparison=reference_fields))
    del fine; gc.collect()
    print('REFERENCE', rr['tip_displacement'], 'refinement difference', refinement, flush=True)
    cloud = np.array(np.meshgrid(np.linspace(LOW[0], HIGH[0], 33), np.linspace(LOW[1], HIGH[1], 9),
                                np.linspace(LOW[2], HIGH[2], 9), indexing='ij')).reshape(3, -1).T
    reference_read = reference.read(cloud)@ur.reshape(-1, 3)
    records = []
    for grid in args.grids:
        for kind in ('q1', 'mls'):
            guard(); start = time.perf_counter(); extra = {}
            if kind == 'q1':
                candidate = Q1Static(grid)
                extra['prototype_equivalence'] = prototype_check(grid, candidate)
            else:
                _, _, blend = setup(grid)
                K, support = assemble(blend, box_rule(blend.h, 4), guard=guard)
                candidate = MLSStatic(grid, blend, K)
                extra['integration_support'] = support
                extra['constraint_rank'] = candidate.solver.rank
                extra['minimum_constrained_eigenvalue'] = float(candidate.solver.eigen[0])
                # Repeat dense-rule verification on the coarse case. For this
                # constant tangent, aligned order 4 integrates the polynomial exactly.
                if grid == min(args.grids):
                    K5, _ = assemble(blend, box_rule(blend.h, 5), guard=guard)
                    extra['quadrature_4_vs_5_matrix_relative'] = float(sparse.linalg.norm(K-K5)/sparse.linalg.norm(K))
                    del K5
            u, row = solve_metrics(candidate)
            row.update(name=f'{kind}-g{grid}', space=kind, grid=grid, **extra)
            row['mode_audit'] = mode_audit(candidate)
            row.update(field_errors(candidate, u, reference, ur, order=4))
            if kind == 'mls' and grid == min(args.grids):
                row['comparison_order3_control'] = field_errors(candidate, u, reference, ur, order=3)
            row['tip_relative_to_reference'] = row['tip_displacement']/rr['tip_displacement']-1
            row['tip_relative_to_check_reference'] = row['tip_displacement']/rf['tip_displacement']-1
            row['target_reaction_relative_to_reference'] = row['support_reaction_at_target_mean']/rr['support_reaction_at_target_mean']-1
            row['seconds_including_audits'] = time.perf_counter()-start
            controls = dict(force_balance=row['force_balance_relative']<1e-6,
                            moment_balance=row['moment_balance_relative']<1e-6,
                            solve_residual=row['free_residual_relative']<1e-6,
                            physical_clamp=row['clamp_max']<1e-9,
                            work_identity=row['work_identity_relative']<1e-6,
                            mode_clamp=row['mode_audit']['clamp_max']<1e-8)
            if kind=='q1':
                controls['prototype_equivalence'] = extra['prototype_equivalence']['matrix_relative']<1e-10
            if 'quadrature_4_vs_5_matrix_relative' in extra:
                controls['dense_quadrature'] = extra['quadrature_4_vs_5_matrix_relative']<1e-9
            row['numerical_checks'] = controls
            row['numerical_checks_passed'] = all(controls.values())
            # Moderate engineering screen, not a continuum accuracy certificate.
            row['five_percent_tip_and_reaction_screen'] = max(abs(row['tip_relative_to_reference']), abs(row['target_reaction_relative_to_reference']))<.05
            np.savez_compressed(args.out/(row['name']+'.npz'), reference=cloud,
                                displacement=candidate.read(cloud)@u.reshape(-1, 3),
                                reference_displacement=reference_read, nodes=candidate.nodes,
                                node_displacement=u.reshape(-1, 3),
                                mode_displacements=(candidate.read(cloud)@physical_modes(candidate.nodes)[0].reshape(len(candidate.nodes), -1)).reshape(len(cloud), 3, -1),
                                exact_modes=physical_modes(cloud)[0])
            records.append(row)
            result = dict(scope='small-strain static, uniform x fibers, same-law no mass/stabilization; CPU only',
                          storage=storage, reference=rr, reference_check=rf, check_grid=args.check_grid,
                          reference_refinement_relative=refinement, reference_within_two_percent=abs(refinement)<.02,
                          reference_field_comparison=reference_fields,
                          records=records, production_defaults_changed=False)
            save(args.out/'results.json', result)
            save(args.out/(row['name']+'.json'), row)
            print(row['name'], 'tip=', row['tip_displacement'], 'relative=', row['tip_relative_to_reference'],
                  'target reaction relative=', row['target_reaction_relative_to_reference'], 'checks=', controls, flush=True)
            del candidate; gc.collect()
    scalar_rows = [{k:v for k,v in r.items() if not isinstance(v, (dict,list))} for r in records]
    with (args.out/'summary.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in scalar_rows for k in r)))
        writer.writeheader(); writer.writerows(scalar_rows)
    plot(args.out, records, rr, refinement)
    if not all(r['numerical_checks_passed'] for r in records):
        raise SystemExit('numerical control failed; inspect results before interpretation')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grids', nargs='+', type=int, default=[17,33])
    p.add_argument('--reference-grid', type=int, default=65)
    p.add_argument('--check-grid', type=int, default=97)
    p.add_argument('--out', type=Path, default=Path('docs/results/mechanics-comparison/v1'))
    args=p.parse_args()
    if any(g not in (17,33) for g in args.grids):p.error('candidate grids must be 17 or 33')
    if args.reference_grid<=max(args.grids) or args.check_grid<=args.reference_grid:
        p.error('reference and check grids must increase beyond candidates')
    run(args)


if __name__=='__main__':main()
