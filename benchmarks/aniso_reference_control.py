"""Separate reference time refinement and smooth-compatible-history control."""
import argparse
import json
from pathlib import Path

import numpy as np

from engine.aniso_phase1.smooth_history import fit_smooth_history
from engine.aniso_phase1.convergence_reference import (
    geometry, knots, union_knots, tensor_rule, sites, load_fields, save_fields, directions)
from engine.aniso_phase1.consistent_transfer import material_response
from .aniso_convergence_reference import save, fingerprint
from .aniso_history_increment import guard
from .aniso_refinement import OLD, run_path, compare, initial_trace_audit


OUT = Path('docs/results/reference-control/v1')
PRIOR = Path('docs/results/refinement/v1')
BASE_DT = .00000390625


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    original = load_fields(OLD/'initial-switch-state.npz')
    X, w = tensor_rule(knots(geometry(17)), 5)
    position, fit = fit_smooth_history(original['position'], X, w)
    smooth = dict(position=position, velocity=original['velocity'])
    save_fields(out/'smooth-initial.npz', **smooth)
    probes = tensor_rule(knots(geometry(17)), 7)
    fitted7, _ = fit_smooth_history(original['position'], *probes)
    fit_check = compare(smooth, dict(position=fitted7,velocity=original['velocity']), probes)
    save(out/'fit-integration-check.json',fit_check)
    audit = dict(protocol='smooth-position-control-v1', fit=fit,
                 original_fingerprint=fingerprint(original), smooth_fingerprint=fingerprint(smooth),
                 initial_difference=compare(smooth, original, probes),
                 original_trace=initial_trace_audit(original), smooth_trace=initial_trace_audit(smooth),
                 storage=guard(), initial_energy={})
    for name, fields in [('original', original), ('smooth', smooth)]:
        x, F = fields['position'].evaluate(probes[0])
        psi, P = material_response(F, directions(probes[0]), geometry(17).params)
        v = fields['velocity'].evaluate(probes[0])[0]
        audit['initial_energy'][name] = dict(elastic=float(probes[1] @ psi),
            kinetic=float(.5*np.sum(probes[1][:, None]*v*v)), min_det=float(np.linalg.det(F).min()))
    boundary = geometry(97).X[geometry(97).fixed]
    audit['clamp_position_absolute'] = float(np.max(abs(position.evaluate(boundary)[0]-boundary)))
    audit['velocity_unchanged'] = fingerprint(dict(v=original['velocity'])) == fingerprint(dict(v=smooth['velocity']))
    if not audit['velocity_unchanged'] or audit['clamp_position_absolute'] > 1e-12:
        raise RuntimeError('initial-state control failed')
    save(out/'initial-audit.json', audit)
    return audit


def run(out, kind, grid, dt, device=None, tolerance=1e-9):
    folder = out/kind
    folder.mkdir(parents=True, exist_ok=True)
    initial = load_fields(OLD/'initial-switch-state.npz' if kind == 'original' else out/'smooth-initial.npz')
    axes = union_knots(knots(geometry(17)), knots(geometry(grid)))
    particles = sites(*tensor_rule(axes, 2))
    return run_path(folder, grid, 'reference', initial, particles, dt,
                    tolerance=tolerance, tensor=True, device=device)


def paths(out, kind, grid=97):
    found = {}
    initial = load_fields(OLD/'initial-switch-state.npz' if kind == 'original' else out/'smooth-initial.npz')
    expected = fingerprint(initial)
    folders = [out/kind]
    if kind == 'original':
        folders.insert(0, PRIOR)
    for folder in folders:
        for path in folder.glob(f'g{grid}-reference-dt*.json'):
            data = json.loads(path.read_text())
            config = data['config']
            if (path.with_suffix('.npz').exists() and all(data['checks'].values())
                    and config['initial'] == expected and config['order'] == 5
                    and config['duration'] == .003 and config['case'] == 'reference'
                    and config['tolerance'] == 1e-9):
                found[data['config']['dt']] = (path, data)
    return found


def analyze(out):
    result = dict(protocol='smooth-position-control-v1', planned_runs_complete=False,
                  time={}, time81={}, storage=guard(),
                  initial=json.loads((out/'initial-audit.json').read_text()))
    for grid, key in ((97,'time'), (81,'time81')):
        probes = tensor_rule(union_knots(knots(geometry(17)), knots(geometry(grid))), 3)
        for kind in ('original', 'smooth'):
            available = paths(out, kind, grid)
            pairs = []
            previous = None
            for dt, (path, data) in sorted(available.items(), reverse=True):
                fields = load_fields(path.with_suffix('.npz'))
                if previous is not None and abs(previous[0]-2*dt) < 1e-15:
                    metrics = compare(previous[1], fields, probes)
                    pair = dict(dt_coarse=previous[0], dt_fine=dt, metrics=metrics,
                                passed=all(metrics['relative'][k] < .01 for k in ('v','F','P')))
                    if pairs and abs(pairs[-1]['dt_fine']-previous[0]) < 1e-15:
                        pair['absolute_observed_order'] = {k: float(np.log2(
                            pairs[-1]['metrics']['all'][k+'_rms']/metrics['all'][k+'_rms'])) for k in ('v','F','P')}
                    pairs.append(pair)
                    print(grid, kind, dt, metrics['relative'], flush=True)
                previous = dt, fields
            result[key][kind] = dict(pairs=pairs, passed=bool(pairs and pairs[-1]['passed']))
            save(out/'results.json', result)
    original = result['time']['original']['pairs']
    smooth = result['time']['smooth']['pairs']
    result['smooth_to_original_same_dt'] = []
    for a in original:
        for b in smooth:
            if a['dt_fine'] == b['dt_fine']:
                result['smooth_to_original_same_dt'].append(dict(dt=a['dt_fine'],
                    relative_ratio={k: b['metrics']['relative'][k]/a['metrics']['relative'][k] for k in ('v','F','P')},
                    absolute_ratio={k: b['metrics']['all'][k+'_rms']/a['metrics']['all'][k+'_rms'] for k in ('v','F','P')}))
    result['space'] = {}
    result['local_enrichment'] = dict(implemented=False, gates={},
        reason='Require the original-history reference to resolve scheme differences and show localization.')
    spatial_probes = tensor_rule(union_knots(knots(geometry(81)),knots(geometry(97))),3)
    for kind in ('original','smooth'):
        a, b = paths(out,kind,81), paths(out,kind,97)
        common = sorted(set(a) & set(b))
        if common:
            dt = common[0]
            first = load_fields(a[dt][0].with_suffix('.npz'))
            second = load_fields(b[dt][0].with_suffix('.npz'))
            result['space'][kind] = dict(dt=dt,metrics=compare(first,second,spatial_probes))
    if 'original' in result['space']:
        dt = result['space']['original']['dt']
        refs = [load_fields(paths(out,'original',g)[dt][0].with_suffix('.npz')) for g in (81,97)]
        candidate = json.loads((PRIOR/'time-results.json').read_text())
        result['candidate_vs_reference'] = {}
        result['reference_to_scheme_ratios'] = {}
        from .aniso_refinement import field_axes
        for grid in (17,33):
            fields = [load_fields(PRIOR/candidate['paths'][f'g{grid}-{case}']['final_file'])
                      for case in ('shifted','enriched')]
            probes = tensor_rule(field_axes(*refs,*fields),3)
            delta = compare(refs[0],refs[1],probes)
            scheme = compare(fields[0],fields[1],probes)
            result['reference_to_scheme_ratios'][str(grid)] = {
                k:delta['all'][k+'_rms']/max(scheme['all'][k+'_rms'],1e-30) for k in ('x','F','P','v')}
            result['candidate_vs_reference'][str(grid)] = compare(fields[1],refs[1],probes)
        gates = dict(candidate_time=candidate['all_passed'],
            reference_time=all(result[key]['original']['passed'] for key in ('time','time81')),
            reference_small=all(v < .1 for row in result['reference_to_scheme_ratios'].values() for v in row.values()),
            localized=all(row['switch']['F_squared_error_share'] > .5 for row in result['candidate_vs_reference'].values()))
        result['local_enrichment']['gates'] = gates
        result['local_enrichment']['eligible'] = all(gates.values())
    records = [json.loads(p.read_text()) for kind in ('original','smooth') for p in (out/kind).glob('g*-reference-dt*.json')]
    result['new_runs'] = len(records)
    result['new_steps'] = sum(len(r['rows']) for r in records)
    result['planned_runs_complete'] = all(
        dt in paths(out,kind,grid)
        for kind in ('original','smooth') for grid in (81,97)
        for dt in ((BASE_DT,BASE_DT/2,BASE_DT/4) if (kind,grid)==('smooth',97) else (BASE_DT/2,BASE_DT/4)))
    result['numerical_checks'] = dict(all_runs=bool(records) and all(all(r['checks'].values()) for r in records))
    result['initial_step_energy_changes'] = {}
    for kind in ('original','smooth'):
        initial_energy = result['initial']['initial_energy'][kind]
        E0 = initial_energy['elastic']+initial_energy['kinetic']
        for path in (out/kind).glob('g*-reference-dt*.json'):
            data = json.loads(path.read_text())
            result['initial_step_energy_changes'][kind+'/'+path.stem] = data['rows'][0]['mechanical']-E0
    result['numerical_checks']['initial_step_dissipation'] = bool(records) and all(
        change <= 1e-12 for change in result['initial_step_energy_changes'].values())
    audit_path = out/'quadrature-audit.json'
    if audit_path.exists():
        result['quadrature'] = json.loads(audit_path.read_text())
        result['numerical_checks']['quadrature'] = all(
            kind in result['quadrature'] and result['quadrature'][kind]['dt'] == min(paths(out,kind))
            and result['quadrature'][kind]['endpoint']['passed_1e-6']
            and result['quadrature'][kind]['probe_passed_1e-6'] for kind in ('original','smooth'))
    if records:
        rows = [row for r in records for row in r['rows']]
        result['numerical_extrema'] = dict(
            max_residual=max(row['scaled_residual_inf'] for row in rows),
            min_det=min(row['min_det'] for row in rows),
            max_clamp_speed=max(row['clamp_speed'] for row in rows),
            max_budget_residual=max(abs(row['energy_budget_residual']) for row in rows),
            max_history_error=max(max(r['history_error'].values()) for r in records))
    result['storage_end'] = guard()
    save(out/'results.json', result)
    plot(out, result)
    return result


def audit(out, kind=None):
    """Independent final-state integration and common-probe checks."""
    from .aniso_refinement import endpoint_audit
    path = out/'quadrature-audit.json'
    result = json.loads(path.read_text()) if path.exists() else {}
    for kind in ((kind,) if kind is not None else ('original','smooth')):
        records = paths(out,kind)
        dt = min(records)
        final = load_fields(records[dt][0].with_suffix('.npz'))
        coarse = load_fields(paths(out,kind,81)[dt][0].with_suffix('.npz'))
        signature = fingerprint(final)
        if (kind in result and result[kind].get('final_fingerprint') == signature
                and result[kind].get('coarse_fingerprint') == fingerprint(coarse)
                and result[kind]['dt'] == dt and result[kind]['endpoint']['passed_1e-6']
                and result[kind]['probe_passed_1e-6']):
            continue
        # Reference test directions/mappings depend on geometry alone. This
        # helper evaluates energy, force and tangent at the supplied final F.
        initial = load_fields(OLD/'initial-switch-state.npz' if kind == 'original' else out/'smooth-initial.npz')
        result[kind] = dict(dt=dt, final_fingerprint=signature, coarse_fingerprint=fingerprint(coarse),
                            endpoint=endpoint_audit(out/kind,97,'reference',final,initial=initial))
        axes = union_knots(knots(geometry(81)),knots(geometry(97)))
        reports = [compare(coarse,final,tensor_rule(axes,order)) for order in (3,5)]
        change = {k:abs(reports[0]['all'][k+'_rms']/reports[1]['all'][k+'_rms']-1)
                  for k in ('x','F','P','v')}
        result[kind]['probe_3_5_relative_change'] = change
        result[kind]['probe_passed_1e-6'] = max(change.values()) < 1e-6
        save(out/'quadrature-audit.json',result)
    if not all(row['endpoint']['passed_1e-6'] and row['probe_passed_1e-6'] for row in result.values()):
        raise RuntimeError('independent quadrature checks failed')
    return result


def plot(out, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    ticks = sorted({p['dt_fine']*1e6 for row in result['time'].values() for p in row['pairs']})
    for ax, key in zip(axes, ('v','F','P')):
        for kind, row in result['time'].items():
            ax.loglog([p['dt_fine']*1e6 for p in row['pairs']],
                      [p['metrics']['relative'][key]*100 for p in row['pairs']], 'o-', label=kind)
        ax.axhline(1, color='gray', linestyle='--', label='1% adjacent difference')
        ax.set(xlabel='Fine time step (microseconds)', ylabel='Adjacent difference (%)', title=key)
        ax.set_xticks(ticks,labels=[f'{value:.4g}' for value in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which='both', alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle('Fixed grid 97: original history vs smooth compatible position history')
    fig.tight_layout(); fig.savefig(out/'time-control.png', dpi=160); plt.close(fig)
    if (all(kind in result.get('space',{}) for kind in ('original','smooth'))
            and result['space']['original']['dt'] == result['space']['smooth']['dt']):
        fig, axes = plt.subplots(1,3,figsize=(10,3.4))
        for ax,key in zip(axes,('F','P','v')):
            ax.bar(['Original','Smooth'],[result['space'][kind]['metrics']['all'][key+'_rms']
                   for kind in ('original','smooth')],color=['#386cb0','#f28e2b'])
            ax.set(title=key,ylabel='Volume-weighted RMS difference')
            ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
        dt = result['space']['original']['dt']*1e6
        fig.suptitle(f'Grid 81 to 97: dt = {dt:.7g} microseconds, 3 ms release')
        fig.tight_layout(); fig.savefig(out/'space-control.png',dpi=160); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('prepare','run','analyze','audit'), required=True)
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--kind', choices=('original','smooth'), default='original')
    p.add_argument('--grid', type=int, default=97)
    p.add_argument('--dt', type=float, default=BASE_DT/2)
    p.add_argument('--device')
    p.add_argument('--tolerance', type=float, default=1e-9)
    p.add_argument('--audit-kind', choices=('original','smooth'))
    args = p.parse_args()
    guard()
    if args.device:
        import warp as wp
        from utils.resource_guard import prepare_warp_cache
        from .aniso_history_increment import DATA_ROOT
        wp.config.kernel_cache_dir = prepare_warp_cache('/tmp/mpm-lite-warp-cache', DATA_ROOT)
    if args.stage == 'prepare': prepare(args.out)
    elif args.stage == 'analyze': analyze(args.out)
    elif args.stage == 'audit': audit(args.out,args.audit_kind)
    else: run(args.out, args.kind, args.grid, args.dt, args.device, args.tolerance)


if __name__ == '__main__':
    main()
