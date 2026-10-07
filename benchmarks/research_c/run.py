"""Append-only C audit, legacy regression, short screens and four full cycles.

Examples (single-thread BLAS recommended):
  python -m benchmarks.research_c.run diagnostics --output <new-directory>
  python -m benchmarks.research_c.run legacy --output <new-directory>
  python -m benchmarks.research_c.run cycles --output <new-directory>

Directories must be new. A failed run saves its traceback and never reports
partial trajectories as completed. Timings are observational, not speedups.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time
import traceback

import numpy as np
import scipy
import scipy.linalg as la

from engine.aniso_phase1.research_c.model import Model, quadrature
from engine.aniso_phase1.research_c.dynamics import AVF
from engine.aniso_phase1.research_c.legacy import legacy_v20, ARCHIVE, ROOT
from engine.aniso_phase1.research_c.migration import migrate


DT = [.0005, .00025, .000125, .0000625]
STAGES = [('load', 0., .5), ('hold', .5, .6), ('unload', .6, 1.1), ('end_hold', 1.1, 1.6)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    with Path(path).open('x') as file:
        json.dump(data, file, indent=2, allow_nan=False)
        file.write('\n')


def sources():
    paths = []
    for name in ('engine/aniso_phase1/research_c', 'benchmarks/research_c', 'tests/research_c'):
        paths += sorted((ROOT/name).glob('*.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def baseline():
    records = {}
    for name in ('source-delivered-sha256.json', 'artifact-sha256.json'):
        path = ROOT/'docs/results/lite-aniso-mainline/v22'/name
        entries = json.loads(path.read_text())
        bad = [n for n, digest in entries.items() if not (ROOT/n).is_file() or sha(ROOT/n) != digest]
        records[name] = dict(manifest_sha256=sha(path), count=len(entries), mismatches=bad)
        if bad:
            raise ValueError(f'baseline changed: {bad}')
    return records


def protocol(kind):
    data = dict(schema_version=1, producer='C', kind=kind, baseline=baseline(), source_sha256=sources(),
        baseline_archive_sha256=sha(ROOT/'docs/results/lite-aniso-mainline/v22/source-delivered.zip'),
        units=dict(length='m', time='s', mass='kg', energy='J', stress='Pa'), dtype='float64', device='CPU',
        frame='reference Cartesian; q=(carrier displacement,local coefficients); x=X+Nq; F=I+Dq',
        local_coefficient_units='m; dimensionless 4t(1-t)(2t-1)^k, k=0,1,2; shared xyz',
        material=dict(mu=10, lam=20, k_f=200, fiber=[2**-.5, 2**-.5, 0]), density=1.,
        physical_box=[[.125, .875], [.375, .625], [.375, .625]], reference_volume=.046875,
        constraints='rigid material volumes x<=.25 and x>=.75; original .005 m cosine loading',
        inertia='constant reference continuum full N^T rho dV N; C derived, no independent APIC microinertia',
        model='small component-closed Q1 carriers + 3 local Q2/Q3/Q4 bubbles, not full v22 candidate',
        stabilization='zero for this conforming small FE reference; original v20 stabilization in legacy only',
        full_material_orders=[5, 3, 3], path_order=3, dt=DT, duration=1.6, seed=220930,
        thresholds=dict(time_relative=.02, reaction_absolute_N=1e-5, stress_absolute_Pa=1e-4,
            true_residual_atol=1e-11, true_residual_rtol=1e-8, constraint_atol=1e-9,
            inertia_relative=1e-6, derivative_relative=2e-5, energy_ledger_absolute_J=1e-9,
            legacy_relative=1e-6, legacy_reaction_absolute_N=1e-8, legacy_stress_absolute_Pa=1e-8),
        time_comparison='raw interval reactions at common end times, no averaging/filtering/deleted first step; weighted full PK1',
        near_zero='pass absolute or relative gate; report both with fixed absolute floor',
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
            threads={n: os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
            cpu=platform.processor(), load=os.getloadavg(), platform=platform.platform(),
            cuda='not used or certified', timing='shared host, observational; no formal speedup claim'),
        disk=dict(system_free_bytes=shutil.disk_usage('/').free, data_free_bytes=shutil.disk_usage('/root/autodl-tmp').free))


    if kind == 'legacy':
        archived = json.loads((ARCHIVE/'cycle-protocol.json').read_text())
        data.update(model='legacy_v20: exact archived gauss3-condensed-L0 execution chain',
            frame='legacy carrier TOTAL positions Y; material F=gradient(B,Y); no added identity',
            local_coefficient_units='not applicable: no new local alpha in legacy_v20',
            inertia='archived Gauss3 kinetic sites, geometry-dependent APIC metric and material-null constraint',
            stabilization='unchanged original carrier patch potential and exact stationary material-null constraint',
            full_material_orders=None, material_rule='unchanged original 192 material points',
            kinetic_rule='archived positive Gauss3 sites and original moving transfer',
            path_order=2, dt=[archived['cases'][0]['dt']],
            archived_protocol=archived, archived_protocol_sha256=sha(ARCHIVE/'cycle-protocol.json'),
            solver_tolerance='unchanged FastIntegratedAVF implementation; legacy 1e-17 absolute residual',
            thresholds=dict(legacy_relative=1e-6, legacy_reaction_absolute_N=1e-8,
                            legacy_stress_absolute_Pa=1e-8, legacy_state_absolute=1e-7),
            time_comparison='complete archived L0 step-by-step replay, plus all seven stored state snapshots',
            near_zero='legacy regression uses the stated fixed absolute and relative tolerances')
    return data


def legacy(dest):
    solver, p, meta = legacy_v20()
    name = 'gauss3-condensed-L0'; dt = p['cases'][0]['dt']; n = round(p['duration']/dt)
    source = ARCHIVE/'cases'/name
    refrows = [json.loads(line) for line in (source/'steps.jsonl').read_text().splitlines()]
    with np.load(source/'stress.npz') as data:
        refP = data['P'].copy()
    errors = dict(reaction_max_abs_N=0., stress_max_abs_Pa=0., total_energy_max_abs_J=0.,
                  state_max_abs=0., reaction_squared=0., reference_reaction_squared=0.,
                  stress_squared=0., reference_stress_squared=0.)
    start = time.monotonic()
    with (dest/'steps.jsonl').open('x', buffering=1) as log:
        for k in range(n):
            row = solver.step(dt)
            P = solver.energy.evaluate(solver.state.Y)['P']
            er = row['reaction_N']-refrows[k]['reaction_N']; ep = P-refP[k+1]
            errors['reaction_max_abs_N'] = max(errors['reaction_max_abs_N'], abs(er))
            errors['stress_max_abs_Pa'] = max(errors['stress_max_abs_Pa'], float(np.max(abs(ep))))
            errors['total_energy_max_abs_J'] = max(errors['total_energy_max_abs_J'], abs(row['total_J']-refrows[k]['total_J']))
            errors['reaction_squared'] += er*er; errors['reference_reaction_squared'] += refrows[k]['reaction_N']**2
            errors['stress_squared'] += float(np.sum(ep*ep)); errors['reference_stress_squared'] += float(np.sum(refP[k+1]**2))
            audit = source/f'audit-{k+1:06d}.npz'
            if audit.exists():
                with np.load(audit) as snap:
                    for field in ('x', 'Y', 'v', 'C'):
                        errors['state_max_abs'] = max(errors['state_max_abs'], float(np.max(abs(getattr(solver.state, field)-snap[field]))))
            log.write(json.dumps(row, allow_nan=False)+'\n')
            if (k+1) % 200 == 0:
                print('legacy', k+1, n, 'seconds', round(time.monotonic()-start, 1), flush=True)
    errors['reaction_relative'] = float(np.sqrt(errors['reaction_squared']/errors['reference_reaction_squared']))
    errors['stress_relative'] = float(np.sqrt(errors['stress_squared']/errors['reference_stress_squared']))
    passed = (errors['reaction_relative'] < 1e-6 or errors['reaction_max_abs_N'] < 1e-8) and \
             (errors['stress_relative'] < 1e-6 or errors['stress_max_abs_Pa'] < 1e-8) and errors['state_max_abs'] < 1e-7
    return dict(completed=True, passed=bool(passed), steps=n, final_time=solver.state.time,
                errors=errors, reference_files={f.name: sha(f) for f in source.iterdir() if f.is_file()},
                seconds=time.monotonic()-start, scope='full archived L0 cycle regression; archived L1-L3 not rerun')


def diagnostics(dest):
    m = Model(); rng = np.random.default_rng(220930); q = rng.normal(scale=1e-4, size=(m.space.n, 3))
    d = rng.normal(size=q.shape); out = m.evaluate(q, tangent=True)
    curves = []
    for eps in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        a, b = m.evaluate(q+eps*d), m.evaluate(q-eps*d)
        fdU = (a['U']-b['U'])/(2*eps); exact = float(np.sum(out['force']*d))
        fd = (a['force']-b['force'])/(2*eps); action = m.tangent_action(q, d)
        curves.append(dict(eps=eps, energy_absolute=abs(fdU-exact),
            energy_relative=abs(fdU-exact)/max(abs(exact), 1e-8),
            tangent_relative=float(la.norm(fd-action)/max(la.norm(action), 1e-8))))
    reference = Model(orders=(7, 5, 5)); massrows = []
    for order in (2, 3, 4, 5, 6):
        try:
            candidate = Model(orders=(order, 3, 3))
            massrows.append(dict(order=order, matrix_relative=float(la.norm(candidate.M-reference.M)/la.norm(reference.M)),
                local_block_relative=float(la.norm(candidate.M[20:, 20:]-reference.M[20:, 20:])/la.norm(reference.M[20:, 20:])),
                min_eigenvalue=float(la.eigvalsh(candidate.M)[0]), accepted=True))
        except ValueError as exc:
            massrows.append(dict(order=order, accepted=False, reason=str(exc)))
    B = m.B[:, :, m.free].reshape(-1, len(m.free)); singular = la.svdvals(B)
    K = m.evaluate(m.rest().q, tangent=True)['K'][np.ix_(m.free, m.free)]
    lam, phi = la.eigh(K, m.Mff); omega = np.sqrt(lam)
    modal = dict(frequencies_rad_s=omega.tolist(), mass_rank=int(np.linalg.matrix_rank(m.M)),
                 material_gradient_rank=int(np.linalg.matrix_rank(B)), allowed_vector_dofs=len(m.free),
                 gradient_singular_values=singular.tolist(), material_null_elimination='none',
                 eigen_residual=float(la.norm(K@phi-(m.Mff@phi)*lam)/la.norm(K@phi)))
    # Compare nonlinear AVF at tiny amplitude to independent linear modes.
    s = m.rest(); s.time = 1.2
    amplitude = 1e-6/la.norm(phi[:, 0]); s.q.ravel()[m.free] = amplitude*phi[:, 0]
    sol = AVF(m, s, residual_atol=1e-13, residual_rtol=1e-9)
    errors = []
    for k in range(100):
        sol.step(.0005)
        t = (k+1)*.0005
        exact = amplitude*phi[:, 0]*np.cos(omega[0]*t)
        errors.append(float(la.norm(sol.state.q.ravel()[m.free]-exact)/1e-6))
    modal['nonlinear_small_amplitude_vs_analytic_max_relative'] = max(errors)
    # Freeze path quadrature convergence and two solver tolerances, independently.
    W = rng.normal(scale=.01, size=q.shape); path = []
    for order in (1, 2, 3, 5):
        sol = AVF(m, path_order=order); force, _ = sol.path(q, W, .0005)
        path.append(dict(order=order, work_error_J=m.evaluate(q+.0005*W)['U']-out['U']-.0005*float(np.sum(force*W))))
    a, b = Model(local_modes=1), m; state = a.rest(); state.q[-1, 0] = .001; state.velocity[-1, 1] = .001
    expanded, expansion = migrate(a, b, state)
    restored, nested = migrate(b, a, expanded)
    expanded.q[-1, 0] = .001
    rejected, loss = migrate(b, a, expanded)
    rule, rule_report = migrate(m, Model(orders=(6, 4, 4)), m.rest())
    result = dict(derivative_curves=curves, inertia_convergence=massrows, modal=modal, path_quadrature=path,
                  migration=[expansion, nested, loss, rule_report],
                  nested_cumulative_absolute_J={k: expansion['absolute_jumps'][k]+nested['absolute_jumps'][k]
                                                for k in expansion['absolute_jumps']},
                  nested_cumulative_signed_J={k: expansion['signed_jumps'][k]+nested['signed_jumps'][k]
                                              for k in expansion['signed_jumps']},
                  passed=bool(curves[-2]['tangent_relative'] < 2e-5 and massrows[-2]['matrix_relative'] < 1e-6
                              and max(errors) < .02 and expansion['accepted'] and nested['accepted']
                              and not loss['accepted'] and rule_report['accepted']))
    np.savez_compressed(dest/'matrices.npz', mass=m.M3, stiffness=m.evaluate(m.rest().q, tangent=True)['K'],
                        free=m.free, eigenvalues=lam, eigenvectors=phi, N=m.N, D=m.D, X=m.X, V=m.V)
    return result


def trajectory(dest, dt, duration=1.6, *, moving=True, local_modes=3, atol=1e-11, origin=(.003, .002, .001)):
    dest.mkdir(); m = Model(local_modes); sol = AVF(m, moving_grid=moving, residual_atol=atol, grid_origin=origin)
    n = round(duration/dt)
    if not np.isclose(n*dt, duration, rtol=0, atol=1e-12):
        raise ValueError('duration must be a complete number of steps')
    # Memory-mapped complete histories allow checksums/checkpoints without a
    # large resident trajectory. All points and time levels are retained.
    P = np.lib.format.open_memmap(dest/'PK1.npy', mode='w+', dtype=float, shape=(n+1, len(m.X), 3, 3))
    P[0] = m.evaluate(sol.state.q)['P']
    reaction = np.zeros(n); timegrid = np.arange(n+1)*dt
    sums = dict(boundary_work_J=0., constraint_kinetic_loss_J=0., path_quadrature_error_J=0.,
                transfer_energy_jump_J=0., energy_balance_J=0.)
    maximum = dict(true_residual=0., displacement_constraint=0., velocity_constraint=0.,
                   transfer_error=0., budget_defect_J=0., linear_momentum_error=0.)
    det = 1.; crossings = 0; start = time.monotonic(); done = 0
    snapshots = {round(t/dt) for t in (.05, .5, .6, .85, 1.1, 1.4, 1.6) if t <= duration+1e-12}
    try:
        with (dest/'steps.jsonl').open('x', buffering=1) as log:
            for k in range(n):
                row = sol.step(dt); done = k+1; P[k+1] = sol.last['P']; reaction[k] = row['reaction_N']
                for key in sums:
                    sums[key] += row[key]
                for key in maximum:
                    maximum[key] = max(maximum[key], abs(row[key]))
                crossings += row['crossed_particles']; det = min(det, row['min_det_F'])
                log.write(json.dumps(row, allow_nan=False)+'\n')
                if k+1 in snapshots:
                    kin = m.space.kinematics(m.X, sol.state.q, sol.state.velocity)
                    np.savez_compressed(dest/f'audit-{k+1:06d}.npz', q=sol.state.q, velocity=sol.state.velocity,
                        time=sol.state.time, x=kin['x'], F=kin['F'], v=kin['v'], C=kin['C'], P=sol.last['P'],
                        fiber_strain=np.einsum('pi,pi->p', kin['F']@m.params.fiber_direction,
                                              kin['F']@m.params.fiber_direction)-1)
                if (k+1) % 800 == 0:
                    P.flush()
                    print(dest.name, k+1, n, 'seconds', round(time.monotonic()-start, 1), flush=True)
    except Exception:
        write(dest/'failure.json', dict(completed=False, steps=done, requested_steps=n, final_time=sol.state.time,
              failures=sol.failures, error=traceback.format_exc()))
        raise
    P.flush()
    np.savez_compressed(dest/'trajectory.npz', time=timegrid, reaction=reaction, q=sol.state.q, velocity=sol.state.velocity,
                        X=m.X, V=m.V)
    end = sol.last; audit = m.evaluate(sol.state.q)
    summary = dict(completed=True, steps=n, requested_steps=n, final_time=sol.state.time,
        dt=dt, moving_grid=moving, local_modes=local_modes, source_signature=m.signature,
        min_det_F=det, crossings=crossings, maximum=maximum, sums=sums,
        final_total_J=end['row']['total_J'], final_stress_rms_Pa=end['row']['stress_rms_Pa'],
        snapshot_recompute_error=float(np.max(abs(audit['P']-P[-1]))), seconds=time.monotonic()-start,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        system_free_bytes=shutil.disk_usage('/').free)
    write(dest/'status.json', summary)
    return summary


def compare(coarse, fine, duration):
    with np.load(coarse/'trajectory.npz') as a, np.load(fine/'trajectory.npz') as b:
        tc = a['time']; tf = b['time']; rc = a['reaction']; rf = b['reaction']; V = a['V']
    ratio = round((tc[1]-tc[0])/(tf[1]-tf[0])); ids = np.arange(len(tc))*ratio
    if ids[-1] != len(tf)-1 or not np.allclose(tc, tf[ids], rtol=0, atol=1e-12):
        raise ValueError('time grids must share exact output times')
    Pc = np.load(coarse/'PK1.npy', mmap_mode='r'); Pf = np.load(fine/'PK1.npy', mmap_mode='r')
    aligned_r = rf[ids[1:]-1]
    regions = [('full', 0., duration)]+[(n, lo, min(hi, duration)) for n, lo, hi in STAGES if lo < duration]
    rows = []
    def metric(a, b, weights=None, floor=1e-8):
        if weights is None:
            err = float(np.sqrt(np.mean((a-b)**2))); scale = float(np.sqrt(np.mean(b*b)))
        else:
            err = float(np.sqrt(np.mean(np.sum((a-b)**2, axis=(-1, -2))@(weights/weights.sum()))))
            scale = float(np.sqrt(np.mean(np.sum(b*b, axis=(-1, -2))@(weights/weights.sum()))))
        return dict(absolute_rms=err, reference_rms=scale, relative=err/max(scale, floor),
                    passed=bool(err <= floor or err/max(scale, floor) <= .02))
    for name, lo, hi in regions:
        mask = (tc[1:] > lo+1e-12)&(tc[1:] <= hi+1e-12)
        points = np.flatnonzero(mask)+1
        r = metric(rc[mask], aligned_r[mask], floor=1e-5)
        stress = metric(Pc[points], Pf[ids[points]], V, 1e-4)
        terminal = metric(Pc[points[-1]][None], Pf[ids[points[-1]]][None], V, 1e-4)
        rows.append(dict(stage=name, reaction=r, stress=stress, endpoint_stress=terminal,
                         passed=r['passed'] and stress['passed'] and terminal['passed']))
    return dict(coarse=coarse.name, fine=fine.name, stages=rows, passed=all(r['passed'] for r in rows))


def cycles(dest):
    # C9 screen precedes long cycles. Each ablation has the same short window.
    variants = [('no_transfer', dict(moving=False)), ('moving', {}),
                ('shifted_origin', dict(origin=(.017, -.009, .013))),
                ('no_alpha', dict(local_modes=0)), ('tight_solve', dict(atol=1e-13))]
    short = {name: trajectory(dest/name, .0005, .05, **kw) for name, kw in variants}
    screen = [compare(dest/'no_transfer', dest/'moving', .05),
              compare(dest/'moving', dest/'shifted_origin', .05),
              compare(dest/'moving', dest/'tight_solve', .05)]
    # The four dt levels are screened at a common duration before C10.
    for i, dt in enumerate(DT):
        trajectory(dest/f'screen-L{i}', dt, .05)
    time_screen = [compare(dest/f'screen-L{i}', dest/f'screen-L{i+1}', .05) for i in range(3)]
    write(dest/'short-screen.json', dict(variants=short, same_model=screen, temporal=time_screen,
        alpha_ablation='different physical space; record response, no equality gate'))
    if not all(s['passed'] for s in screen) or not time_screen[-1]['passed']:
        return dict(completed=False, passed=False, reason='C9 screen failed; full cycles not launched')
    runs = []
    for i, dt in enumerate(DT):
        runs.append(trajectory(dest/f'L{i}', dt))
    comparisons = [compare(dest/f'L{i}', dest/f'L{i+1}', 1.6) for i in range(3)]
    contraction = []
    for previous, finest in zip(comparisons[-2]['stages'], comparisons[-1]['stages']):
        for field in ('reaction', 'stress', 'endpoint_stress'):
            a, b = previous[field]['absolute_rms'], finest[field]['absolute_rms']
            floor = 1e-5 if field == 'reaction' else 1e-4
            contraction.append(dict(stage=finest['stage'], field=field, previous=a, finest=b,
                                    passed=bool(b < a or b <= floor)))
    return dict(completed=True, passed=bool(comparisons[-1]['passed'] and all(r['passed'] for r in contraction)),
                runs=runs, comparisons=comparisons, finest_contraction=contraction,
                scope='small reference with lossless moving-grid storage, not Eulerian MPM or full v22 spatial certification')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('kind', choices=['legacy', 'diagnostics', 'cycles'])
    parser.add_argument('--output', required=True, type=Path); args = parser.parse_args()
    dest = args.output.resolve(); dest.mkdir(parents=True, exist_ok=False)
    frozen = protocol(args.kind); write(dest/'protocol.json', frozen)
    try:
        result = globals()[args.kind](dest)
    except Exception:
        write(dest/'failure.json', dict(completed=False, error=traceback.format_exc()))
        raise
    result['source_unchanged'] = frozen['source_sha256'] == sources()
    result['baseline_after'] = baseline()
    write(dest/'summary.json', result)
    write(dest/'artifacts-sha256.json', {str(f.relative_to(dest)): sha(f) for f in sorted(dest.rglob('*')) if f.is_file()})
    print(json.dumps({k: v for k, v in result.items() if k in ('passed', 'completed', 'source_unchanged', 'reason')}), flush=True)
    if result.get('passed') is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
