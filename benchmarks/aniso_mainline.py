"""Frozen, bounded acceptance run for the original Lite anisotropic path.

python -m benchmarks.aniso_mainline freeze|tests|run|analyze --output PATH
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import time
import traceback
import unittest
import numpy as np
import warp as wp
from demos.aniso import Config, Scene
from utils.resource_guard import prepare_warp_cache, inspect_storage

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'docs/results/lite-aniso-mainline/v1'
BASE = Config('tensile', 9, .001, smooth_loading=True, linear_solver='pcg')
CASES = {'ISO': (0., 0.), 'F0': (200., 0.), 'F45': (200., 45.), 'F90': (200., 90.)}
TESTS = [
    'tests.test_aniso_phase1_material', 'tests.test_aniso_phase1_warp',
    'tests.test_aniso_tensile', 'tests.test_aniso_demo', 'tests.test_aniso_mainline',
    *['tests.test_aniso_variational.VariationalTests.'+name for name in (
        'test_potential_gradient_exact_hessian_and_full_spectrum',
        'test_compression_indefinite_exact_and_spd_modified_tangent',
        'test_original_energy_armijo_and_nonlinear_fallback',
        'test_reject_spectral_clamp_region_without_commit',
        'test_pcg_refuses_legacy_nonsymmetric_equation')],
    *['tests.test_aniso_phase1_solver.TestAnisoPhase1Solver.'+name for name in (
        'test_kf_zero_matches_legacy_one_step', 'test_block_renumbering_preserves_center_history',
        'test_zero_newton_iterations_do_not_commit_center_F',
        'test_negative_trial_jacobian_fails_without_particle_commit')],
    *['tests.test_aniso_history_reference.HistoryReferenceTests.'+name for name in (
        'test_prestretched_body_crosses_sparse_blocks_without_reset',
        'test_rotation_and_nonuniform_direction_transport')],
]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def hashes():
    files = set()
    for folder in ('engine', 'utils'):
        files.update((ROOT/folder).rglob('*.py'))
    files.update([ROOT/'demos/aniso.py', ROOT/'benchmarks/aniso_mainline.py', ROOT/'pyproject.toml', ROOT/'uv.lock'])
    for name in TESTS:
        files.add(ROOT / (name.split('.')[0]+'/'+name.split('.')[1]+'.py'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def freeze(out):
    if (out/'protocol.json').exists():
        raise RuntimeError('protocol already exists; use a new output directory')
    wp.init()
    configs = {}
    for case, (kf, angle) in CASES.items():
        for level, dt in [('coarse', .001), ('fine', .0005)]:
            config = replace(BASE, kf=kf, fiber_angle=angle, dt=dt)
            configs[f'{case}-{level}'] = asdict(config)
    protocol = dict(
        name='lite-aniso-mainline-v1', frozen_at=datetime.now(timezone.utc).isoformat(),
        device='cpu', precision='float64', solver='AnisotropicLiteImplicitSolver(MPMSolver)',
        git_commit=None, source_sha256=hashes(), configs=configs,
        units={'length':'m', 'time':'s', 'mass':'kg', 'stress':'Pa', 'force':'N'},
        physical={'mu':10., 'lambda':20., 'density':1., 'gravity':0.,
                  'body_bounds':[[.125,.375,.375],[.875,.625,.625]], 'volume':.046875,
                  'particles':192, 'particle_volume':.046875/192, 'grips_x':[.25,.75]},
        numerics={'enable_apic':True, 'damping':1., 'ppc_solver':1, 'samples_per_cell_axis':2,
                  'max_newton_iters':16, 'max_cg_iters':2187, 'cg_atol':1e-12,
                  'v_tol':1e-10, 'residual_atol':1e-10, 'cg_tol':1e-4},
        duration=.5, loading='u(t)=0.5*speed*T*(1-cos(pi*t/T)); monotone half cycle only',
        acceptance={'time_interval':[.05,.5], 'relative_time_error_max':.05,
                    'reaction_rms_floor_N':.001, 'norm':'trapezoid time-weighted RMS on common coarse physical times',
                    'direction_effect_factor':2., 'direction_rule':'fine pair RMS difference > 2*(both absolute time RMS errors), at least one pair',
                    'momentum_balance_abs_max_N':1e-7, 'grid_grip_velocity_abs_max_m_s':1e-12,
                    'particle_grip_displacement_abs_max_m':.001,
                    'particle_grip_note':'averages of particles initially inside grips; not exact moving material clamps',
                    'min_det_F':0., 'required_tests':'existing test-source tolerances frozen by source_sha256; no skips allowed'},
        tests=TESTS, cuda={'status':'not_run', 'reason':'only visible A100 occupied at 100%; CPU-only delivery claim'},
        environment={'platform':platform.platform(), 'processor':platform.processor(),
                     'python':platform.python_version(), 'numpy':np.__version__, 'warp':wp.__version__,
                     'cpu_model':next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), ''),
                     'storage':asdict(inspect_storage())},
        scope='3D, single elastic material, uniform reference fiber; no spatial convergence or performance claim')
    write_json(out/'protocol.json', protocol)


def verify(protocol):
    current = hashes()
    changed = [p for p, h in protocol['source_sha256'].items() if current.get(p) != h]
    if changed:
        raise RuntimeError('frozen sources changed: '+', '.join(changed))


def tests(out, protocol):
    if (out/'tests.json').exists():
        raise RuntimeError('test results already exist; preserve them in a separate output directory')
    class Result(unittest.TextTestResult):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.records = []
        def addSuccess(self, test):
            super().addSuccess(test)
            self.records.append({'test':test.id(), 'status':'passed'})
        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.records.append({'test':test.id(), 'status':'failed', 'detail':self._exc_info_to_string(err,test)})
        def addError(self, test, err):
            super().addError(test, err)
            self.records.append({'test':test.id(), 'status':'error', 'detail':self._exc_info_to_string(err,test)})
        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            self.records.append({'test':test.id(), 'status':'skipped', 'reason':reason})
    os.environ['ANISO_TEST_DEVICE'] = protocol['device']
    suite = unittest.defaultTestLoader.loadTestsFromNames(protocol['tests'])
    with (out/'tests.log').open('w') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=Result).run(suite)
    write_json(out/'tests.json', {'required_checks_passed':result.wasSuccessful() and not result.skipped,
                                 'tests_run':result.testsRun, 'records':result.records, 'cuda':protocol['cuda']})
    return result.wasSuccessful() and not result.skipped


def run_case(out, name, values, protocol):
    from engine.aniso_phase1.tensile import grid_values
    dest = out/'cases'/name
    if dest.exists():
        raise RuntimeError(f'{dest} exists; preserving previous records')
    dest.mkdir(parents=True)
    write_json(dest/'config.json', values)
    count = round(protocol['duration']/values['dt'])
    frames, times, Fs = [], [], []
    scene = None
    status = dict(run_completed=False, requested_steps=count, completed_steps=0, error=None)
    start = time.monotonic()
    try:
        scene = Scene(Config(**values), protocol['device'])
        assert type(scene.solver).__name__ == 'AnisotropicLiteImplicitSolver'
        frames.append(scene.reference.copy()); times.append(0.); Fs.append(scene.solver.ptc_F.numpy().copy())
        with (dest/'steps.jsonl').open('w', buffering=1) as log:
            for step in range(count):
                wp.config.kernel_cache_dir = prepare_warp_cache(wp.config.kernel_cache_dir)
                if not scene.step():
                    raise RuntimeError(f'step {step+1} failed; no particle commit')
                points, F = scene.frame()
                metrics = scene.metrics()
                nodes = scene.solver.energy_ledger.nodes
                velocity = grid_values(scene.solver, scene.solver.grid_v_new, nodes)
                selected = (nodes[:,0]*scene.solver.dx <= .25) | (nodes[:,0]*scene.solver.dx >= .75)
                expected = np.zeros_like(velocity)
                expected[nodes[:,0]*scene.solver.dx >= .75,0] = scene.loading_rows[-1]['loading_velocity']
                metrics['grid_grip_velocity_error'] = float(np.max(np.abs(velocity[selected]-expected[selected])))
                metrics['min_particle_det_F'] = float(np.linalg.det(F).min())
                if not np.isfinite(points).all() or not np.isfinite(F).all() or metrics['min_particle_det_F'] <= 0 or metrics['min_det_F'] <= 0:
                    raise RuntimeError('invalid committed state')
                log.write(json.dumps(metrics, allow_nan=False)+'\n')
                status['completed_steps'] = step+1
                if (step+1) % max(1,count//20) == 0 or step+1 == count:
                    frames.append(points.copy()); Fs.append(F.copy()); times.append(scene.solver.sim_time)
                    print(name, step+1, '/', count, flush=True)
        status['run_completed'] = True
    except Exception:
        status['error'] = traceback.format_exc()
    finally:
        status['wall_seconds'] = time.monotonic()-start
        if scene is not None:
            status['sim_time'] = scene.solver.sim_time
            status['last_step_stats'] = scene.solver.last_step_stats
            (dest/'trajectory.csv').write_text(scene.solver.energy_ledger.csv())
            if frames:
                np.savez_compressed(dest/'frames.npz', time=times, x=frames, F=Fs,
                                    reference=scene.reference, a0=scene.config.params.fiber_direction)
        # Preserve nonfinite failure diagnostics as text instead of invalid JSON.
        def clean(value):
            if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
            if isinstance(value, float) and not np.isfinite(value): return str(value)
            return value
        write_json(dest/'status.json', clean(status))
    del scene
    gc.collect()
    return status['run_completed']


def analyze(out, protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a = protocol['acceptance']
    histories, statuses, checks = {}, {}, {}
    for name in protocol['configs']:
        dest = out/'cases'/name
        statuses[name] = json.loads((dest/'status.json').read_text()) if (dest/'status.json').exists() else {'run_completed':False}
        if not statuses[name]['run_completed']:
            continue
        rows = [json.loads(line) for line in (dest/'steps.jsonl').read_text().splitlines()]
        histories[name] = {key:np.array([r[key] for r in rows]) for key in ('time','displacement','right_force','right_elastic_force','right_inertial_force')}
        checks[name] = {
            'max_momentum_balance_error_N':max(abs(r['momentum_balance_error']) for r in rows),
            'max_grid_grip_velocity_error_m_s':max(r['grid_grip_velocity_error'] for r in rows),
            'max_particle_grip_displacement_error_m':max(abs(r['measured_grip_displacement']-r['displacement']) for r in rows),
            'max_residual_norm':max(r['last_residual_norm'] for r in rows),
            'min_center_det_F':min(r['min_det_F'] for r in rows),
            'min_particle_det_F':min(r['min_particle_det_F'] for r in rows),
            'final_loading_work_J':rows[-1]['loading_work'],
            'final_mechanical_minus_loading_work_J':rows[-1]['mechanical_minus_loading_work'],
            'all_steps_converged':all(r['converged'] for r in rows),
        }
        c=checks[name]
        c['passed'] = bool(c['all_steps_converged'] and c['max_residual_norm'] <= 1e-10
                          and c['max_momentum_balance_error_N'] <= a['momentum_balance_abs_max_N']
                          and c['max_grid_grip_velocity_error_m_s'] <= a['grid_grip_velocity_abs_max_m_s']
                          and c['max_particle_grip_displacement_error_m'] <= a['particle_grip_displacement_abs_max_m'])
    sensitivity, directions = {}, {}
    aligned, perturbations = {}, {}
    for case in CASES:
        if not all(case+'-'+v in histories for v in ('coarse','fine')): continue
        c, f = histories[case+'-coarse'], histories[case+'-fine']
        mask = (c['time'] >= a['time_interval'][0]-1e-12) & (c['time'] <= a['time_interval'][1]+1e-12)
        t = c['time'][mask]
        def rms(y): return float(np.sqrt(np.trapezoid(y*y, t)/(t[-1]-t[0])))
        rf = np.interp(t, f['time'], f['right_force'])
        delta = c['right_force'][mask]-rf
        absolute = rms(delta)
        relative = absolute/max(rms(rf), a['reaction_rms_floor_N'])
        sensitivity[case] = {'relative':relative, 'absolute_rms_N':absolute, 'max_absolute_N':float(np.max(abs(delta))),
                             'fine_rms_N':rms(rf), 'passed':relative <= a['relative_time_error_max'],
                             'inertial_to_total_rms':rms(np.interp(t, f['time'], f['right_inertial_force']))/max(rms(rf),a['reaction_rms_floor_N'])}
        aligned[case]=rf; perturbations[case]=absolute
    for left, right in itertools.combinations(aligned,2):
        diff = rms(aligned[left]-aligned[right])
        threshold = a['direction_effect_factor']*(perturbations[left]+perturbations[right])
        directions[left+'-'+right] = {'rms_difference_N':diff, 'threshold_N':threshold, 'resolved':diff>threshold}
    test_record = json.loads((out/'tests.json').read_text()) if (out/'tests.json').exists() else {}
    summary = {'run_completed':all(s['run_completed'] for s in statuses.values()),
               'required_checks_passed':test_record.get('required_checks_passed',False),
               'time_sensitivity_passed':len(sensitivity)==4 and all(r['passed'] for r in sensitivity.values()),
               'direction_effect_resolved':any(r['resolved'] for r in directions.values()),
               'trajectory_checks_passed':len(checks)==8 and all(r['passed'] for r in checks.values()),
               'space_accuracy':'not_verified', 'cuda':protocol['cuda'],
               'cases':statuses, 'trajectory_checks':checks, 'time_sensitivity':sensitivity, 'direction_pairs':directions}
    summary['accepted'] = all(summary[key] for key in ('run_completed','required_checks_passed','time_sensitivity_passed','direction_effect_resolved','trajectory_checks_passed'))
    write_json(out/'summary.json', summary)
    fig, axes = plt.subplots(1,3,figsize=(15,4), constrained_layout=True)
    for case, color in zip(CASES, ('black','tab:red','tab:orange','tab:blue')):
        for level, style in [('coarse','--'),('fine','-')]:
            h=histories.get(case+'-'+level)
            if h is None: continue
            for ax,key in zip(axes,('right_force','right_elastic_force','right_inertial_force')):
                ax.plot(h['displacement'],h[key],style,color=color,label=f'{case} dt={protocol["configs"][case+"-"+level]["dt"]}')
    for ax,title in zip(axes,('Total dynamic reaction','Elastic reaction','Inertial reaction')):
        ax.set(xlabel='Command displacement (m)',ylabel='Right reaction (N)',title=title); ax.grid(alpha=.3)
    axes[0].legend(fontsize=7)
    fig.savefig(out/'reaction-displacement.png',dpi=180); plt.close(fig)
    fig, axes = plt.subplots(4,3,figsize=(12,8),sharex=True,sharey=True,constrained_layout=True)
    for row, case in enumerate(CASES):
        path=out/'cases'/(case+'-fine')/'frames.npz'
        if not path.exists(): continue
        with np.load(path) as frames:
            for col, fraction in enumerate((.25,.5,1.)):
                idx=int(np.argmin(abs(frames['time']-protocol['duration']*fraction)))
                x, F = frames['x'][idx], frames['F'][idx]
                direction=np.einsum('pij,j->pi',F,frames['a0']); direction/=np.linalg.norm(direction,axis=1)[:,None]
                axes[row,col].scatter(x[:,0],x[:,1],s=3,color='tab:blue')
                axes[row,col].quiver(x[::3,0],x[::3,1],direction[::3,0]*.025,direction[::3,1]*.025,
                                     angles='xy',scale_units='xy',scale=1,color='darkorange',width=.003)
                axes[row,col].set(title=f'{case} t={frames["time"][idx]:.3f} s',xlim=(.1,.91),ylim=(.33,.67),aspect='equal',xlabel='x (m)',ylabel='y (m)')
    fig.suptitle('Physical positions and normalized F a0; XY projection; displacement scale = 1')
    fig.savefig(out/'scene-comparison.png',dpi=180); plt.close(fig)
    return summary['accepted']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','tests','run','analyze'))
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--case', choices=tuple(CASES))
    args=parser.parse_args(); out=args.output
    out.mkdir(parents=True,exist_ok=True)
    wp.config.kernel_cache_dir=prepare_warp_cache(os.environ.get('MPM_LITE_WARP_CACHE','/tmp/mpm-lite-warp-cache'))
    if args.action=='freeze': freeze(out); return
    protocol=json.loads((out/'protocol.json').read_text()); verify(protocol)
    if args.action=='tests':
        if not tests(out,protocol): raise SystemExit(1)
    elif args.action=='run':
        if not json.loads((out/'tests.json').read_text())['required_checks_passed']:
            raise RuntimeError('required checks must pass before formal trajectories')
        completed=True
        for name, values in protocol['configs'].items():
            if args.case and not name.startswith(args.case+'-'): continue
            completed=run_case(out,name,values,protocol) and completed
        if not completed: raise SystemExit(1)
    elif not analyze(out,protocol): raise SystemExit(2)


if __name__=='__main__': main()
