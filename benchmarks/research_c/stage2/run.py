"""C stage 2 evidence runner. Run with two BLAS threads; no parent mutations."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import shutil
import time
import numpy as np
import scipy.linalg as la

from engine.aniso_phase1.research_d.frozen_inputs import load_frozen_inputs
from engine.aniso_phase1.research_d.common_kinetic import PointInertia
from engine.aniso_phase1.research_d.identity import sha, digest, write_json
from engine.aniso_phase1.research_c.stage2.model import Boundary, DynamicModel, recover_mass, rest_stiffness
from engine.aniso_phase1.research_c.stage2.dynamics import AVF, StepRejected
from engine.aniso_phase1.research_c.stage2.checkpoint import source_manifest, binding, save_checkpoint, load_checkpoint

ROOT = Path(__file__).resolve().parents[3]
PARENT = '55682a7b8e90b62c1306818cdf9c1f060b4286a174da069ce2217aa5b53b3c7c'
BUNDLE = ROOT/'docs/results/parallel-v22/integration/20260930T054100Z-common-inputs'
TRUSTED = Path('/root/autodl-tmp/mpm-lite-research-d/common-inputs/20260930T054100Z-common-inputs')
DTS = [.0005, .00025, .000125, .0000625]


def relative(a, b, floor=1e-30):
    return float(la.norm(a-b)/max(la.norm(b), floor))


def load():
    return load_frozen_inputs(BUNDLE, ROOT, trusted_data_root=TRUSTED, expected_sha256=PARENT)


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


def protocol():
    return dict(schema_version=1, parent_bundle_sha256=PARENT, seed=20260930,
        dtype='float64', device='cpu', full_shape=[369, 3], free_shape=[219, 3],
        units=dict(length='m', time='s', mass='kg', stress='Pa', energy='J'),
        boundaries=dict(breakpoints_s=[0., .5, .6, 1.1, 1.6], peak_m=.005,
                        displacement='parent cosine', velocity='analytic derivative'),
        dt_s=DTS, end_s=1.6, steps=[3200, 6400, 12800, 25600],
        material_order=6, independent_material_check_order=7, mass_order=5,
        independent_mass_order=6, AVF_path_order=3, independent_path_order=5,
        tolerances=dict(residual_atol=1e-10, residual_rtol=1e-7, energy_ledger_J=1e-9,
            boundary=1e-9, finite_difference_gradient=2e-5, finite_difference_tangent=2e-4,
            modal_residual=1e-7, mass_symmetry=1e-10, full_mass_order_difference=1e-8,
            parent_mass_rtol=1e-10, parent_mass_atol=1e-12,
            material_dynamic_difference=.005, cycle_relative=.02,
            near_zero_reaction_N=1e-5, near_zero_PK1_Pa=1e-4),
        finite_difference_steps=[1e-4, 3e-5, 1e-5],
        rank_threshold='max(64*n*eps*lambda_max,10*norm(Mscaled5-Mscaled6,2))',
        short_window=dict(initial_end_s=.0005, same_initial_bytes=True,
            preflight_times=[.25, .5, .85, 1.2], preflight_continuous_cycle=False,
            terminal_windows_certified=False),
        resources=dict(max_wall_hours=2., max_cpu_hours=8., threads=2,
            max_memory_GiB=8., minimum_data_free_GiB=5., safety_factor=1.5,
            checkpoint_every_accepted_steps=20, output_every_step=True,
            timing_exclusive_machine=False),
        scopes=dict(no_transfer=True, no_damping=True, no_mass_lumping=True,
                    no_space_reduction=True, spatial_certification=False,
                    cuda=False, B_E_external_atomicity=False))


def prepare(out, cache=None):
    out.mkdir(parents=True, exist_ok=False)
    p = protocol(); write_json(out/'dynamic-protocol.json', p)
    write_json(out/'execution-source.json', source_manifest(ROOT))
    start = time.perf_counter()
    s, inertia, archived, audit = load()
    audit.update(actual_full_shape=[s.ndof, 3], actual_free_shape=list(s.q_shape),
        Ks_sha256=digest(s.Ks), direction_sha256=digest(s.A), material=dict(mu=10., lam=20., k_f=200.),
        mass=inertia.contract(), archived_initial_is_loaded=True)
    write_json(out/'baseline-check.json', audit)
    if cache:
        # Cache is a development convenience; independent action checks below
        # establish that its matrices belong to this actual frozen space.
        with np.load(cache, allow_pickle=False) as z:
            M5, M6, K = z['M5'], z['M6'], z['K']
    else:
        M5 = recover_mass(inertia, lambda i, n: log('mass5', i, n))
        M6 = recover_mass(PointInertia(s, order=6), lambda i, n: log('mass6', i, n))
        K = rest_stiffness(s, lambda i, j: log('stiffness', i, j))
    np.savez_compressed(out/'operators.npz', M5=M5, M6=M6, K=K)
    from .audits import certify
    results=certify(out,s,M5,M6,K,BUNDLE,p['seed'])
    ready=results['mass']['passed']
    write_json(out/'prepare-summary.json',dict(seconds=time.perf_counter()-start,
        peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        passed=ready,status='ready' if ready else 'blocked_singular_mass'))
    if not ready:
        for name in ('constraint-impulse-audit','step-ledger','transaction-step-audit',
                     'throughput-preflight','short-window-screen','endpoint-attribution'):
            write_json(out/(name+'.json'),dict(status='not_run',passed=False,
                reason='C3 free scalar mass rank 216/219; dynamic gate closed',
                reduced_or_regularized_model_used=False))
        write_json(out/'cycles-summary.json',dict(status='not_run',reason='blocked_singular_mass',
            accepted_steps=[0,0,0,0],expected_steps=[3200,6400,12800,25600],
            end_s=1.6,dynamic_cycle=False,finite_level_time_comparison=False))
    log('free mass rank',results['mass']['free_scalar_rank'], '/', s.q_shape[0],
        'dynamic admission',ready)


def open_model(out, *, order=6, hold=None):
    s, _, _, _ = load()
    with np.load(out/'operators.npz') as z:
        return DynamicModel(s, z['M5'], order=order, rest_K=z['K'], boundary=Boundary(s, hold=hold))


def derivatives(out):
    from engine.aniso_phase1.research_c.stage2.model import FrozenPotential
    s, _, _, _ = load(); model = FrozenPotential(s)
    rng = np.random.default_rng(20260930)
    d = s.direction_coefficients(rng.normal(size=s.q_shape)); d /= la.norm(d)
    e = s.direction_coefficients(rng.normal(size=s.q_shape)); e /= la.norm(e)
    rows = []
    for name, q in [('zero', np.zeros((s.ndof, 3))), ('archived_loaded', s.expand(s.q0))]:
        base = model.evaluate(q, d); other = model.evaluate(q, e)
        symmetry = abs(np.sum(d*other['tangent_action'])-np.sum(e*base['tangent_action']))
        symmetry /= max(abs(np.sum(d*other['tangent_action'])), 1e-15)
        scans=[]
        for h in protocol()['finite_difference_steps']:
            plus=model.evaluate(q+h*d); minus=model.evaluate(q-h*d)
            expected=float(np.sum(base['force']*d))
            grad=(plus['U']-minus['U'])/(2*h)
            scans.append(dict(h=h, gradient_absolute=abs(grad-expected),
                gradient_relative=abs(grad-expected)/max(abs(expected), 1e-6),
                tangent_relative=relative((plus['force']-minus['force'])/(2*h),base['tangent_action'])))
        rows.append(dict(state=name, scans=scans, symmetry_relative=float(symmetry),
            stabilization_J=base['stabilization_U'], stabilization_force_norm=float(la.norm(base['force']-base['material_force']))))
        log('derivatives', name, scans)
    passed=all(r['symmetry_relative']<1e-7 and
        sum(x['tangent_relative']<2e-4 and (x['gradient_relative']<2e-5 or x['gradient_absolute']<1e-8)
            for x in r['scans'])>=2 for r in rows)
    write_json(out/'potential-derivative-audit.json',dict(passed=passed, states=rows,
        gradient_convention='positive potential gradient', original_Ks_retained=True))


def trajectory(out, model, state, dt, steps, name, **options):
    solver = AVF(model, state, **options)
    rows=[]; states=[]
    start=time.perf_counter()
    for i in range(steps):
        row=solver.step(dt); rows.append(row); states.append(solver.state)
        log(name, i+1, '/', steps, 'residual', row['true_residual'], 'seconds', row['wall_seconds'])
    write_json(out/f'{name}-ledger.json', dict(rows=rows, seconds=time.perf_counter()-start))
    np.savez_compressed(out/f'{name}-states.npz',q=np.stack([state.q]+[v.q for v in states]),
        velocity=np.stack([state.velocity]+[v.velocity for v in states]),
        time=np.array([state.time]+[v.time for v in states]))
    save_checkpoint(out/f'{name}-checkpoint.json', solver, dt, source_manifest(ROOT), PARENT)
    return solver, rows


def short(out):
    model=open_model(out); initial=model.rest()
    with np.load(out/'initial-dynamic-state.npz') as z:
        np.testing.assert_array_equal(initial.q,z['q']); np.testing.assert_array_equal(initial.velocity,z['velocity'])
    points=[model.space.test_vectors[f'points{k}'] for k in range(3)]
    solvers=[]; costs=[]; allrows=[]
    for dt in DTS:
        solver,rows=trajectory(out,model,initial,dt,round(.0005/dt),f'initial-{dt:.8f}')
        solvers.append(solver); costs += [r['wall_seconds'] for r in rows]; allrows += rows
    comparisons=[]
    for a,b in zip(solvers[:-1],solvers[1:]):
        fa=model.fields(a.state,points); fb=model.fields(b.state,points)
        comparisons.append(dict(PK1_absolute_Pa=float(la.norm(fa['PK1']-fb['PK1'])/np.sqrt(fa['PK1'].size)),
            PK1_relative=relative(fa['PK1'],fb['PK1']), q_absolute=float(la.norm(a.state.q-b.state.q)),
            velocity_absolute=float(la.norm(a.state.velocity-b.state.velocity)), physical_time=.0005))
    write_json(out/'short-window-screen.json',dict(passed=all(r['min_detF']>0 and r['accepted'] for r in allrows),
        numerical_scope='initial contiguous window only; no full cycle conclusion',
        comparisons= comparisons, end_s=.0005, steps=[1,2,4,8], initial_sha256=sha(out/'initial-dynamic-state.npz')))
    # Same initial state, one variable per branch.
    final=solvers[0].state
    baseline=allrows[0]
    variants=[]
    for name,order,path,atol,rtol in [('tighter',6,3,1e-12,1e-9),('path5',6,5,1e-10,1e-7),('material7',7,3,1e-10,1e-7)]:
        m=model if order==6 else open_model(out,order=order)
        state=m.rest()
        v,rows=trajectory(out,m,state,DTS[0],1,name,path_order=path,residual_atol=atol,residual_rtol=rtol)
        fields=m.fields(v.state,points); reference=model.fields(final,points)
        variants.append(dict(factor=name, q_absolute=float(la.norm(v.state.q-final.q)),
            velocity_absolute=float(la.norm(v.state.velocity-final.velocity)),
            reaction_absolute_N=abs(rows[0]['reaction_N']-baseline['reaction_N']),
            PK1_absolute_Pa=float(np.sqrt(np.mean((fields['PK1']-reference['PK1'])**2))),
            PK1_relative=relative(fields['PK1'],reference['PK1'])))
    full6=model.evaluate(final.q); full7=open_model(out,order=7).evaluate(final.q)
    material=dict(force_relative=relative(full6['force'],full7['force']),
        weak_PK1_relative=relative(full6['weak_moments'],full7['weak_moments']),
        energy_absolute_J=abs(full6['U']-full7['U']),
        passed=relative(full6['force'],full7['force'])<.005)
    write_json(out/'dynamic-material-audit.json',material)
    write_json(out/'endpoint-attribution.json',dict(variants=variants,
        scope='first 500 microseconds from common initial state', terminal_attribution='not_run',
        asymptotic_convergence_certified=False))
    # Representative admissible states are cost samples, never claimed to be
    # checkpoints from a continuously advanced 1.6 s trajectory.
    preflight=[]
    for t,name in zip([.25,.5,.85,1.2],['loading','holding','unloading','endholding']):
        state=model.rest(); state.time=t
        factor=float(np.max(model.boundary.lift(t))/.005)
        state.q=model.space.expand(factor*model.space.q0,model.boundary.lift(t))
        state.velocity=model.boundary.speed(t)
        if t>1.1:
            state.q[model.free]=1e-4*model.space.q0
        solver,rows=trajectory(out,model,state,.0005,2,'preflight-'+name)
        costs += [r['wall_seconds'] for r in rows]
        preflight.append(dict(phase=name, initial_time=t, accepted_steps=2,
            source='scaled archived displacement, synthetic admissible initial velocity', rows=rows))
    estimate=48000*max(costs)*1.5
    resource_record=dict(accepted_steps=len(allrows)+8, representative_phases=preflight,
        max_step_seconds=max(costs), minimum_step_seconds=min(costs),
        conservative_full_cycle_wall_hours=estimate/3600,
        conservative_cpu_hours=estimate*2/3600, safety_factor=1.5,
        allowed=protocol()['resources'], data_free_bytes=shutil.disk_usage(TRUSTED).free,
        system_free_bytes=shutil.disk_usage('/').free,
        peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        four_cycle_state_bytes=48000*2*369*3*8,
        status='blocked_resource' if estimate>2*3600 else 'pending_correctness_gates',
        timing_includes='full path material, nonlinear solve, final material, ledger; constructor recorded separately',
        machine_exclusive=False)
    write_json(out/'throughput-preflight.json',resource_record)
    write_json(out/'cycles-summary.json',dict(status='not_run', reason=resource_record['status'],
        accepted_steps=[0,0,0,0], expected_steps=[3200,6400,12800,25600],
        end_s=1.6, dynamic_cycle=False, finite_level_time_comparison=False))
    log('preflight estimated four-cycle hours', estimate/3600)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','derivatives','short'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--operator-cache',type=Path)
    args=parser.parse_args()
    if args.phase=='prepare':
        prepare(args.out,args.operator_cache)
        if not json.loads((args.out/'prepare-summary.json').read_text())['passed']:
            raise SystemExit(2)
    elif args.phase=='derivatives': derivatives(args.out)
    else: short(args.out)


if __name__=='__main__': main()
