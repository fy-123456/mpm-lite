"""Two-grid closed-loop audit of a material-carried Q1 reference prototype.

CPU-only by design; does not modify or time the production MPM solver.
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import eigh

from engine.aniso_phase1.consistent_transfer import MaterialQ1, bent_nodes, material_response
from engine.aniso_phase1.beam_reference import beam_matrices
from utils.resource_guard import inspect_storage, prepare_warp_cache


DATA_ROOT = '/mnt/0c18569c-b839-4255-bae0-6f48c9fc835b/yin/tmp'


def guard():
    prepare_warp_cache('/tmp/mpm-lite-warp-cache', DATA_ROOT)
    return asdict(inspect_storage(DATA_ROOT))


def run(grid, ppc, order, dt, steps, out):
    guard()
    start = time.perf_counter()
    m = MaterialQ1(grid, ppc, order)
    x = bent_nodes(m)
    xp, Fp = m.particles.N @ x, m.gradient(m.particles, x)
    rng = np.random.default_rng(123)
    audits = []
    modes = dict(translation=np.tile([.05, -.02, .01], (len(x), 1)),
                 stretch=(m.X-m.lo)*np.array([.08, -.02, 0]),
                 bending=(bent_nodes(m, .02)-m.X),
                 random=rng.normal(size=x.shape)*.03)
    _, _, free, _, _, matrices = beam_matrices(grid)
    eigen, vectors = eigh(matrices[1][free][:, free].toarray(), subset_by_index=(0, 2))
    for k in range(3):
        mode = np.zeros(x.size)
        mode[free] = vectors[:, k]
        mode = mode.reshape(-1, 3)
        modes[f'weak_{k}'] = .05*mode/np.linalg.norm(mode, axis=1).max()
    for name, v in modes.items():
        _, _, _, _, row = m.commit(x, xp, Fp, v, dt)
        # Translation has no dF; report its absolute error instead of a ratio to zero.
        if name == 'translation':
            row['increment_recovery_relative'] = None
        rebuilt = m.project(m.particles.N@v)
        row.update(mode=name, velocity_roundtrip_relative=float(np.linalg.norm(rebuilt-v)/np.linalg.norm(v)))
        audits.append(row)

    # Exact pose update excludes Euler time integration's artificial stretching.
    angle = np.pi/4
    Q = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    posed = (x-.5) @ Q.T+.5+np.array([2.35*m.h, .2*m.h, 0])
    posed_rebuilt, pose_info = m.recover(m.particles.N@posed, m.gradient(m.particles, posed))
    psi, P = material_response(m.gradient(m.quadrature, x), m.quadrature.A, m.params)
    psi1, P1 = material_response(m.gradient(m.quadrature, posed_rebuilt), m.quadrature.A, m.params)
    pose_info.update(energy_relative=float((m.quadrature.weight@psi1)/(m.quadrature.weight@psi)-1),
                     stress_relative=float(np.linalg.norm(P1-Q@P)/np.linalg.norm(P)),
                     note='carried material topology; not an Eulerian cell/block crossing test')

    # Unconstrained PIC projection: momentum conservation and contraction of K.
    vp_random = rng.normal(size=xp.shape)*.1
    projected = m.particles.N @ m.project(vp_random)
    mass = m.mass[:, None]
    transfer = dict(linear_momentum_error=float(np.linalg.norm((mass*(projected-vp_random)).sum(axis=0))),
                    angular_momentum_error=float(np.linalg.norm((mass*np.cross(xp, projected-vp_random)).sum(axis=0))),
                    kinetic_before=float(.5*np.sum(mass*vp_random**2)),
                    kinetic_after=float(.5*np.sum(mass*projected**2)))

    # Material face traces, including locations between nodes.
    yz = np.array(np.meshgrid(np.linspace(.4375, .5625, 7), np.linspace(.4375, .5625, 7))).reshape(2, -1).T
    face = m.sample(np.column_stack((np.full(len(yz), .25), yz)), np.ones(len(yz)))
    initial_face = face.N @ x
    U0 = m.elastic(x)[0]
    rows = [dict(step=0, time=0., elastic=U0, kinetic=0., mechanical=U0)]
    frames = [xp.copy()]
    v = np.zeros_like(xp)
    for step in range(1, steps+1):
        guard()
        xp, Fp, v, info = m.step(xp, Fp, v, dt)
        rebuilt, _ = m.recover(xp, Fp)
        info.update(step=step, time=step*dt,
                    clamp_position_error=float(np.max(np.abs(face.N@rebuilt-initial_face))))
        rows.append(info)
        frames.append(xp.copy())
    name = f'g{grid}-p{ppc}-q{order}-dt{dt:g}'
    np.savez_compressed(out/(name+'.npz'), reference=m.particles.X, positions=np.array(frames),
                        times=np.arange(steps+1)*dt, mechanical=[r['mechanical'] for r in rows])
    with (out/(name+'.csv')).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    checks = dict(positive_mass=bool(m.lumped.min()>0 and m.mass_eigenvalues[0]>0),
                  history=all(abs(r['history_energy_relative'])<1e-7 for r in audits),
                  resolved_modes=all(r['increment_recovery_relative'] is None or r['increment_recovery_relative']<1e-7 for r in audits),
                  objective_pose=abs(pose_info['energy_relative'])<1e-8 and pose_info['stress_relative']<1e-8,
                  clamp=max(r.get('clamp_position_error', 0.) for r in rows)<1e-10,
                  stable_release=all(b['mechanical']<=a['mechanical']+max(1e-12, U0*1e-5) for a,b in zip(rows, rows[1:])),
                  positive_J=min(r['min_det'] for r in rows[1:])>.5,
                  energy_accounting=max(abs(r['energy_budget_residual']) for r in rows[1:])<max(1e-14, U0*1e-8),
                  projection=transfer['linear_momentum_error']<1e-12 and transfer['angular_momentum_error']<1e-12 and transfer['kinetic_after']<=transfer['kinetic_before']+1e-12)
    result = dict(name=name, grid=grid, ppc_axis=ppc, quadrature_order=order, dt=dt, steps=steps,
                  nodes=len(m.X), particles=len(m.mass), material_samples=len(m.quadrature.X),
                  mass_min_eigenvalue=float(m.mass_eigenvalues[0]), mass_condition=float(m.mass_eigenvalues[-1]/m.mass_eigenvalues[0]),
                  min_lumped_mass=float(m.lumped.min()), total_mass=float(m.mass.sum()),
                  clamped_uniform_reference_stiffness_lowest=eigen.tolist(), modes=audits, pose=pose_info,
                  transfer=transfer, dynamics=rows, final_mechanical_relative=rows[-1]['mechanical']/U0-1,
                  checks=checks, passed=all(checks.values()), seconds=time.perf_counter()-start)
    (out/(name+'.json')).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(name, 'PASS' if result['passed'] else 'FAIL', checks, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grids', nargs='+', type=int, default=[17, 33])
    p.add_argument('--ppc', type=int, default=2)
    p.add_argument('--order', type=int, default=3)
    p.add_argument('--dt', type=float, default=.001)
    p.add_argument('--steps', type=int, default=6)
    p.add_argument('--out', type=Path, default=Path('docs/results/consistent-transfer/v1'))
    args = p.parse_args()
    if not np.isfinite(args.dt) or args.dt<=0 or args.steps<1:
        p.error('positive finite dt and steps required')
    storage = guard()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for grid in args.grids:
        results.append(run(grid, args.ppc, args.order, args.dt, args.steps, args.out))
        (args.out/'results.json').write_text(json.dumps(dict(scope='CPU material-carried Q1, compatible elastic histories, consistent-mass PIC, no production MPM promotion',
                                                          storage=storage, results=results), indent=2, allow_nan=False)+'\n')
    if not all(r['passed'] for r in results):
        raise SystemExit('one or more acceptance checks failed; inspect results')


if __name__ == '__main__':
    main()
