"""Frozen v10 reference study: hard grips and a labelled smooth-grip control."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import unittest
import numpy as np
from engine.aniso_phase1 import boundary_reference as ref
from engine.aniso_phase1.beam_reference import reference_hessian

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'docs/results/lite-aniso-mainline/v10-reference'
CASES = {'ISO': (0., 0.), 'F0': (200., 0.), 'F45': (200., 45.), 'F90': (200., 90.)}


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def hashes():
    names = ['engine/aniso_phase1/boundary_reference.py', 'engine/aniso_phase1/beam_reference.py',
             'benchmarks/aniso_boundary_reference.py']
    return {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in names}


def hessian(label):
    kf, angle = CASES[label]
    a = np.deg2rad(angle)
    return reference_hessian(kf=kf, direction=(np.cos(a), np.sin(a), 0.))


class ReferenceTests(unittest.TestCase):
    def test_independent_tensor_assembly_matches_quadrature(self):
        from benchmarks.aniso_static_space import rule, q1_maps, stiffness
        X, V = rule(17, 2, True)
        nodes, G = q1_maps(X, 17)
        for label in CASES:
            H = hessian(label)
            xn, K, _ = ref.assemble(17, H)
            np.testing.assert_array_equal(nodes, xn)
            expected = stiffness(G, V, H)
            self.assertLess(np.max(abs((K-expected).data)), 1e-12)
            self.assertLess(np.linalg.norm((K-K.T).data), 1e-12)

    def test_affine_rotation_and_work_identity(self):
        H = hessian('F45')
        nodes, K, _ = ref.assemble(17, H)
        skew = np.array([[0., .02, -.01], [-.02, 0., .03], [.01, -.03, 0.]])
        u = nodes @ skew.T+np.array([.003, -.001, .002])
        self.assertLess(np.linalg.norm(K @ u.T.ravel()), 1e-12)
        X, _ = next(ref.integration_chunks(16))
        np.testing.assert_allclose(ref.gradient(X, 17, u), np.broadcast_to(skew, (len(X), 3, 3)), atol=1e-14)
        for boundary in ('hard', 'smooth'):
            _, _, result = ref.solve(17, H, boundary)
            self.assertTrue(result['passed'], result)
            self.assertLess(result['work_identity_relative'], 1e-9)

    def test_old_reference_reproduced_and_region_partition(self):
        H = hessian('F45')
        _, u, r = ref.solve(33, H)
        old = ROOT/'docs/results/lite-aniso-mainline/v9-space/cases/reference-F45-g33.npz'
        with np.load(old) as z:
            np.testing.assert_allclose(u, z['u'], rtol=1e-9, atol=1e-12)
        result = ref.compare((33, u, r), (33, u, r), H)
        self.assertEqual(result['regions']['whole']['P_relative'], 0.)
        self.assertAlmostEqual(result['regions']['whole']['volume_m3'], .046875, places=13)


def freeze(out):
    if (out/'protocol.json').exists():
        raise RuntimeError('preserve protocol')
    runs = [(label, 'hard', g) for label in CASES for g in (33, 65, 129)]
    runs += [('F45', 'smooth', g) for g in (33, 65, 129)]
    write(out/'protocol.json', dict(frozen_at=datetime.now(timezone.utc).isoformat(), source_sha256=hashes(),
        runs=runs, body=[ref.LO.tolist(), ref.HI.tolist()], mu=10., lam=20., directions=CASES,
        displacement_m=.005, grids=[33, 65, 129], boundary_transition_m=.0625, spring_kappa_Pa_per_m2=1e6,
        regions=dict(near_grip='distance to x=.25/.75 < 1/16', interior='.3125 < x < .6875',
                     deep_interior='.375 < x < .625'),
        gates=dict(reaction=.01, F_and_P_interior=.02, F_and_P_global=.02, residual=1e-8),
        scope='small-strain static; exact same-law tangent; no mass, no time integration',
        smooth_control='different boundary model; sensitivity only, never substitutes for hard-grip convergence',
        maximum_grid=129, adaptive_extension='requires a separately frozen protocol'))


def tests(out):
    with (out/'tests.log').open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceTests))
    write(out/'tests.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
                               failures=len(result.failures), errors=len(result.errors)))
    return result.wasSuccessful()


def run(out, protocol):
    if not json.loads((out/'tests.json').read_text())['passed']:
        raise RuntimeError('reference implementation checks required')
    dest = out/'cases'; dest.mkdir(exist_ok=False)
    records = []
    for label, boundary, grid in protocol['runs']:
        name = f'{label}-{boundary}-g{grid}'
        start = time.monotonic()
        nodes, u, r = ref.solve(grid, hessian(label), boundary)
        r.update(case=label, seconds=time.monotonic()-start)
        write(dest/(name+'.json'), r)
        np.savez_compressed(dest/(name+'.npz'), nodes=nodes, u=u)
        records.append(r)
        print(name, 'R', r['reaction_N'], 'residual', r['relative_residual'], 'seconds', r['seconds'], flush=True)
        if not r['passed']:
            raise RuntimeError(name+' failed solver checks')
    write(out/'runs.json', dict(completed=True, records=records))
    return True


def analyze(out, protocol):
    data = json.loads((out/'runs.json').read_text())
    pairs = []
    for label, boundary in [(label, 'hard') for label in CASES]+[('F45', 'smooth')]:
        values = []
        for grid in protocol['grids']:
            name = f'{label}-{boundary}-g{grid}'
            with np.load(out/'cases'/(name+'.npz')) as z:
                u = z['u'].copy()
            values.append((grid, u, json.loads((out/'cases'/(name+'.json')).read_text())))
        for a, b in zip(values[:-1], values[1:]):
            result = ref.compare(a, b, hessian(label))
            result.update(case=label, boundary=boundary)
            pairs.append(result)
            print(label, boundary, result['grids'], 'R gap', result['reaction_relative'],
                  'P interior', result['regions']['interior']['P_relative'], flush=True)
    hard_last = [r for r in pairs if r['boundary']=='hard' and r['grids'][-1]==129]
    result = dict(completed=True, runs=len(data['records']), pairs=pairs,
        physical_solver_checks_passed=all(r['passed'] for r in data['records']),
        hard_reaction_gate_passed=all(r['reaction_passed'] for r in hard_last),
        hard_interior_gate_passed=all(r['interior_passed'] for r in hard_last),
        hard_global_gate_passed=all(r['global_passed'] for r in hard_last),
        full_reference_certified=all(r['reaction_passed'] and r['global_passed'] for r in hard_last))
    write(out/'summary.json', result)
    return True


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('freeze', 'tests', 'run', 'analyze'))
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    if args.action == 'freeze':
        freeze(out); return
    protocol = json.loads((out/'protocol.json').read_text())
    if protocol['source_sha256'] != hashes():
        raise RuntimeError('frozen reference source changed')
    ok = tests(out) if args.action == 'tests' else run(out, protocol) if args.action == 'run' else analyze(out, protocol)
    raise SystemExit(0 if ok else 2)


if __name__ == '__main__':
    main()
