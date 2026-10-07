"""Independent sparse NumPy/SciPy replay of all dynamic-space raw snapshots.

This checker reads saved arrays; it does not call the solver or its map builder.
It also audits unprojected reactions, so constrained residual projection cannot
hide an incorrect material force. Writes only to the refinement output.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/results/lite-aniso-mainline'
CORNERS = np.array(list(itertools.product((0, 1), repeat=3)))


def pk1(F, A, kf):
    U, s, Vt = np.linalg.svd(F)
    logs = np.log(s)
    principal = (20*logs + 20*logs.sum(axis=1)[:, None])/s
    FA = F @ A
    I4 = np.sum(F*FA, axis=(1, 2))
    return (U*principal[:, None, :]) @ Vt + 2*kf*(I4-1)[:, None, None]*FA


def interpolation(x, centers, dx):
    lookup = {tuple(c): i for i, c in enumerate(centers)}
    q = x/dx-.5
    base = np.floor(q).astype(int)
    f = q-base
    rows, cols, values = [], [], []
    for p in range(len(x)):
        for corner in CORNERS:
            w = float(np.prod(np.where(corner, f[p], 1-f[p])))
            if w:
                rows.append(p); cols.append(lookup[tuple(base[p]+corner)]); values.append(w)
    return sp.csr_matrix((values, (rows, cols)), shape=(len(x), len(centers)))


def replay(z, cfg, row):
    dx, dt, beta = 1/(cfg['grid']-1), cfg['dt'], cfg['flip_ratio']
    centers, nodes = z['coords'], z['grid_nodes']
    lookup = {tuple(n): i for i, n in enumerate(nodes)}
    ids = np.array([[lookup[tuple(c+o)] for o in CORNERS] for c in centers])
    rows = np.repeat(np.arange(len(centers)), 8)
    shape = len(centers), len(nodes)
    H = sp.csr_matrix((np.full(len(rows), .125), (rows, ids.ravel())), shape=shape)
    D = [sp.csr_matrix((np.tile((2*CORNERS[:, k]-1)/(4*dx), len(centers)),
                       (rows, ids.ravel())), shape=shape) for k in range(3)]
    S = interpolation(z['particle_x_before'], centers, dx)
    V = np.asarray(S.T @ z['particle_volume']).ravel()
    assert np.all(V > 0)
    W = sp.diags(1/V) @ S.T @ sp.diags(z['particle_volume'])
    raw, new = z['grid_velocity_raw'], z['grid_velocity_new']
    G = [S @ d for d in D]
    L = np.stack([g @ new for g in G], axis=2)
    Lraw = np.stack([g @ raw for g in G], axis=2)
    F = (np.eye(3)+dt*L) @ z['particle_F_before']
    C = L + beta*(z['particle_C_before']-Lraw)
    v = beta*(z['particle_velocity_before'] + S @ (H @ (new-raw))) + (1-beta)*(S @ (H @ new))
    x = z['particle_x_before'] + dt*(S @ (H @ new))
    Fc = (W @ F.reshape(len(F), 9)).reshape(-1, 3, 3)
    A = (W @ z['particle_A0'].reshape(len(F), 9)).reshape(-1, 3, 3)
    P = pk1(Fc, A, cfg['kf'])
    force = np.zeros_like(new)
    for k in range(3):
        B = W @ sum(G[j].multiply(z['particle_F_before'][:, j, k, None]) for j in range(3))
        force += B.T @ (V[:, None]*P[:, :, k])
    mass = H.T @ (S.T @ z['particle_mass'])
    inertia = mass[:, None]*(new-raw)/dt
    errors = {key: float(np.max(abs(value-z[target]))) for key, value, target in (
        ('F', F, 'particle_F_after'), ('C', C, 'particle_C_after'), ('L', L, 'particle_L_after'),
        ('v', v, 'particle_velocity_after'), ('x', x, 'particle_x_after'), ('Fc', Fc, 'center_F_committed'))}
    for side, mask in [('left', nodes[:, 0]*dx <= .25), ('right', nodes[:, 0]*dx >= .75)]:
        errors[side+'_elastic_force'] = abs(float(force[mask, 0].sum())-row[side+'_elastic_force'])
        errors[side+'_force'] = abs(float((force+inertia)[mask, 0].sum())-row[side+'_force'])
    assert max(errors.values()) < 1e-11, errors
    return errors


def check(out):
    records, trajectories = [], []
    for root in (BASE/'v9-dynamic-space', out):
        p = json.loads((root/'protocol.json').read_text())
        for name, cfg in p['configs'].items():
            folder = root/'cases'/name
            status = json.loads((folder/'status.json').read_text())
            rows = [json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
            assert status['run_completed'] and len(rows) == round(p['duration']/cfg['dt'])
            assert abs(rows[-1]['time']-p['duration']) < 1e-12
            assert abs(rows[-1]['displacement']-.005) < 1e-12
            with np.load(folder/'frames.npz') as z:
                assert len(z['time']) == 21
                np.testing.assert_array_equal(z['F'][0], np.broadcast_to(np.eye(3), z['F'][0].shape))
                assert np.min(np.linalg.det(z['F'])) > 0
            paths = sorted(folder.glob('audit-*.npz'))
            assert len(paths) == 4
            for path in paths:
                with np.load(path) as z:
                    errors = replay(z, cfg, rows[int(path.stem.split('-')[-1])-1])
                records.append(dict(case=name, snapshot=path.name, errors=errors))
            trajectories.append(dict(case=name, steps=len(rows), completed=True))
    result = dict(passed=len(records)==96 and len(trajectories)==24, trajectories=trajectories,
        snapshots=records, max_oracle_error=max(max(r['errors'].values()) for r in records),
        method='independent trilinear sparse NumPy/SciPy replay; saved particle updates and unprojected grip forces')
    (out/'artifact-check.json').write_text(json.dumps(result, indent=2)+'\n')
    print('verified', len(records), 'snapshots,', len(trajectories), 'trajectories; max error', result['max_oracle_error'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, default=BASE/'v9-dynamic-refined')
    check(parser.parse_args().output)
