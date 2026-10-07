"""Independent quadratic reference to check F45 gradient resolution efficiently.

Q2 uses the same hard grips/material/body, exact tensor-product Gauss integration,
and no stabilization or mass. Polynomial degree is changed, not the boundary.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time
import unittest
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator
from engine.aniso_phase1.boundary_reference import LO, HI, tensor_matrix, gradient as q1_gradient
from benchmarks.aniso_boundary_reference import hessian, write

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'docs/results/lite-aniso-mainline/v10-reference-q2'
TRIPLES = np.array(list(itertools.product(range(3), repeat=3)))


def basis(q):
    q = np.asarray(q)
    return np.stack((2*q*q-3*q+1, 4*q-4*q*q, 2*q*q-q), axis=-1), np.stack((4*q-3, 4-8*q, 4*q-1), axis=-1)


def axis(lo, hi, cells_per_unit, order=3):
    h = 1/cells_per_unit; count = round((hi-lo)/h)
    z, w = np.polynomial.legendre.leggauss(order); q, w = (z+1)/2, w/2
    N, D = basis(q)
    local = (h*(N.T @ (w[:, None]*N)), D.T @ (w[:, None]*D)/h, D.T @ (w[:, None]*N))
    ids = 2*np.arange(count)[:, None]+np.arange(3)
    rows, cols = np.repeat(ids, 3, axis=1).ravel(), np.tile(ids, (1, 3)).ravel()
    return tuple(sp.csr_matrix((np.tile(a.ravel(), count), (rows, cols)), shape=(2*count+1,)*2) for a in local)


def assemble(cells_per_unit, H):
    if cells_per_unit % 16: raise ValueError('fixed hard grips must align with cells')
    axes = [axis(a, b, cells_per_unit) for a, b in zip(LO, HI)]
    counts = np.rint((HI-LO)*cells_per_unit*2).astype(int)
    nodes = LO+np.array(list(itertools.product(*[range(n+1) for n in counts])))/(2*cells_per_unit)
    gram = {}
    for i in range(3):
        for j in range(3):
            parts = [a[0] for a in axes]
            if i == j: parts[i]=axes[i][1]
            else: parts[i], parts[j]=axes[i][2], axes[j][2].T
            gram[i, j] = tensor_matrix(parts)
    blocks = [[sum(H[3*a+i, 3*b+j]*gram[i, j] for i in range(3) for j in range(3)
                   if H[3*a+i, 3*b+j] != 0) for b in range(3)] for a in range(3)]
    return nodes, sp.bmat(blocks, format='csr')


def gradient(X, cells_per_unit, u):
    n = cells_per_unit; counts = np.rint((HI-LO)*n).astype(int)
    q = (X-LO)*n; cell = np.minimum(np.floor(q).astype(int), counts-1); q -= cell
    ids = np.ravel_multi_index((2*cell[:, None, :]+TRIPLES).reshape(-1, 3).T, tuple(2*counts+1)).reshape(-1, 27)
    N, D = basis(q); value = u[ids]; out = np.empty((len(X), 3, 3))
    for k in range(3):
        weights = np.ones((len(X), 27))*n
        for j in range(3): weights *= (D if j == k else N)[:, j, TRIPLES[:, j]]
        out[:, :, k] = np.einsum('pna,pn->pa', value, weights)
    return out


def solve(n):
    nodes, K = assemble(n, hessian('F45')); count = len(nodes)
    fixed = (nodes[:, 0] <= .25+1e-12) | (nodes[:, 0] >= .75-1e-12)
    free = np.flatnonzero(np.tile(~fixed, 3)); u = np.zeros(3*count); u[:count][nodes[:, 0] >= .75-1e-12] = .005
    rhs = -(K @ u)[free]; A = K[free][:, free].tocsr(); d = 1/A.diagonal(); it = 0
    def callback(_):
        nonlocal it
        it += 1
    y, info = cg(A, rhs, M=LinearOperator(A.shape, matvec=lambda x: d*x), rtol=2e-11, atol=1e-14, maxiter=20000, callback=callback)
    u[free] = y; f = K @ u
    residual = float(np.linalg.norm(f[free])/np.linalg.norm(rhs))
    r = dict(family='Q2', cells_per_unit=n, node_grid_equivalent=2*n+1, nodes=count,
        free_dofs=len(free), iterations=it, linear_info=int(info), relative_residual=residual,
        passed=bool(info == 0 and residual < 1e-8), reaction_N=float(f[:count][nodes[:, 0] >= .75-1e-12].sum()),
        energy_J=float(.5*u @ f), mass_included=False, stiffness_shift=0.)
    return nodes, u.reshape(3, count).T, r


def compare(a, b):
    na, ua, ra = a; nb, ub, rb = b
    divisions = int(np.lcm(na, nb)); h = 1/divisions
    counts = np.rint((HI-LO)*divisions).astype(int)
    cells = np.array(list(itertools.product(*[range(n) for n in counts])))
    z, w = np.polynomial.legendre.leggauss(3); z, w = (z+1)/2, w/2
    q = np.array(list(itertools.product(z, repeat=3))); V = h**3*np.prod(np.array(list(itertools.product(w, repeat=3))), axis=1)
    totals = {k: np.zeros(4) for k in ('whole', 'near_grip', 'interior', 'deep_interior')}
    for start in range(0, len(cells), 1024):
        c = cells[start:start+1024]; X = LO+h*(c[:, None, :]+q).reshape(-1, 3); weights = np.tile(V, len(c))
        L0 = q1_gradient(X, na+1, ua) if ra['family']=='Q1' else gradient(X, na, ua)
        L1 = gradient(X, nb, ub); H=hessian('F45')
        P0, P1 = (L0.reshape(-1, 9) @ H.T).reshape(-1, 3, 3), (L1.reshape(-1, 9) @ H.T).reshape(-1, 3, 3)
        norm = lambda x: np.sum(x*x, axis=(1, 2))
        values = np.column_stack((norm(L0-L1), norm(L1), norm(P0-P1), norm(P1)))*weights[:, None]
        dist = np.minimum(abs(X[:, 0]-.25), abs(X[:, 0]-.75))
        masks = dict(whole=np.ones(len(X), bool), near_grip=dist < .0625,
            interior=(X[:, 0]>.3125)&(X[:, 0]<.6875), deep_interior=(X[:, 0]>.375)&(X[:, 0]<.625))
        for k, mask in masks.items(): totals[k] += values[mask].sum(axis=0)
    regions = {k: dict(F_relative=float(np.sqrt(t[0]/t[1])), P_relative=float(np.sqrt(t[2]/t[3]))) for k, t in totals.items()}
    reaction = abs(ra['reaction_N']-rb['reaction_N'])/abs(rb['reaction_N'])
    return dict(families=[ra['family'], rb['family']], cells_per_unit=[na, nb], reaction_relative=reaction,
        regions=regions, near_grip_share=float(totals['near_grip'][2]/totals['whole'][2]),
        reaction_passed=reaction <= .01, interior_passed=max(regions['interior'].values()) <= .02,
        global_passed=max(regions['whole'].values()) <= .02)


class Q2Tests(unittest.TestCase):
    def test_exact_quadrature_and_partition(self):
        for a, b in zip(axis(.125, .875, 16, 3), axis(.125, .875, 16, 4)):
            self.assertLess(np.max(abs((a-b).data)), 1e-12)
        N, D = basis(np.array([.123, .456, .789]))
        np.testing.assert_allclose(N.sum(axis=1), 1., atol=1e-14)
        np.testing.assert_allclose(D.sum(axis=1), 0., atol=1e-14)

    def test_affine_and_quadratic_exactness_rotation(self):
        nodes, K = assemble(16, hessian('F45'))
        u = np.column_stack((nodes[:, 0]**2, nodes[:, 1]*nodes[:, 2], nodes[:, 2]**2))
        X = np.random.default_rng(99).uniform(LO, HI, size=(40, 3))
        exact = np.zeros((40, 3, 3)); exact[:, 0, 0] = 2*X[:, 0]; exact[:, 1, 1] = X[:, 2]
        exact[:, 1, 2] = X[:, 1]; exact[:, 2, 2] = 2*X[:, 2]
        np.testing.assert_allclose(gradient(X, 16, u), exact, atol=1e-13)
        W = np.array([[0., .03, .01], [-.03, 0., -.02], [-.01, .02, 0.]])
        self.assertLess(np.linalg.norm(K @ (nodes @ W.T).T.ravel()), 1e-11)


def hashes():
    files = ['benchmarks/aniso_q2_reference.py', 'engine/aniso_phase1/boundary_reference.py',
             'engine/aniso_phase1/beam_reference.py', 'benchmarks/aniso_boundary_reference.py']
    return {f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('action', choices=('freeze', 'tests', 'run', 'analyze'))
    p.add_argument('--output', type=Path, default=DEFAULT); a=p.parse_args(); out=a.output; out.mkdir(parents=True, exist_ok=True)
    if a.action == 'freeze':
        if (out/'protocol.json').exists(): raise RuntimeError('preserve protocol')
        write(out/'protocol.json', dict(frozen_at=datetime.now(timezone.utc).isoformat(), source_sha256=hashes(),
            cells_per_unit=[16, 32, 64], case='F45', boundary='original hard grips', exact_quadrature_order=3,
            unchanged_gates=dict(reaction=.01, interior=.02, global_field=.02),
            reason='Q1 interior stress remains sensitive even away from clamps; independently check polynomial resolution'))
        return
    protocol=json.loads((out/'protocol.json').read_text()); assert protocol['source_sha256']==hashes()
    if a.action == 'tests':
        with (out/'tests.log').open('x') as stream:
            r=unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Q2Tests))
        write(out/'tests.json', dict(passed=r.wasSuccessful(), tests=r.testsRun)); raise SystemExit(0 if r.wasSuccessful() else 2)
    if a.action == 'run':
        assert json.loads((out/'tests.json').read_text())['passed']; dest=out/'cases'; dest.mkdir(exist_ok=False)
        rows=[]
        for n in protocol['cells_per_unit']:
            start=time.monotonic(); nodes,u,r=solve(n); r['seconds']=time.monotonic()-start
            write(dest/f'F45-q2-n{n}.json',r); np.savez_compressed(dest/f'F45-q2-n{n}.npz',nodes=nodes,u=u)
            rows.append(r); print(r,flush=True); assert r['passed']
        write(out/'runs.json',dict(completed=True,records=rows));return
    values=[]
    for n in protocol['cells_per_unit']:
        with np.load(out/f'cases/F45-q2-n{n}.npz') as z:u=z['u'].copy()
        values.append((n,u,json.loads((out/f'cases/F45-q2-n{n}.json').read_text())))
    pairs=[compare(a,b) for a,b in zip(values[:-1],values[1:])]
    root=ROOT/'docs/results/lite-aniso-mainline/v10-reference/cases'
    with np.load(root/'F45-hard-g129.npz') as z:u=z['u'].copy()
    r=json.loads((root/'F45-hard-g129.json').read_text());r['family']='Q1'
    cross=compare((128,u,r),values[-1])
    write(out/'summary.json',dict(completed=True,pairs=pairs,cross_family=cross,
        full_reference_certified=pairs[-1]['reaction_passed'] and pairs[-1]['global_passed']))
    print(json.dumps(pairs,indent=2),flush=True)


if __name__ == '__main__': main()
