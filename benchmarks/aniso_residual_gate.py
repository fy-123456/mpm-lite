"""Shared massless gate for v10 candidates; independent of the Warp matvec."""
import itertools
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, minres, spsolve
from benchmarks.aniso_static_space import lite_geometry, stiffness
from engine.aniso_phase1.beam_reference import element_stiffness
from engine.aniso_phase1.projected_history import frozen_maps

CORNERS = np.array(list(itertools.product((0, 1), repeat=3)))
MODES = ('projected_center', 'residual_center', 'residual_corotated')


def geometry(grid, beam=False):
    if not beam:
        g = lite_geometry(grid, 2)
    else:
        h = 1/(grid-1)
        lo, hi = np.array([.25, .4375, .4375]), np.array([.75, .5625, .5625])
        counts = np.rint((hi-lo)/h*2).astype(int)
        axes = [lo[k]+(np.arange(n)+.5)*h/2 for k, n in enumerate(counts)]
        x = np.array(list(itertools.product(*axes)))
        Vp = np.full(len(x), h**3/8)
        base = np.floor(x/h-.5).astype(int)
        centers = np.unique((base[:, None, :]+CORNERS).reshape(-1, 3), axis=0)
        nodes = np.unique((centers[:, None, :]+CORNERS).reshape(-1, 3), axis=0)
        maps, _, V, W, S, D = frozen_maps(x, Vp, np.tile(np.eye(3), (len(x), 1, 1)), centers, nodes, h)
        g = dict(grid=grid, nodes=nodes*h, centers=centers, points=x, volume=V, particle_volume=Vp,
                 D=D, maps={'center': D, 'projected_center': maps, 'particle': tuple(S @ d for d in D)},
                 weights={'center': V, 'projected_center': V, 'particle': Vp})
    x = g['nodes'][:, 0]
    fixed = x <= .25+1e-12 if beam else (x <= .25+1e-12) | (x >= .75-1e-12)
    g.update(fixed=fixed, free=np.flatnonzero(np.tile(~fixed, 3)), beam=beam)
    return g


def hourglass_matrix(g, H):
    h = 1/(g['grid']-1)
    # For uniform history at I, the exact objective stabilizer linearizes to
    # full Q1 minus center integration, without an arbitrary diagonal spring.
    local = element_stiffness(h, H, True)-element_stiffness(h, H, False)
    nodes = np.rint(g['nodes']/h).astype(int)
    lookup = {tuple(n): i for i, n in enumerate(nodes)}
    ids = np.array([[lookup[tuple(c+o)] for o in CORNERS] for c in g['centers']])
    dofs = (ids[:, :, None]+len(nodes)*np.arange(3)).reshape(len(ids), 24)
    rows = np.repeat(dofs, 24, axis=1).ravel()
    cols = np.tile(dofs, (1, 24)).ravel()
    values = (g['volume'][:, None]/h**3*local.ravel()).ravel()
    return sp.csr_matrix((values, (rows, cols)), shape=(3*len(nodes),)*2)


def matrix(g, H, mode):
    if mode == 'projected_center':
        return stiffness(g['maps']['projected_center'], g['volume'], H)
    local = stiffness(g['maps']['particle'], g['particle_volume'], H)
    if mode == 'residual_center':
        return local
    if mode == 'residual_corotated':
        return local+hourglass_matrix(g, H)
    raise ValueError(mode)


def rank_gate(g, K, full_spectrum=False):
    A = K[g['free']][:, g['free']].tocsr()
    scale = float(np.max(np.asarray(abs(A).sum(axis=1))))
    if full_spectrum or len(g['free']) <= 500:
        vals = np.linalg.eigvalsh(A.toarray())
        complete = True
    else:
        # Energy is PSD analytically; these are the smallest-magnitude values.
        vals = np.sort(eigsh(A, k=6, sigma=-1e-12*max(scale, 1.), which='LM', return_eigenvectors=False))
        complete = False
    threshold = 1e-9*max(scale, 1.)
    return dict(free_dofs=A.shape[0], full_spectrum=complete, min_eigenvalue=float(vals[0]),
                zero_or_soft_modes_observed=int(np.sum(vals <= threshold)), scale=scale,
                threshold=threshold, checked_eigenvalues=vals[:12].tolist(),
                passed=bool(vals[0] > threshold), mass_included=False, diagonal_shift_added=False)


def static_solve(g, K, nonsingular):
    n = len(g['nodes'])
    u = np.zeros((3, n))
    load = np.zeros(3*n)
    if g['beam']:
        tip = np.isclose(g['nodes'][:, 0], .75)
        load[n+np.flatnonzero(tip)] = -1e-4/tip.sum()
    else:
        u[0, g['nodes'][:, 0] >= .75-1e-12] = .005
    u = u.ravel()
    free = g['free']; A = K[free][:, free].tocsr(); b = (load-K @ u)[free]
    if nonsingular:
        y = spsolve(A, b); info = 0
    else:
        d = 1/np.sqrt(A.diagonal())
        y, info = minres(sp.diags(d) @ A @ sp.diags(d), d*b, rtol=1e-12, maxiter=30000)
        y = d*y
    u[free] = y
    force = K @ u
    residual = np.linalg.norm((force-load)[free])/max(np.linalg.norm(b), 1e-30)
    result = dict(relative_residual=float(residual), info=int(info), solved=bool(info == 0 and residual < 1e-8),
        energy_J=float(.5*u @ force), unique_static_solution=nonsingular)
    if g['beam']:
        result['tip_displacement_m'] = float(u.reshape(3, n)[1, tip].mean())
    else:
        result['reaction_N'] = float(force[:n][g['nodes'][:, 0] >= .75-1e-12].sum())
    return u.reshape(3, n).T, result
