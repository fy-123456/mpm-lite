"""Independent tensor-product Q1 reference with massless boundary diagnostics.

The tensor assembly does not use Lite gradients or the v9 quadrature assembler.
Hard grips reproduce the existing physical problem. Smooth volume springs are
a separately labelled boundary-sensitivity problem with fixed physical units.
"""
import itertools
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, spsolve, LinearOperator

LO = np.array([.125, .375, .375])
HI = np.array([.875, .625, .625])
CORNERS = np.array(list(itertools.product((0, 1), repeat=3)))


def axis_matrices(lo, hi, h, foundation=None):
    n = int(round((hi-lo)/h))
    rows = np.repeat(np.arange(n)[:, None]+[0, 1], 2, axis=1).ravel()
    cols = np.tile(np.arange(n)[:, None]+[0, 1], (1, 2)).ravel()
    blocks = (h*np.array([[2., 1.], [1., 2.]])/6,
              np.array([[1., -1.], [-1., 1.]])/h,
              np.array([[-1., -1.], [1., 1.]])/2)
    matrices = [sp.csr_matrix((np.tile(a.ravel(), n), (rows, cols)), shape=(n+1, n+1)) for a in blocks]
    if foundation is not None:
        z, w = np.polynomial.legendre.leggauss(4)
        q, w = (z+1)/2, w/2
        N = np.stack((1-q, q), axis=1)
        x = lo+h*(np.arange(n)[:, None]+q)
        values = h*np.einsum('eq,q,qi,qj->eij', foundation(x), w, N, N)
        matrices.append(sp.csr_matrix((values.ravel(), (rows, cols)), shape=(n+1, n+1)))
    return matrices


def spring_profile(x, side, width=.0625, kappa=1e6):
    t = (.25-x)/width if side == 'left' else (x-.75)/width
    t = np.clip(t, 0., 1.)
    return kappa*t**3*(10-15*t+6*t*t)


def tensor_matrix(factors):
    return sp.kron(sp.kron(factors[0], factors[1], format='csr'), factors[2], format='csr')


def assemble(grid, H):
    h = 1/(grid-1)
    if (grid-1) % 16:
        raise ValueError('reference grids must resolve the fixed 1/16 transition width')
    axes = [axis_matrices(lo, hi, h) for lo, hi in zip(LO, HI)]
    counts = np.rint((HI-LO)/h).astype(int)
    nodes = LO+h*np.array(list(itertools.product(*[range(n+1) for n in counts])))
    gram = {}
    for i in range(3):
        for j in range(3):
            factors = [a[0] for a in axes]
            if i == j:
                factors[i] = axes[i][1]
            else:
                factors[i], factors[j] = axes[i][2], axes[j][2].T
            gram[i, j] = tensor_matrix(factors)
    blocks = []
    for a in range(3):
        row = []
        for b in range(3):
            terms = [H[3*a+i, 3*b+j]*gram[i, j] for i in range(3) for j in range(3) if H[3*a+i, 3*b+j] != 0]
            row.append(sum(terms) if terms else sp.csr_matrix(gram[0, 0].shape))
        blocks.append(row)
    return nodes, sp.bmat(blocks, format='csr'), axes


def solve(grid, H, boundary='hard', direct_limit=40000):
    nodes, K, axes = assemble(grid, H)
    n = len(nodes)
    u = np.zeros(3*n)
    spring = None
    if boundary == 'hard':
        left, right = nodes[:, 0] <= .25+1e-12, nodes[:, 0] >= .75-1e-12
        free = np.flatnonzero(np.tile(~(left | right), 3))
        u[:n][right] = .005
        system, rhs = K, -(K @ u)
    elif boundary == 'smooth':
        h = 1/(grid-1)
        spring = []
        for side in ('left', 'right'):
            Mx = axis_matrices(LO[0], HI[0], h, lambda x: spring_profile(x, side))[-1]
            spring.append(tensor_matrix([Mx, axes[1][0], axes[2][0]]))
        Q = spring[0]+spring[1]
        system = K+sp.block_diag([Q]*3, format='csr')
        rhs = np.zeros(3*n)
        rhs[:n] = spring[1] @ np.full(n, .005)
        free = np.arange(3*n)
    else:
        raise ValueError('boundary must be hard or smooth')
    A = system[free][:, free].tocsr()
    b = rhs[free]
    iterations = 0
    if len(free) <= direct_limit:
        u[free] = spsolve(A, b)
        info, iterations, method = 0, 1, 'sparse_LU'
    else:
        inverse = 1/A.diagonal()
        M = LinearOperator(A.shape, matvec=lambda v: inverse*v)
        def count(_):
            nonlocal iterations
            iterations += 1
        u[free], info = cg(A, b, M=M, rtol=2e-11, atol=1e-14, maxiter=20000, callback=count)
        method = 'CG_Jacobi_no_mass_or_diagonal_stiffness'
    residual = np.linalg.norm(A @ u[free]-b)/max(np.linalg.norm(b), 1e-30)
    field = u.reshape(3, n).T
    elastic = float(.5*u @ (K @ u))
    if boundary == 'hard':
        reaction = float((K @ u)[:n][right].sum())
        spring_energy = 0.
    else:
        reaction = float(np.sum(spring[1] @ (.005-field[:, 0])))
        spring_energy = 0.
        for Q, target in zip(spring, (0., .005)):
            d = field.copy(); d[:, 0] -= target
            spring_energy += .5*sum(float(d[:, a] @ (Q @ d[:, a])) for a in range(3))
    record = dict(grid=grid, boundary=boundary, nodes=n, free_dofs=len(free), method=method,
        iterations=iterations, linear_info=int(info), relative_residual=float(residual),
        passed=bool(info == 0 and residual < 1e-8 and np.isfinite(u).all()),
        reaction_N=reaction, elastic_J=elastic, spring_J=spring_energy,
        total_energy_J=elastic+spring_energy,
        work_identity_relative=abs(elastic+spring_energy-.5*reaction*.005)/max(abs(elastic+spring_energy), 1e-30),
        mass_included=False, stiffness_shift=0., spring_width_m=.0625 if spring is not None else None,
        spring_kappa_Pa_per_m2=1e6 if spring is not None else None)
    return nodes, field, record


def gradient(points, grid, u):
    h = 1/(grid-1)
    counts = np.rint((HI-LO)/h).astype(int)
    q = (points-LO)/h
    cell = np.minimum(np.floor(q).astype(int), counts-1)
    if np.any(cell < 0) or np.any(q > counts+1e-10):
        raise ValueError('points outside material domain')
    f = q-cell
    ids = np.ravel_multi_index((cell[:, None, :]+CORNERS).reshape(-1, 3).T, tuple(counts+1)).reshape(-1, 8)
    factors = np.where(CORNERS[None, :, :], f[:, None, :], 1-f[:, None, :])
    result = np.empty((len(points), 3, 3))
    values = u[ids]
    for k in range(3):
        g = (2*CORNERS[None, :, k]-1)*factors[:, :, [j for j in range(3) if j != k]].prod(axis=2)/h
        result[:, :, k] = np.einsum('pna,pn->pa', values, g)
    return result


def integration_chunks(divisions, chunk_cells=8192):
    h = 1/divisions
    counts = np.rint((HI-LO)/h).astype(int)
    cells = np.array(list(itertools.product(*[range(n) for n in counts])))
    q = np.array(list(itertools.product((.5-1/np.sqrt(12), .5+1/np.sqrt(12)), repeat=3)))
    for start in range(0, len(cells), chunk_cells):
        yield LO+h*(cells[start:start+chunk_cells, None, :]+q).reshape(-1, 3), h**3/8


def compare(a, b, H):
    """Exact L2 integration on a common partition, with fixed physical zones."""
    ga, ua, ra = a
    gb, ub, rb = b
    divisions = int(np.lcm(ga-1, gb-1))
    totals = {name: np.zeros(8) for name in ('whole', 'near_grip', 'interior', 'deep_interior')}
    peaks = np.zeros(2)
    for X, weight in integration_chunks(divisions):
        L0, L1 = gradient(X, ga, ua), gradient(X, gb, ub)
        P0, P1 = (L0.reshape(-1, 9) @ H.T).reshape(-1, 3, 3), (L1.reshape(-1, 9) @ H.T).reshape(-1, 3, 3)
        distance = np.minimum(abs(X[:, 0]-.25), abs(X[:, 0]-.75))
        masks = dict(whole=np.ones(len(X), bool), near_grip=distance < .0625,
                     interior=(X[:, 0] > .3125) & (X[:, 0] < .6875),
                     deep_interior=(X[:, 0] > .375) & (X[:, 0] < .625))
        norm = lambda t: np.sum(t*t, axis=(1, 2))
        values = np.column_stack((norm(L0-L1), norm(L1), norm(P0-P1), norm(P1),
            np.einsum('pij,pij->p', L0-L1, P0-P1), np.einsum('pij,pij->p', L1, P1),
            norm(P0), np.ones(len(X))))
        for name, mask in masks.items():
            totals[name] += weight*values[mask].sum(axis=0)
        peaks = np.maximum(peaks, [np.sqrt(norm(P0)).max(), np.sqrt(norm(P1)).max()])
    regions = {}
    for name, t in totals.items():
        regions[name] = dict(F_relative=float(np.sqrt(t[0]/max(t[1], 1e-30))),
            P_relative=float(np.sqrt(t[2]/max(t[3], 1e-30))),
            energy_norm_relative=float(np.sqrt(max(t[4], 0)/max(t[5], 1e-30))),
            stress_difference_L2_squared=float(t[2]), stress_reference_L2_squared=float(t[3]), volume_m3=float(t[7]))
    reaction = abs(ra['reaction_N']-rb['reaction_N'])/abs(rb['reaction_N'])
    return dict(grids=[ga, gb], common_partition_divisions=divisions, reaction_relative=reaction,
        regions=regions, near_grip_share_of_stress_difference=float(totals['near_grip'][2]/max(totals['whole'][2], 1e-30)),
        sampled_peak_stress_Pa=peaks.tolist(),
        reaction_passed=reaction <= .01,
        interior_passed=max(regions['interior']['F_relative'], regions['interior']['P_relative']) <= .02,
        global_passed=max(regions['whole']['F_relative'], regions['whole']['P_relative']) <= .02)
