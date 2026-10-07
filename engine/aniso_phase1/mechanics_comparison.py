"""Same-domain static mechanics comparison, without mass or stabilization.

Q1 is the linearization of the material-carried prototype. MLS uses its full
blended gradient. Across spaces, a physical field is mapped separately into
each space; coefficient vectors of different sizes are never compared.
"""
import itertools
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import splu

from .aligned_quadrature import LOW, HIGH, evaluate
from .beam_reference import element_stiffness, reference_hessian
from .trace_probe import CORNERS, ConstrainedStatic, face_points


class Q1Static:
    """Full-integration Q1, excluding the unnecessary dynamic mass factor."""
    def __init__(self, grid):
        if grid < 17 or (grid-1) % 16:
            raise ValueError('use grid=16*k+1 to align the physical beam faces')
        self.grid = grid
        self.h = 1/(grid-1)
        self.counts = np.rint((HIGH-LOW)/self.h).astype(int)
        self.shape = tuple(self.counts+1)
        lattice = np.array(list(itertools.product(*[range(n) for n in self.shape])))
        self.nodes = LOW+self.h*lattice
        self.free = np.flatnonzero(np.repeat(lattice[:, 0]>0, 3))
        cells = np.array(list(itertools.product(*[range(n) for n in self.counts])))
        vertices = cells[:, None, :]+CORNERS
        ids = np.ravel_multi_index(vertices.reshape(-1, 3).T, self.shape).reshape(-1, 8)
        dofs = (3*ids[:, :, None]+np.arange(3)).reshape(-1, 24)
        ke = element_stiffness(self.h, reference_hessian(), full=True)
        rows = np.repeat(dofs, 24, axis=1).ravel()
        cols = np.tile(dofs, (1, 24)).ravel()
        self.K = sparse.coo_matrix((np.tile(ke.ravel(), len(cells)), (rows, cols)),
                                   shape=(3*len(self.nodes),)*2).tocsr()
        self.factor = splu(self.K[self.free][:, self.free].tocsc())

    def read(self, points, gradients=False):
        x = np.asarray(points)
        if np.any(x < LOW-1e-12) or np.any(x > HIGH+1e-12):
            raise ValueError('query outside physical material domain')
        local = (x-LOW)/self.h
        cell = np.maximum(0, np.minimum(np.floor(local).astype(int), self.counts-1))
        q = local-cell
        f = np.where(CORNERS[None] == 1, q[:, None], 1-q[:, None])
        vertices = cell[:, None]+CORNERS
        cols = np.ravel_multi_index(vertices.reshape(-1, 3).T, self.shape)
        rows = np.repeat(np.arange(len(x)), 8)
        shape = (len(x), len(self.nodes))
        N = sparse.coo_matrix((f.prod(2).ravel(), (rows, cols)), shape=shape).tocsr()
        if not gradients:
            return N
        D = []
        for d in range(3):
            val = (2*CORNERS[None, :, d]-1)*f[:, :, [j for j in range(3) if j!=d]].prod(2)/self.h
            D.append(sparse.coo_matrix((val.ravel(), (rows, cols)), shape=shape).tocsr())
        return N, D

    def solve(self, f):
        u = np.zeros_like(f)
        u[self.free] = self.factor.solve(f[self.free])
        return u, self.K@u-f

    def free_residual(self, residual):
        return residual[self.free]


class MLSStatic:
    def __init__(self, grid, blend, K):
        self.grid, self.h, self.blend, self.nodes, self.K = grid, blend.h, blend, blend.nodes, K
        C = evaluate(blend, face_points(grid, LOW[0], 6, True))[0]
        self.solver = ConstrainedStatic(K, C)

    def read(self, points, gradients=False):
        N, G = evaluate(self.blend, points)
        return (N, G) if gradients else N

    def solve(self, f):
        return self.solver.solve(f)

    def free_residual(self, residual):
        return self.solver.Z.T@residual


def load_and_trace(space, order=4):
    """Uniform traction, work-conjugate area-mean y displacement."""
    # Split at BOTH Q1 element faces and half-cell MLS partition knots.
    # A rule aligned only to MLS is not exact on the Q1 face trace.
    counts = np.rint((HIGH[1:]-LOW[1:])/(space.h/2)).astype(int)
    axes = [np.linspace(a, b, n+1) for a, b, n in zip(LOW[1:], HIGH[1:], counts)]
    z, w = np.polynomial.legendre.leggauss(order)
    q = np.array(list(itertools.product(z, repeat=2)))
    ww = np.prod(np.array(list(itertools.product(w, repeat=2))), axis=1)/4
    points, weights = [], []
    for cell in itertools.product(*[range(n) for n in counts]):
        lo = np.array([a[i] for a, i in zip(axes, cell)])
        hi = np.array([a[i+1] for a, i in zip(axes, cell)])
        yz = (lo+hi)/2+q*(hi-lo)/2
        points.append(np.column_stack((np.full(len(q), HIGH[0]), yz)))
        weights.append(ww*np.prod(hi-lo))
    points, weights = np.concatenate(points), np.concatenate(weights)
    mean = np.asarray(space.read(points).T@weights).ravel()/weights.sum()
    trace = np.zeros((len(space.nodes), 3))
    trace[:, 1] = mean
    return trace.ravel(), float(weights.sum())


def solve_metrics(space, total_force=-1e-4, mean_target=-5e-4):
    trace, area = load_and_trace(space)
    f = total_force*trace
    u, R = space.solve(f)
    delta = float(trace@u)
    U = .5*float(u@(space.K@u))
    forces = f.reshape(-1, 3)
    reactions = R.reshape(-1, 3)
    resultant = reactions.sum(0)
    moment = np.cross(space.nodes-LOW, reactions).sum(0)
    external_moment = np.cross(space.nodes-LOW, forces).sum(0)
    checks = space.read(face_points(space.grid, LOW[0], 7, True))@u.reshape(-1, 3)
    record = dict(tip_displacement=delta, strain_energy=U, effective_stiffness=total_force/delta,
                  target_mean_displacement=mean_target,
                  force_at_target_mean=total_force*mean_target/delta,
                  support_reaction_at_target_mean=float(resultant[1]*mean_target/delta),
                  support_reaction=resultant.tolist(), support_moment=moment.tolist(),
                  force_balance_relative=float(np.linalg.norm(resultant+forces.sum(0))/abs(total_force)),
                  moment_balance_relative=float(np.linalg.norm(moment+external_moment)/(abs(total_force)*.5)),
                  free_residual_relative=float(np.linalg.norm(space.free_residual(R))/np.linalg.norm(space.free_residual(f))),
                  clamp_max=float(np.max(np.abs(checks))),
                  work_identity_relative=float(abs(2*U-f@u)/abs(f@u)), face_area=area,
                  load_sum=forces.sum(0).tolist(), nodes=len(space.nodes), dofs=3*len(space.nodes))
    return u, record


MODES = ('stretch', 'shear', 'bend_y', 'bend_z', 'twist')


def physical_modes(points):
    """Clamped polynomial fields and their exact gradients, fixed amplitude."""
    x, y, z = (np.asarray(points)-[.25, .5, .5]).T
    p = np.zeros((len(x), 3, 5))
    G = np.zeros((len(x), 3, 3, 5))
    p[:, 0, 0] = x; G[:, 0, 0, 0] = 1
    p[:, 1, 1] = x; G[:, 1, 0, 1] = 1
    p[:, 0, 2] = -2*x*y; p[:, 1, 2] = x*x
    G[:, 0, 0, 2] = -2*y; G[:, 0, 1, 2] = -2*x; G[:, 1, 0, 2] = 2*x
    p[:, 0, 3] = -2*x*z; p[:, 2, 3] = x*x
    G[:, 0, 0, 3] = -2*z; G[:, 0, 2, 3] = -2*x; G[:, 2, 0, 3] = 2*x
    p[:, 1, 4] = -x*z; p[:, 2, 4] = x*y
    G[:, 1, 0, 4] = -z; G[:, 1, 2, 4] = -x; G[:, 2, 0, 4] = y; G[:, 2, 1, 4] = x
    return p, G


def common_rule(order=4):
    z, w = np.polynomial.legendre.leggauss(order)
    q = np.array(list(itertools.product(z, repeat=3)))
    weights = np.prod(np.array(list(itertools.product(w, repeat=3))), axis=1)*np.prod(HIGH-LOW)/8
    return (LOW+HIGH)/2+q*(HIGH-LOW)/2, weights


def mode_audit(space):
    points, weights = common_rule()
    exact, grad = physical_modes(points)
    H = reference_hessian().reshape(3, 3, 3, 3)
    gram_ref = np.einsum('q,qaim,aibj,qbjn->mn', weights, grad, H, grad)
    nodal, _ = physical_modes(space.nodes)
    P = nodal.reshape(-1, len(MODES))
    gram = P.T@(space.K@P)
    ratios = eigh((gram+gram.T)/2, gram_ref, eigvals_only=True)
    sampled = (space.read(points)@P.reshape(len(space.nodes), -1)).reshape(len(points), 3, -1)
    field_error = np.sqrt(np.einsum('q,qam,qam->m', weights, sampled-exact, sampled-exact) /
                          np.einsum('q,qam,qam->m', weights, exact, exact))
    clamp_points = face_points(space.grid, LOW[0], 7, True)
    clamp = space.read(clamp_points)@P.reshape(len(space.nodes), -1)
    return dict(modes={name:dict(stiffness_ratio=float(gram[i, i]/gram_ref[i, i]),
                                physical_field_relative=float(field_error[i])) for i, name in enumerate(MODES)},
                restricted_ratio_min=float(ratios[0]), restricted_ratio_max=float(ratios[-1]),
                clamp_max=float(np.max(np.abs(clamp))),
                scope='five common physical polynomial fields; NOT the entire displacement-space spectrum')


def field_errors(space, u, reference, reference_u, order=4):
    """Physical L2 and material-energy norm on a common fine partition.

    Split at both reference Q1 cells and candidate MLS knots. Candidate knots
    align with fine Q1 cells for the default 17/33 versus 65 reference.
    """
    axes = []
    for lo, hi in zip(LOW, HIGH):
        q1 = np.arange(round((hi-lo)/reference.h)+1)*reference.h+lo
        mls = (np.arange(np.floor(lo/space.h)-1, np.ceil(hi/space.h)+1)+.5)*space.h
        candidate = np.arange(round((hi-lo)/space.h)+1)*space.h+lo
        axes.append(np.unique(np.round(np.r_[q1, candidate, mls[(mls>lo)&(mls<hi)]], 14)))
    z, w = np.polynomial.legendre.leggauss(order)
    q = np.array(list(itertools.product(z, repeat=3)))
    ww = np.prod(np.array(list(itertools.product(w, repeat=3))), axis=1)/8
    H = reference_hessian().reshape(3, 3, 3, 3)
    totals = np.zeros(5)
    batch_points, batch_weights = [], []
    def accumulate(points, weights):
        N, G = space.read(points, True)
        Nr, Gr = reference.read(points, True)
        v = N@u.reshape(-1, 3); vr = Nr@reference_u.reshape(-1, 3)
        D = np.stack([g@u.reshape(-1, 3) for g in G], axis=2)
        Dr = np.stack([g@reference_u.reshape(-1, 3) for g in Gr], axis=2)
        e = D-Dr
        error_density = np.einsum('qai,aibj,qbj->q', e, H, e)
        return np.array([np.einsum('q,qi,qi->', weights, v-vr, v-vr),
                         np.einsum('q,qi,qi->', weights, vr, vr),
                         np.dot(weights, error_density),
                         np.einsum('q,qai,aibj,qbj->', weights, Dr, H, Dr),
                         np.dot(weights*(points[:, 0]<LOW[0]+.25*(HIGH[0]-LOW[0])), error_density)])
    for cell in itertools.product(*[range(len(a)-1) for a in axes]):
        lo = np.array([a[i] for a, i in zip(axes, cell)])
        hi = np.array([a[i+1] for a, i in zip(axes, cell)])
        batch_points.append((hi+lo)/2+q*(hi-lo)/2)
        batch_weights.append(ww*np.prod(hi-lo))
        if len(batch_points)*len(q)>=512:
            totals += accumulate(np.concatenate(batch_points), np.concatenate(batch_weights))
            batch_points.clear(); batch_weights.clear()
    if batch_points:
        totals += accumulate(np.concatenate(batch_points), np.concatenate(batch_weights))
    return dict(displacement_l2_relative=float(np.sqrt(totals[0]/totals[1])),
                strain_energy_norm_relative=float(np.sqrt(max(totals[2], 0)/totals[3])),
                root_quarter_strain_error_fraction=float(totals[4]/totals[2]) if totals[2]>1e-30 else 0.,
                comparison_quadrature_order=order)
