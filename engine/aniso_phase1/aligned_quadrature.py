"""Diagnostic quadrature on the physical beam, split at frozen MLS knots.

No production transfer, material mapping, or default boundary is changed.
"""
import itertools
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from .quadratic import polynomial
from .trace_probe import CORNERS, stiffness

LOW = np.array([.25, .4375, .4375])
HIGH = np.array([.75, .5625, .5625])


@dataclass(frozen=True)
class Rule:
    points: np.ndarray
    weights: np.ndarray
    cells: np.ndarray


def partitions(h, low=LOW, high=HIGH):
    if h <= 0 or np.any(np.asarray(high) <= low):
        raise ValueError('positive spacing and box dimensions required')
    axes = []
    for a, b in zip(low, high):
        knots = (np.arange(np.floor(a/h)-1, np.ceil(b/h)+1)+.5)*h
        axes.append(np.r_[a, knots[(knots > a) & (knots < b)], b])
    return axes


def box_rule(h, order=4, low=LOW, high=HIGH, orders=None):
    """Positive reference volumes; orders optionally sets each subbox order."""
    axes = partitions(h, low, high)
    boxes = list(itertools.product(*[range(len(a)-1) for a in axes]))
    if orders is not None and len(orders) != len(boxes):
        raise ValueError('one order per subbox required')
    points, weights, cells = [], [], []
    for cell, index in enumerate(boxes):
        n = order if orders is None else int(orders[cell])
        if n < 1: raise ValueError('positive quadrature order required')
        z, w = np.polynomial.legendre.leggauss(n)
        lo = np.array([a[i] for a, i in zip(axes, index)])
        hi = np.array([a[i+1] for a, i in zip(axes, index)])
        q = np.array(list(itertools.product(z, repeat=3)))
        points.append((hi+lo)/2 + q*(hi-lo)/2)
        weights.append(np.prod(np.array(list(itertools.product(w, repeat=3))), axis=1)*np.prod(hi-lo)/8)
        cells.append(np.full(n**3, cell, dtype=int))
    return Rule(np.concatenate(points), np.concatenate(weights), np.concatenate(cells))


def face_rule(h, x=.75, order=4):
    axes = partitions(h)[1:];z, w = np.polynomial.legendre.leggauss(order)
    points, weights = [], []
    for iy, iz in itertools.product(*[range(len(a)-1) for a in axes]):
        lo = np.array([axes[0][iy], axes[1][iz]])
        hi = np.array([axes[0][iy+1], axes[1][iz+1]])
        yz = (hi+lo)/2 + np.array(list(itertools.product(z, repeat=2)))*(hi-lo)/2
        points.append(np.column_stack([np.full(len(yz), x), yz]))
        weights.append(np.prod(np.array(list(itertools.product(w, repeat=2))), axis=1)*np.prod(hi-lo)/4)
    return np.concatenate(points), np.concatenate(weights)


def evaluate(blend, points):
    """Vectorized evaluation of the SAME BlendedMLS polynomial/partition maps.

    Group queries by partition box; patch fit stays frozen. Includes dw terms.
    """
    points = np.asarray(points);h = blend.h
    base = np.floor(points/h-.5).astype(int)
    keys, inverse = np.unique(base, axis=0, return_inverse=True)
    rows, cols, values = [], [], [[] for _ in range(4)]
    for group, key in enumerate(keys):
        idsq = np.flatnonzero(inverse == group);x = points[idsq];f = x/h-.5-key
        for corner in CORNERS:
            terms = np.where(corner, f, 1-f);w = terms.prod(1)
            dw = np.stack([(1 if corner[d] else -1)*np.prod(np.delete(terms, d, axis=1), axis=1)/h for d in range(3)], axis=1)
            if not np.any(w) and not np.any(dw): continue
            patch, fit = blend.patch(key+corner)
            P, D = polynomial((x-(key+corner+.5)*h)/h, True)
            N = P@fit;G = np.einsum('qdk,kn->qdn', D, fit)/h
            rows.append(np.repeat(idsq, len(patch)));cols.append(np.tile(patch, len(idsq)))
            values[0].append((w[:, None]*N).ravel())
            for d in range(3):values[d+1].append((w[:, None]*G[:, d]+dw[:, d, None]*N).ravel())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    matrices = [sparse.coo_matrix((np.concatenate(v), (rows, cols)), shape=(len(points), len(blend.nodes))).tocsr() for v in values]
    for m in matrices:m.eliminate_zeros()
    return matrices[0], matrices[1:]


def assemble(blend, rule, batch=512, guard=lambda: None):
    K = sparse.csr_matrix((3*len(blend.nodes),)*2);nnz = 0;max_width = 0
    for start in range(0, len(rule.weights), batch):
        guard();end = start+batch
        _, G = evaluate(blend, rule.points[start:end])
        M = np.zeros((len(rule.weights[start:end]), 9, 9));M[:, 0, 0] = 1
        K += stiffness(G, rule.weights[start:end], M)
        union = sum(abs(g) for g in G);width = np.diff(union.indptr)
        nnz += int(width.sum());max_width = max(max_width, int(width.max()))
    return K, dict(support_mean=nnz/len(rule.weights), support_max=max_width)


def adaptive_rule(blend, low_order, fields, fraction=.8, batch=512, guard=lambda: None):
    """Diagnostic high/low energy indicator, frozen before any solve.

    The high-rule construction/evaluation is charged to rule building. Fields
    are training probes; separate modes must verify the resulting rule.
    """
    from .beam_reference import reference_hessian
    H=reference_hessian().reshape(3,3,3,3);p=np.asarray(fields).reshape(-1,len(blend.nodes),3)
    energies=[]
    for order in (low_order,4):
        rule=box_rule(blend.h,order);E=np.zeros((rule.cells.max()+1,len(p)))
        for start in range(0,len(rule.weights),batch):
            guard();end=start+batch;_,G=evaluate(blend,rule.points[start:end])
            D=np.stack([g@p.transpose(1,0,2).reshape(len(blend.nodes),-1) for g in G],axis=-1)
            D=D.reshape(-1,len(p),3,3)
            density=.5*np.einsum('qpai,aibj,qpbj->qp',D,H,D)*rule.weights[start:end,None]
            np.add.at(E,rule.cells[start:end],density)
        energies.append(E)
    score=np.sum(abs(energies[1]-energies[0])/np.maximum(energies[1].sum(0),1e-25),axis=1)
    rank=np.argsort(-score);count=int(np.searchsorted(np.cumsum(score[rank]),fraction*score.sum())+1)
    orders=np.full(len(score),low_order);orders[rank[:count]]=4
    return box_rule(blend.h,orders=orders),dict(indicator_fraction=fraction,promoted_cells=count,
        subboxes=len(score),training_probes=len(p),indicator_sum=float(score.sum()),orders=orders.tolist())
