"""Immutable positive rules on non-overlapping reference-material partitions."""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.optimize import linprog
from ..joint_sampling import partition_material


def readonly(value, dtype=float):
    a = np.array(value, dtype=dtype, copy=True)
    # An immutable bytes owner also prevents setflags(write=True).
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def digest_arrays(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class MaterialRule:
    X: np.ndarray
    weight: np.ndarray
    A2: np.ndarray
    A4: np.ndarray
    partition: np.ndarray
    method: str = 'full'

    def __post_init__(self):
        for name in ('X', 'weight', 'A2', 'A4', 'partition'):
            object.__setattr__(self, name, readonly(getattr(self, name), int if name == 'partition' else float))
        n = len(self.weight)
        if n == 0 or self.X.shape != (n, 3) or self.A2.shape != (n, 3, 3) or self.A4.shape != (n, 9, 9) or self.partition.shape != (n,):
            raise ValueError('inconsistent material rule shapes')
        if not all(np.isfinite(a).all() for a in (self.X, self.weight, self.A2, self.A4)) or np.any(self.weight <= 0):
            raise ValueError('finite samples with positive reference weights required')
        if not np.allclose(self.A2, self.A2.transpose(0, 2, 1), atol=1e-10) or np.linalg.eigvalsh(self.A2).min() < -1e-10:
            raise ValueError('A2 must be symmetric positive semidefinite')
        if not np.allclose(np.trace(self.A2, axis1=1, axis2=2), 1., atol=1e-10):
            raise ValueError('unit reference directions required')
        if not np.allclose(self.A4, self.A4.transpose(0, 2, 1), atol=1e-10) or np.linalg.eigvalsh(self.A4).min() < -1e-10:
            raise ValueError('A4 must be symmetric positive semidefinite')
        if not np.allclose(self.A4 @ np.eye(3).ravel(), self.A2.reshape(n, 9), atol=1e-10):
            raise ValueError('inconsistent second/fourth moments')

    @property
    def signature(self):
        return digest_arrays(self.X, self.weight, self.A2, self.A4, self.partition)

    @property
    def nbytes(self):
        return sum(getattr(self, k).nbytes for k in ('X', 'weight', 'A2', 'A4', 'partition'))

    def save(self, path):
        np.savez_compressed(path, X=self.X, weight=self.weight, A2=self.A2,
                            A4=self.A4, partition=self.partition, method=self.method)

    @classmethod
    def from_directions(cls, X, weight, directions, partition=None):
        a = np.asarray(directions, dtype=float)
        if a.shape != (len(X), 3) or not np.isfinite(a).all() or np.any(np.linalg.norm(a, axis=1) == 0):
            raise ValueError('nonzero finite reference directions required')
        a = a / np.linalg.norm(a, axis=1)[:, None]
        A = np.einsum('pi,pj->pij', a, a)
        M = np.einsum('pi,pj->pij', A.reshape(-1, 9), A.reshape(-1, 9))
        return cls(X, weight, A, M, np.zeros(len(X), int) if partition is None else partition)


def compress(full, groups_per_partition, method='conditional', direction_jump=.35):
    """Geometry/direction-only grouping, preserving volume and first X moment.

    Explicit partitions MUST resolve material interfaces. Within each partition,
    large A2 jumps force a split even if that exceeds the requested budget.
    Representatives retain actual X/A2/A4 tuples; conditional groups use the
    weighted centroid and conditional moments (not the square of mean A2).
    """
    if int(groups_per_partition) != groups_per_partition or groups_per_partition < 1:
        raise ValueError('positive integer group budget required')
    if method not in ('conditional', 'representatives') or not np.isfinite(direction_jump) or direction_jump <= 0:
        raise ValueError('invalid grouping configuration')
    positions, volumes, seconds, fourths, labels = [], [], [], [], []
    for region in np.unique(full.partition):
        ids = np.flatnonzero(full.partition == region)
        X, A, w = full.X[ids], full.A2[ids], full.weight[ids]
        h = max(float(np.ptp(X, axis=0).max()), 1e-12)
        leaves = partition_material(X, A, w, h, groups_per_partition, True)
        queue = list(leaves)
        leaves = []
        while queue:
            leaf = queue.pop(0)
            mean = np.average(A[leaf], axis=0, weights=w[leaf])
            if len(leaf) > 1 and np.linalg.norm(A[leaf]-mean, axis=(1, 2)).max() > direction_jump:
                split = partition_material(X[leaf], A[leaf], w[leaf], h, 2, True)
                if len(split) > 1:
                    queue.extend(leaf[s] for s in split)
                    continue
            leaves.append(leaf)
        for leaf in leaves:
            selected = ids[leaf]
            wn = full.weight[selected] / full.weight[selected].sum()
            if method == 'conditional':
                positions.append(wn @ full.X[selected])
                volumes.append(full.weight[selected].sum())
                seconds.append(np.einsum('p,pij->ij', wn, full.A2[selected]))
                fourths.append(np.einsum('p,pij->ij', wn, full.A4[selected]))
                labels.append(region)
            else:
                # Positive cubature with exact affine spatial moments, <=4
                # actual sites per leaf. No material response enters the fit.
                center = wn @ full.X[selected]
                dx = (full.X[selected] - center) / h
                constraints = np.vstack((np.ones(len(leaf)), dx.T))
                fitted = linprog(np.sum(dx*dx, axis=1), A_eq=constraints,
                                  b_eq=[1., 0., 0., 0.], bounds=(0., None), method='highs')
                if not fitted.success or np.linalg.norm(constraints @ fitted.x-[1, 0, 0, 0]) > 1e-8:
                    local, ww = np.arange(len(leaf)), full.weight[selected]
                else:
                    local = np.flatnonzero(fitted.x > 0)
                    ww = fitted.x[local] * full.weight[selected].sum()
                positions.extend(full.X[selected[local]])
                volumes.extend(ww)
                seconds.extend(full.A2[selected[local]])
                fourths.extend(full.A4[selected[local]])
                labels.extend([region]*len(local))
    rule = MaterialRule(positions, volumes, seconds, fourths, labels, method)
    audit = moment_audit(full, rule)
    if audit['volume_absolute'] > 1e-10 * full.weight.sum() or audit['first_moment_absolute'] > 1e-10 * full.weight.sum():
        raise ValueError('positive grouping failed spatial moment closure')
    return rule


def moment_audit(full, candidate):
    if set(full.partition) != set(candidate.partition):
        raise ValueError('material partition missing or duplicated under another identity')
    errs = []
    for region in np.unique(full.partition):
        a, b = full.partition == region, candidate.partition == region
        errs.append((abs(full.weight[a].sum()-candidate.weight[b].sum()),
                     np.linalg.norm(full.weight[a] @ full.X[a]-candidate.weight[b] @ candidate.X[b])))
    return dict(volume_absolute=float(max(e[0] for e in errs)),
                first_moment_absolute=float(max(e[1] for e in errs)),
                min_weight=float(candidate.weight.min()), partitions=len(errs),
                full_samples=len(full.weight), samples=len(candidate.weight))


def local_fallback(full, candidate, partitions):
    """Replace only named material partitions; never overlay duplicate volume."""
    partitions = set(partitions)
    if not partitions.issubset(set(full.partition)):
        raise ValueError('unknown fallback partition')
    a = np.isin(full.partition, list(partitions))
    b = ~np.isin(candidate.partition, list(partitions))
    rule = MaterialRule(*(np.concatenate((getattr(full, k)[a], getattr(candidate, k)[b]))
                          for k in ('X', 'weight', 'A2', 'A4', 'partition')), method='local_fallback')
    moment_audit(full, rule)
    return rule
