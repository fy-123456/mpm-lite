"""Moving Q1 grid storage with explicit, lossless v/C residual histories.

This is a joint grid-plus-residual representation, not a grid-only MPM force
solve. The generalized material solve remains Lagrangian. A residual cannot
be discarded: high-order modes and even C need not belong to the grid space.
"""
from dataclasses import dataclass
import itertools

import numpy as np


CORNERS = np.array(list(itertools.product((0, 1), repeat=3)))


@dataclass
class GridPacket:
    nodes: np.ndarray
    ids: np.ndarray
    weights: np.ndarray
    gradients: np.ndarray
    grid_velocity: np.ndarray
    residual_v: np.ndarray
    residual_C: np.ndarray
    cells: np.ndarray

    def decode(self):
        values = self.grid_velocity[self.ids]
        v = np.einsum('pa,pai->pi', self.weights, values)+self.residual_v
        C = np.einsum('paj,pai->pij', self.gradients, values)+self.residual_C
        return v, C


def encode(x, v, C, mass, h=.04, origin=(.003, .002, .001)):
    x, v, C, mass = map(np.asarray, (x, v, C, mass))
    if not np.isfinite(h) or h <= 0 or np.shape(origin) != (3,) or not np.isfinite(origin).all():
        raise ValueError('finite grid origin and positive spacing required')
    if x.shape != v.shape or C.shape != (len(x), 3, 3) or mass.shape != (len(x),):
        raise ValueError('grid state shape mismatch')
    if not all(np.isfinite(a).all() for a in (x, v, C, mass)) or np.any(mass <= 0):
        raise ValueError('finite state and strictly positive masses required')
    scaled = (x-origin)/h; cells = np.floor(scaled).astype(np.int64); f = scaled-cells
    nodes, inv = np.unique((cells[:, None]+CORNERS).reshape(-1, 3), axis=0, return_inverse=True)
    ids = inv.reshape(len(x), 8)
    factors = np.where(CORNERS[None], f[:, None], 1-f[:, None])
    weights = np.prod(factors, axis=2)
    gradients = np.stack([(2*CORNERS[:, j]-1)[None]/h *
                          np.prod(factors[:, :, [k for k in range(3) if k != j]], axis=2)
                          for j in range(3)], axis=2)
    dx = nodes[ids]*h+origin-x[:, None]
    affine = v[:, None]+np.einsum('pij,paj->pai', C, dx)
    nodal_mass = np.zeros(len(nodes)); momentum = np.zeros((len(nodes), 3))
    wm = mass[:, None]*weights
    np.add.at(nodal_mass, ids.ravel(), wm.ravel())
    np.add.at(momentum, ids.ravel(), (wm[:, :, None]*affine).reshape(-1, 3))
    grid = np.zeros_like(momentum)
    np.divide(momentum, nodal_mass[:, None], out=grid, where=nodal_mass[:, None] > 0)
    gv = np.einsum('pa,pai->pi', weights, grid[ids])
    gC = np.einsum('paj,pai->pij', gradients, grid[ids])
    return GridPacket(nodes, ids, weights, gradients, grid, v-gv, C-gC, cells)


def roundtrip(model, state, previous_cells=None, h=.04, origin=(.003, .002, .001)):
    kin = model.space.kinematics(model.X, state.q, state.velocity)
    packet = encode(kin['x'], kin['v'], kin['C'], model.mass, h, origin)
    v, C = packet.decode()
    # This reconstruction is used for the committed velocity. The residual is
    # part of the decoded physical field, not a bypass via old coefficients.
    import scipy.linalg as la
    velocity = la.solve(model.M, model.N.T@(model.mass[:, None]*v), assume_a='pos')
    err = max(float(np.max(abs(v-kin['v']))), float(np.max(abs(C-kin['C']))),
              float(np.max(abs(velocity-state.velocity))))
    if err > 1e-9:
        raise ValueError('joint grid representation lost a local mode')
    loss = model.kinetic(velocity)-model.kinetic(state.velocity)
    return velocity, packet, dict(transfer_error=err, transfer_energy_jump_J=loss,
        residual_velocity_norm=float(np.linalg.norm(packet.residual_v)),
        residual_affine_norm=float(np.linalg.norm(packet.residual_C)),
        grid_nodes=len(packet.nodes), crossed_particles=0 if previous_cells is None else
        int(np.count_nonzero(np.any(packet.cells != previous_cells, axis=1))))
