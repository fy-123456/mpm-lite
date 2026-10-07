"""Small material-carried Q1 consistency prototype (CPU, float64).

This is a Lagrangian reference, NOT a new Eulerian MPM transfer or APIC.
One reference basis supplies mass, forces, velocity, spatial gradients and
history recovery. Only compatible elastic histories F = dx/dX are accepted.
No MLS weights, production defaults, stabilization or plasticity are changed.
"""
from dataclasses import dataclass
import itertools
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigvalsh
from scipy.sparse.linalg import splu

from .types import AnisotropicMaterialParams


CORNERS = np.array(list(itertools.product((0, 1), repeat=3)))


def bent_nodes(model, amplitude=.005):
    """Compatible nonuniform bending geometry; the left face stays fixed."""
    x = model.X.copy()
    t = (x[:, 0]-.25)/.5
    x[:, 0] -= 4*amplitude*t*(x[:, 1]-.5)
    x[:, 1] += amplitude*t*t
    return x


@dataclass
class Samples:
    X: np.ndarray
    weight: np.ndarray
    N: sp.csr_matrix
    D: tuple  # reference derivative matrices, each (samples, nodes)
    A: np.ndarray


def material_response(F, A, params):
    """Batched energy and exact PK1 of the existing Hencky + fiber law."""
    if not np.isfinite(F).all() or np.any(np.linalg.det(F) <= 0):
        raise ValueError("non-positive or non-finite material Jacobian")
    U, sigma, Vt = np.linalg.svd(F)
    logs = np.log(sigma)
    tr = logs.sum(axis=1)
    psi = params.mu * (logs * logs).sum(axis=1) + .5 * params.lam * tr**2
    principal = (2 * params.mu * logs + params.lam * tr[:, None]) / sigma
    P = (U * principal[:, None, :]) @ Vt
    FA = F @ A
    strain = np.einsum('qij,qij->q', FA, F) - 1
    psi += .5 * params.k_f * strain**2
    P += 2 * params.k_f * strain[:, None, None] * FA
    return psi, P


def minimize_with_backtracking(objective, start, tolerance=1e-9, max_iterations=400):
    """Small L-BFGS reference with explicit admissibility/Armijo checks.

    An inadmissible trial has infinite energy and is *backtracked*, never
    accepted as a zero-gradient endpoint. No convergence on energy alone.
    """
    y = start.copy()
    value, grad = objective(y)
    history = []
    rejected = 0
    for iteration in range(max_iterations+1):
        if np.isfinite(value) and np.linalg.norm(grad, ord=np.inf) <= tolerance:
            return y, iteration, rejected
        if not np.isfinite(value) or iteration == max_iterations:
            raise RuntimeError('prototype L-BFGS residual did not converge')
        q = grad.copy()
        alpha = []
        for s, z, rho in reversed(history):
            a = rho*np.dot(s, q)
            alpha.append(a)
            q -= a*z
        gamma = 1. if not history else np.dot(history[-1][0], history[-1][1])/np.dot(history[-1][1], history[-1][1])
        direction = gamma*q
        for (s, z, rho), a in zip(history, reversed(alpha)):
            direction += s*(a-rho*np.dot(z, direction))
        direction = -direction
        slope = float(np.dot(grad, direction))
        if not np.isfinite(slope) or slope >= 0:
            direction = -grad
            slope = -float(np.dot(grad, grad))
            history.clear()
        step = 1.
        for _ in range(50):
            trial = y + step*direction
            trial_value, trial_grad = objective(trial)
            if np.isfinite(trial_value) and trial_value <= value+1e-4*step*slope:
                break
            rejected += 1
            step *= .5
        else:
            raise RuntimeError('prototype line search failed before residual convergence')
        s, z = trial-y, trial_grad-grad
        curvature = float(np.dot(s, z))
        if curvature > 1e-12*np.linalg.norm(s)*np.linalg.norm(z):
            history.append((s, z, 1/curvature))
            history = history[-10:]
        y, value, grad = trial, trial_value, trial_grad
    raise RuntimeError('unreachable L-BFGS termination')


class MaterialQ1:
    def __init__(self, grid=17, ppc_axis=2, order=3, density=1., field='smooth'):
        if grid < 9 or (grid - 1) % 8:
            raise ValueError("grid must be 8*k+1, at least 9")
        if ppc_axis < 2:
            raise ValueError("at least two particles per axis required for full-rank Q1 mass")
        if order < 2 or not np.isfinite(density) or density <= 0:
            raise ValueError("use quadrature order >= 2 and positive finite density")
        if field not in ('uniform', 'smooth', 'crossed'):
            raise ValueError("unknown direction field")
        self.h = 1. / (grid - 1)
        self.lo = np.array([.25, .4375, .4375])
        self.counts = np.array([(grid - 1)//2, (grid - 1)//8, (grid - 1)//8])
        self.shape = tuple(self.counts + 1)
        ijk = np.array(list(itertools.product(*[range(n) for n in self.shape])))
        self.X = self.lo + self.h * ijk
        self.fixed = ijk[:, 0] == 0
        self.free = ~self.fixed
        self.field = field
        self.particles = self.rule(ppc_axis, gauss=False)
        self.quadrature = self.rule(order, gauss=True)
        self.mass = density * self.particles.weight
        N = self.particles.N
        self.M = (N.T @ N.multiply(self.mass[:, None])).tocsc()
        self.lumped = np.asarray(N.T @ self.mass).ravel()
        eigen = eigvalsh(self.M.toarray())
        if eigen[0] <= 1e-12 * eigen[-1]:
            raise ValueError("particle sampling cannot resolve the selected velocity space")
        self.mass_eigenvalues = eigen
        self.factor = splu(self.M)
        self.free_factor = splu(self.M[self.free][:, self.free].tocsc())
        self.params = AnisotropicMaterialParams(10., 20., 200.)

    def sample(self, X, weights):
        X = np.asarray(X, dtype=float)
        local = (X - self.lo) / self.h
        if np.any(local < -1e-10) or np.any(local > self.counts + 1e-10):
            raise ValueError("reference samples outside material template")
        cell = np.minimum(np.floor(local).astype(int), self.counts - 1)
        cell = np.maximum(cell, 0)
        q = local - cell
        vertices = cell[:, None, :] + CORNERS
        ids = np.ravel_multi_index(vertices.reshape(-1, 3).T, self.shape)
        rows = np.repeat(np.arange(len(X)), 8)
        f = np.where(CORNERS[None, :, :] == 1, q[:, None, :], 1-q[:, None, :])
        shape = (len(X), len(self.X))
        N = sp.coo_matrix((f.prod(axis=2).ravel(), (rows, ids)), shape=shape).tocsr()
        D = []
        for d in range(3):
            val = (2*CORNERS[None, :, d]-1) * f[:, :, [j for j in range(3) if j != d]].prod(axis=2) / self.h
            D.append(sp.coo_matrix((val.ravel(), (rows, ids)), shape=shape).tocsr())
        angle = np.zeros(len(X))
        if self.field == 'smooth':
            angle = (X[:, 1] - self.lo[1]) / .125 * np.pi/2
        elif self.field == 'crossed':
            angle = np.where(X[:, 1] >= .5, np.pi/2, 0.)
        a = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(len(X))))
        return Samples(X, np.asarray(weights), N, tuple(D), np.einsum('qi,qj->qij', a, a))

    def rule(self, order, gauss):
        if gauss:
            z, w = np.polynomial.legendre.leggauss(order)
            z, w = (z+1)/2, w/2
        else:
            z, w = (np.arange(order)+.5)/order, np.full(order, 1/order)
        local = np.array(list(itertools.product(z, repeat=3)))
        weight = np.prod(np.array(list(itertools.product(w, repeat=3))), axis=1) * self.h**3
        cells = np.array(list(itertools.product(*[range(n) for n in self.counts])))
        X = self.lo + self.h * (cells[:, None, :] + local).reshape(-1, 3)
        return self.sample(X, np.tile(weight, len(cells)))

    @staticmethod
    def gradient(samples, nodal):
        return np.stack([D @ nodal for D in samples.D], axis=2)

    def project(self, values, clamped=False):
        """Consistent-mass least-squares P2G; PIC-style, no APIC affine state."""
        values = np.asarray(values)
        rhs = self.particles.N.T @ (self.mass[:, None] * values)
        if not clamped:
            return self.factor.solve(rhs)
        result = np.zeros((len(self.X), values.shape[1]))
        result[self.free] = self.free_factor.solve(rhs[self.free])
        return result

    def recover(self, positions, Fp, tolerance=1e-8):
        """Recover from particles alone, reject histories outside this space.

        The independent Fp check prevents geometry recovery from silently
        discarding an incompatible/prestressed/plastic deformation history.
        """
        x = self.project(positions)
        F = self.gradient(self.particles, x)
        pos_error = np.linalg.norm(self.particles.N @ x - positions) / max(np.linalg.norm(positions), 1e-30)
        F_error = np.linalg.norm(F - Fp) / max(np.linalg.norm(Fp), 1e-30)
        if not np.isfinite([pos_error, F_error]).all() or max(pos_error, F_error) > tolerance:
            raise ValueError(f"incompatible particle history: x={pos_error:.3g}, F={F_error:.3g}")
        if np.any(np.linalg.det(F) <= 0):
            raise ValueError("inverted recovered material template")
        return x, dict(position_recovery_relative=float(pos_error), F_recovery_relative=float(F_error))

    def elastic(self, x):
        F = self.gradient(self.quadrature, x)
        psi, P = material_response(F, self.quadrature.A, self.params)
        force_gradient = sum(D.T @ (self.quadrature.weight[:, None] * P[:, :, d])
                             for d, D in enumerate(self.quadrature.D))
        return float(self.quadrature.weight @ psi), force_gradient

    def potential(self, v, x, predictor, dt):
        U, force = self.elastic(x + dt*v)
        dv = v-predictor
        return U + .5*float(np.sum(dv*(self.M @ dv))), self.M @ dv + dt*force

    def commit(self, x, positions, Fp, v, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        # Validate input before applying a multiplicative F update.
        recovered, _ = self.recover(positions, Fp)
        if not np.allclose(recovered, x, rtol=1e-9, atol=1e-10):
            raise ValueError("nodal template differs from particle history")
        Fn = self.gradient(self.particles, x)
        G = self.gradient(self.particles, v) @ np.linalg.inv(Fn)
        next_positions = positions + dt * (self.particles.N @ v)
        next_F = (np.eye(3) + dt*G) @ Fp
        restored, info = self.recover(next_positions, next_F)
        predicted = x + dt*v
        U, _ = self.elastic(predicted)
        Urestored, _ = self.elastic(restored)
        dF = self.gradient(self.quadrature, predicted) - self.gradient(self.quadrature, x)
        dFrecovered = self.gradient(self.quadrature, restored) - self.gradient(self.quadrature, x)
        info.update(history_energy_delta=Urestored-U,
                    history_energy_relative=(Urestored-U)/max(abs(U), 1e-20),
                    increment_recovery_relative=float(np.linalg.norm(dFrecovered-dF)/max(np.linalg.norm(dF), 1e-20)),
                    min_det=float(np.linalg.det(next_F).min()))
        return restored, next_positions, next_F, self.particles.N @ v, info

    def step(self, positions, Fp, vp, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        step_start=time.perf_counter()
        x, _ = self.recover(positions, Fp)
        predictor = self.project(vp, clamped=True)
        preparation_seconds=time.perf_counter()-step_start
        Uold = self.elastic(x)[0]
        Kold = .5*float(np.sum(self.mass[:, None]*vp**2))
        Kpredictor = .5*float(np.sum(predictor*(self.M@predictor)))
        # Positive diagonal scaling only; the actual inertia is consistent M.
        scale = np.sqrt(self.lumped[self.free, None])
        def objective(y):
            v = np.zeros_like(x)
            v[self.free] = y.reshape(-1, 3)/scale
            try:
                value, grad = self.potential(v, x, predictor, dt)
            except ValueError:
                return np.inf, np.zeros_like(y)
            return value, (grad[self.free]/scale).ravel()
        start = (predictor[self.free]*scale).ravel()
        solve_start=time.perf_counter()
        solution, iterations, backtracks = minimize_with_backtracking(objective, start)
        value, grad = objective(solution)
        solve_seconds=time.perf_counter()-solve_start
        # Do not accept a small objective change alone as solver convergence.
        residual = float(np.linalg.norm(grad, ord=np.inf))
        if not np.isfinite(value) or residual > 1e-7:
            raise RuntimeError(f"prototype solve failed: residual={residual:g}")
        v = np.zeros_like(x)
        v[self.free] = solution.reshape(-1, 3)/scale
        commit_start=time.perf_counter()
        restored, xp, F, velocities, info = self.commit(x, positions, Fp, v, dt)
        commit_seconds=time.perf_counter()-commit_start
        U, _ = self.elastic(restored)
        K = .5*float(np.sum(self.mass[:, None]*velocities**2))
        Utrial, force = self.elastic(x+dt*v)
        dv = v-predictor
        residual_vector = self.M@dv+dt*force
        # Exact discrete energy identity; separate BE's two remainder terms
        # from transfer loss, nonlinear solve error and history recovery.
        budget = dict(projection_delta=Kpredictor-Kold,
                      inertia_remainder=-.5*float(np.sum(dv*(self.M@dv))),
                      elastic_remainder=Utrial-Uold-dt*float(np.sum(force*v)),
                      solver_work=float(np.sum(v*residual_vector)),
                      kinetic_readback_delta=K-.5*float(np.sum(v*(self.M@v))))
        actual_delta=U+K-Uold-Kold
        budget['energy_budget_residual'] = actual_delta-(sum(budget.values())+info['history_energy_delta'])
        info.update(elastic=U, kinetic=K, mechanical=U+K,
                    iterations=iterations, backtracks=backtracks, scaled_residual_inf=residual,
                    clamp_speed=float(np.max(np.abs(v[self.fixed]))),
                    particle_preparation_seconds=preparation_seconds,
                    material_solve_seconds=solve_seconds,commit_and_history_seconds=commit_seconds,
                    total_step_seconds=time.perf_counter()-step_start, **budget)
        return xp, F, velocities, info
