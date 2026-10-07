"""Exercise the production sparse residual/matvec on a multi-block grid."""
import numpy as np
import warp as wp
from engine.types import real, vec3, mat33
from engine.sp_grid import B
from engine.boundary_utils import boundary_projection_kernel
from .solver import AnisotropicLiteImplicitSolver
from .types import AnisotropicMaterialParams


@wp.kernel
def assign_velocity(ndof: wp.array(dtype=wp.vec2i), values: wp.array(dtype=vec3), grid: wp.array(dtype=vec3, ndim=4)):
    i = wp.tid()
    b, l = ndof[i][0], ndof[i][1]
    grid[b, l//(B*B), (l//B)%B, l%B] = values[i]


@wp.kernel
def assign_deformation(cdof: wp.array(dtype=wp.vec2i), F: mat33, state: wp.array(dtype=mat33, ndim=4)):
    i = wp.tid()
    b, l = cdof[i][0], cdof[i][1]
    state[0, b, l//(B*B), l%(B*B)] = F


class SparseProbe:
    def __init__(self, device='cpu', kf=200., dt=.002, boundary=True, solver_cls=AnisotropicLiteImplicitSolver,
                 force_discretization='variational'):
        self.s = s = solver_cls((65,)*3,
            AnisotropicMaterialParams(10., 20., kf, [1., 1., 0.]), dx=1/64, device=device, gravity=0.,
            force_discretization=force_discretization)
        axis = np.array([.493, .507])
        pts = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
        s.seed_particles(pts, density=1., vol0=.03**3/len(pts))
        if boundary:
            nodes = np.array([[i,j,k] for i in (31,33) for j in range(29,36) for k in range(29,36)])
            kinds = np.where(nodes[:,0] == 31, 1, 2).astype(np.int32)
            normals = np.tile(np.array([1., 2., 0.])/np.sqrt(5), (len(nodes), 1))
            s.paint_boundary(nodes, kinds, boundary_n=normals)
        s.set_dt(dt)
        s.step(max_iters=0, print_every=0)
        self.n = int(s.n_active_nodes.numpy().copy()[0])
        self.nc = int(s.n_active_centers.numpy().copy()[0])
        self.p, self.Ap = wp.zeros_like(s.node_residual), wp.zeros_like(s.node_residual)
        self.base = np.zeros((self.n, 3))

    def project(self, v):
        s = self.s
        full = np.zeros((s.MAX_DOF, 3)); full[:self.n] = v
        a = wp.array(full, dtype=vec3, device=s.device)
        wp.launch(boundary_projection_kernel, dim=self.n, inputs=[
            s.n_active_nodes, s.ndof2bijk, s.block_xyz_by_id, a,
            s.bc_block2bid, s.bc_type, s.bc_norm, s.bc_velo,
            s.hf_bc_p, s.hf_bc_n, s.hf_bc_v, s.hf_bc_type, s.num_hf,
            s.grid_size, s.dx], device=s.device)
        return a[:self.n].numpy().copy()

    def residual(self, v):
        s = self.s
        wp.launch(assign_velocity, dim=self.n, inputs=[s.ndof2bijk, wp.array(v, dtype=vec3, device=s.device), s.grid_v_it], device=s.device)
        if not s.evaluate_residual():
            raise ValueError('nonpositive trial Jacobian')
        return s.node_residual[:self.n].numpy().copy()

    def tangent(self, v, project_pd=False):
        s = self.s
        self.p.zero_()
        wp.copy(self.p, wp.array(v, dtype=vec3, device=s.device), count=self.n)
        s.apply_tangent(self.p, self.Ap, project_pd=project_pd)
        return self.Ap[:self.n].numpy().copy()

    def set_deformation(self, F):
        s = self.s
        wp.launch(assign_deformation, dim=self.nc, inputs=[s.cdof2bijk, mat33(F), s.aniso_committed_F], device=s.device)
        s.ptc_F.assign(wp.array(np.broadcast_to(F,(s.n_ptc,3,3)).copy(), dtype=mat33, device=s.device))

    def variational_check(self, F=None, seed=42):
        """Full free-subspace spectrum on a small ACTUAL sparse grid, not random SPD probes."""
        if F is None:
            F = np.array([[1.08,.06,.01],[0.,.96,.03],[.02,0.,1.02]])
        self.set_deformation(F)
        rng = np.random.default_rng(seed)
        v = self.project(rng.normal(size=(self.n,3))*.01)
        p = self.project(rng.normal(size=(self.n,3)))
        p /= np.linalg.norm(p)
        r = self.residual(v)
        analytic = float(np.sum(r*p))
        grad_errors = {}
        tangent_errors = {}
        Ap = self.tangent(p)
        for eps in (1e-3,1e-4,1e-5,1e-6):
            rp = self.residual(v+eps*p); ep = self.s.incremental_potential()
            rm = self.residual(v-eps*p); em = self.s.incremental_potential()
            grad_errors[str(eps)] = abs((ep-em)/(2*eps)-analytic)/max(abs(analytic),1e-30)
            tangent_errors[str(eps)] = float(np.linalg.norm((rp-rm)/(2*eps)-Ap)/np.linalg.norm(Ap))
        self.residual(v)
        identity = np.eye(3*self.n)
        Q = np.column_stack([self.project(col.reshape(self.n,3)).ravel() for col in identity])
        values, vectors = np.linalg.eigh((Q+Q.T)/2)
        Z = vectors[:,values>.5]
        matrices = {}
        result = dict(potential_fd_errors=grad_errors, tangent_fd_errors=tangent_errors,
                      free_dofs=Z.shape[1], active_blocks=int(self.s.bcn))
        ndof = self.s.ndof2bijk[:self.n].numpy()
        masses = self.s.grid_m[:int(self.s.bcn)].numpy()
        m = np.array([masses[b,l//(B*B),(l//B)%B,l%B] for b,l in ndof])
        reduced_mass = Z.T @ (np.repeat(m,3)[:,None]*Z)
        L = np.linalg.cholesky(reduced_mass)
        for projected, name in ((False,'exact'),(True,'modified')):
            JZ = np.column_stack([self.tangent(col.reshape(self.n,3), projected).ravel() for col in Z.T])
            J = Z.T @ JZ
            matrices[name] = J
            symmetry = np.linalg.norm(J-J.T)/np.linalg.norm(J)
            symmetric = (J+J.T)/2
            eig = np.linalg.eigvalsh(symmetric)
            normalized = np.linalg.solve(L, np.linalg.solve(L, symmetric).T).T
            scaled_eig = np.linalg.eigvalsh((normalized+normalized.T)/2)
            result[name] = dict(symmetry_relative=float(symmetry), min_eigenvalue=float(eig[0]),
                                max_eigenvalue=float(eig[-1]), negative_eigenvalues=int(np.sum(eig<0)),
                                mass_scaled_min=float(scaled_eig[0]), mass_scaled_max=float(scaled_eig[-1]))
        result['tangent_modification_relative'] = float(np.linalg.norm(matrices['modified']-matrices['exact'])/np.linalg.norm(matrices['exact']))
        self.last_dense = (Z, matrices, reduced_mass)
        self.base = v
        return result

    def check(self, seed=42):
        s = self.s
        rng = np.random.default_rng(seed)
        F = np.array([[1.08, .06, .01], [0., .96, .03], [.02, 0., 1.02]])
        wp.launch(assign_deformation, dim=self.nc, inputs=[s.cdof2bijk, mat33(F), s.aniso_committed_F], device=s.device)
        if getattr(s, 'quadrature_kind', 'center') == 'particle':
            s.ptc_F.assign(wp.array(np.broadcast_to(F,(s.n_ptc,3,3)).copy(),dtype=mat33,device=s.device))
        v = self.project(rng.normal(size=(self.n, 3))*.05)
        p, q = (self.project(rng.normal(size=(self.n, 3))) for _ in range(2))
        p /= np.linalg.norm(p); q /= np.linalg.norm(q)
        self.residual(v)
        committed = s.aniso_committed_F[:, :int(s.bcn)].numpy().copy()
        trial_array = s.particle_trial_F if getattr(s, "quadrature_kind", "center") == "particle" else s.aniso_trial_F[:, :int(s.bcn)]
        trial = trial_array.numpy().copy()
        Ap, Aq = self.tangent(p), self.tangent(q)
        repeat = np.max(np.abs(self.tangent(p)-Ap))
        frozen = np.array_equal(trial, trial_array.numpy().copy())
        errors = {}
        for eps in (1e-3, 1e-4, 1e-5, 1e-6):
            fd = (self.residual(v+eps*p)-self.residual(v-eps*p))/(2*eps)
            errors[str(eps)] = float(np.linalg.norm(fd-Ap)/max(np.linalg.norm(Ap), 1e-30))
        self.residual(v)
        rayleigh = []
        symmetry = []
        for _ in range(12):
            a, b = (self.project(rng.normal(size=(self.n,3))) for _ in range(2))
            a /= np.linalg.norm(a); b /= np.linalg.norm(b)
            Aa, Ab = self.tangent(a), self.tangent(b)
            rayleigh.append(float(np.sum(a*Aa)))
            symmetry.append(float(abs(np.sum(a*Ab)-np.sum(b*Aa))/max(np.linalg.norm(Aa)*np.linalg.norm(b), np.linalg.norm(Ab)*np.linalg.norm(a), 1e-30)))
        boundary_error = np.linalg.norm(Ap-self.project(Ap))/max(np.linalg.norm(Ap), 1e-30)
        from .linear import nonsymmetric_solve
        from engine.kernel.d3.kernel_lite_implicit import lite_implicit_precond_kernel_Hii
        rhs = wp.zeros_like(s.node_residual)
        solution = wp.zeros_like(rhs)
        exact = wp.zeros_like(rhs)
        wp.copy(exact, wp.array(p, dtype=vec3, device=s.device), count=self.n)
        s.apply_tangent(exact, rhs)
        def precondition(a, out):
            wp.launch(lite_implicit_precond_kernel_Hii, dim=s.MAX_DOF, inputs=[
                s.n_active_nodes, s.ndof2bijk, s.block_xyz_by_id, s.node_Hii_inv, a, out,
                s.bc_block2bid, s.bc_type, s.bc_norm, s.bc_velo,
                s.hf_bc_p, s.hf_bc_n, s.hf_bc_v, s.hf_bc_type, s.num_hf, s.dx], device=s.device)
        iterations, solve_error, solve_tol = nonsymmetric_solve(s.apply_tangent, rhs, solution,
                                                               precondition, 1e-8, 1e-14, 500)
        solution_error = np.linalg.norm(solution[:self.n].numpy()-p)

        return dict(linear_converged=bool(solve_error <= solve_tol), linear_iterations=int(iterations), linear_residual=float(solve_error),
                    linear_tolerance=float(solve_tol), manufactured_solution_error=float(solution_error),
                    kf=s.aniso_params.k_f, dt=s.dt, active_blocks=int(s.bcn), active_nodes=self.n,
                    active_centers=self.nc, fd_errors=errors, repeat_max_error=float(repeat),
                    max_symmetry_error=max(symmetry), min_rayleigh=min(rayleigh),
                    boundary_projection_error=float(boundary_error), frozen_trial=frozen,
                    frozen_committed=np.array_equal(committed, s.aniso_committed_F[:, :int(s.bcn)].numpy().copy()))
