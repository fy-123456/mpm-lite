"""Generic original quadratic potential with a full three-component boundary lift."""
from dataclasses import dataclass
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from engine.aniso_phase1.high_order_space import scalar_grams, BoxElastic
from engine.aniso_phase1.tensor_reference import coordinates, interpolate
from engine.aniso_phase1.research_a.baseline_adapter import Problem as ParentProblem
from engine.aniso_phase1.selective_patch import projectors
from benchmarks.aniso_v17_modes import controlled_case
from engine.aniso_phase1.research_a.baseline_adapter import BASE
from .cases import CaseSpec


@dataclass
class Problem:
    case: CaseSpec
    edges: list
    degree: int
    A: np.ndarray
    carrier_X: np.ndarray
    Ks: np.ndarray
    construction: dict
    initial_u: object = None

    @classmethod
    def from_archive(cls, case=None, degree=4, mesh="production"):
        case = case or CaseSpec()
        if mesh not in ("production", "construction"):
            raise ValueError("Unknown mesh")
        if mesh == "construction":
            # Same physical problem, inexpensive construction/operator checks.
            # Never used as a substitute for the prescribed production curves.
            from types import SimpleNamespace
            state, energy, *_ = controlled_case()
            with np.load(BASE / "v19/space/reconstruction32.npz", allow_pickle=False) as data:
                target = [data[f"axis{k}"].copy() for k in range(3)]
                A = interpolate(target, 2, data["A"], target, degree)
            parent = SimpleNamespace(carrier_X=state.Y, Ks=energy.Ks)
            edges = target
        else:
            parent = ParentProblem.from_archive(degree)
            A, edges = parent.A, parent.edges
        nodes = case.map_points(parent.carrier_X)
        mapped = case.map_edges(edges)
        if np.all(case.affine_scale == 1):
            Ks = parent.Ks.copy()*(case.mu/10.)
            stabilizer = "original frozen quadratic patch; scales with matrix shear modulus"
        else:
            _, energy, _, h, _ = controlled_case()
            ids = energy.ids
            P = projectors(nodes, ids, h)
            # Reassemble on physical mapped coordinates and physical patch volumes.
            weights = energy.weights*np.prod(case.affine_scale)*(case.mu/10.)
            m = ids.shape[1]
            local = (P.transpose(0, 2, 1)@P)*weights[:, None, None]
            Ks = sp.csr_matrix((local.ravel(), (np.repeat(ids, m, axis=1).ravel(),
                                                np.tile(ids, (1, m)).ravel())), shape=(len(nodes),)*2).toarray()
            stabilizer = "reassembled degree-two physical patch; physical volumes; h_ref=0.125 m"
        obj = cls(case, mapped, degree, A, nodes, Ks,
                  dict(mesh=mesh, carrier="affine pullback of frozen continuous quadratic-preserving map",
                       scale=case.affine_scale.tolist(), stabilization=stabilizer,
                       carrier_identity_reused=bool(np.all(case.affine_scale == 1))))
        del parent
        obj.initial_u = obj.equilibrate(np.empty((obj.n, 0)))[0]["u"]
        return obj

    @property
    def n(self): return self.A.shape[0]
    @property
    def free(self): return (self.carrier_X[:, 0] > .25+1e-12) & (self.carrier_X[:, 0] < .75-1e-12)
    @property
    def Q(self): return np.eye(len(self.carrier_X))[:, self.free]
    @property
    def lift(self): return self.case.boundary(self.carrier_X)
    @property
    def params(self): return self.case.params
    @property
    def fiber_tensor(self): return self.case.params.A0

    def nodes(self):
        return np.stack(np.meshgrid(*coordinates(self.edges, self.degree), indexing="ij"), axis=-1).reshape(-1, 3)

    def invariant_errors(self, W):
        poly = lambda x: np.column_stack((np.ones(len(x)), x, x*x, x[:, 0]*x[:, 1], x[:, 0]*x[:, 2], x[:, 1]*x[:, 2]))
        X = self.nodes(); fixed = (X[:, 0] <= .25+1e-12) | (X[:, 0] >= .75-1e-12)
        return dict(quadratic_polynomial_error=float(np.max(abs(self.A@poly(self.carrier_X)-poly(X)))),
                    local_fixed_grip_value=float(np.max(abs(W[fixed]), initial=0.)),
                    orthogonality_error=float(la.norm(W.T@W-np.eye(W.shape[1]))),
                    patch_affine_residual=float(la.norm(self.Ks@poly(self.carrier_X)[:, :4])))

    def equilibrate(self, W, return_matrix=False):
        if W.ndim != 2 or W.shape[0] != self.n or not np.isfinite(W).all():
            raise ValueError("Invalid local scalar basis")
        Q, lift = self.Q, self.lift; nf = Q.shape[1]
        T = np.column_stack((self.A@Q, W)); m = T.shape[1]
        G = scalar_grams(self.edges, self.degree, np.column_stack((T, self.A@lift)))
        h = self.case.H.reshape(3, 3, 3, 3)
        K = np.zeros((3*m, 3*m)); g = np.zeros((m, 3))
        for a in range(3):
            for b in range(3):
                block = np.zeros((m, m))
                for i in range(3):
                    for j in range(3):
                        if h[a, i, b, j]:
                            block += h[a, i, b, j]*G[i, j][:m, :m]
                            g[:, a] += h[a, i, b, j]*G[i, j][:m, m+b]
                if a == b: block[:nf, :nf] += Q.T@self.Ks@Q
                K[a*m:(a+1)*m, b*m:(b+1)*m] = block
        g[:nf] += Q.T@self.Ks@lift
        K = .5*(K+K.T)
        eigen = la.eigvalsh(K)
        if eigen[0] <= 0: raise RuntimeError("Nonpositive unshifted static stiffness")
        coef = la.solve(K, -g.T.ravel(), assume_a="pos").reshape(3, m).T
        y = lift+Q@coef[:nf]; alpha = coef[nf:]; u = self.A@y+W@alpha
        op = BoxElastic(self.edges, self.degree, self.case.H)
        force = op.apply(u.T.ravel()).reshape(3, -1).T
        carrier_force = self.A.T@force+self.Ks@y
        reduced = np.vstack((Q.T@carrier_force, W.T@force))
        energy = .5*float(np.sum(u*force)+np.sum(y*(self.Ks@y)))
        work = float(np.sum(lift*carrier_force)); right = self.carrier_X[:, 0] >= .75-1e-12
        reaction = carrier_force[right].sum(0)
        moment = np.cross(self.carrier_X[right]-self.case.rotation_center, carrier_force[right]).sum(0)
        residual = float(la.norm(reduced)); work_error = abs(2*energy-work)/max(abs(2*energy), 1e-12)
        X = self.nodes(); fixed = (X[:, 0] <= .25+1e-12) | (X[:, 0] >= .75-1e-12)
        boundary_error = float(np.max(abs(u[fixed]-self.case.boundary(X)[fixed])))
        passed = bool(np.isfinite(u).all() and residual <= 1e-6 and work_error <= 2e-5 and boundary_error <= 1e-8)
        result = dict(case_sha256=self.case.signature, energy_J=energy, reaction_N=float(reaction[0 if self.case.loading == "extension" else 1]),
                      reaction_vector_N=reaction.tolist(), reaction_moment_Nm=moment.tolist(),
                      boundary_work_J=work, work_identity_relative=work_error, free_residual=residual,
                      boundary_error_m=boundary_error, static_min=float(eigen[0]), static_max=float(eigen[-1]),
                      condition_number=float(eigen[-1]/eigen[0]), zero_modes=0, negative_modes=0,
                      static_passed=passed, scalar_local_dofs=W.shape[1], free_vector_dofs=3*m,
                      mass_included=False, stiffness_shift=0., original_strict_residual_passed=residual <= 1e-7)
        if not passed: raise RuntimeError(f"Invalid static scene: {result}")
        field = dict(y=y, local_coefficients=alpha, u=u)
        return (field, result, K, g, coef) if return_matrix else (field, result)
