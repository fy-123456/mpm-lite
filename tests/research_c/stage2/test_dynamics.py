import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from engine.aniso_phase1.research_c.stage2.model import Boundary, DynamicModel, MassRankError
from engine.aniso_phase1.research_c.stage2.dynamics import AVF, StepRejected
from engine.aniso_phase1.research_c.stage2.checkpoint import binding, save_checkpoint, load_checkpoint
from engine.aniso_phase1.research_d.common_state import CommonState, StateTransaction


class Space:
    """Coupled finite-dimensional oracle, never evidence for the real space."""
    ndof = 3; n = 2; signature = 'test-only'; edges = [np.array([0., 1.])]*3
    carrier_X = np.array([[0., 0., 0.], [1., 0., 0.]])
    boundary = {'right_x': 1.}
    free_scalar_ids = np.array([0, 2]); fixed_scalar_ids = np.array([1])
    reference = np.vstack((carrier_X, np.zeros((1, 3))))

    @staticmethod
    def _check(a, shape, name):
        if np.shape(a) != shape or not np.isfinite(a).all(): raise ValueError(name)
        return np.asarray(a)


class Oracle(DynamicModel):
    def __init__(self, hold=0.):
        s = Space()
        mass = np.array([[2., .3, .4], [.3, 1.5, .2], [.4, .2, 1.]])
        super().__init__(s, mass, boundary=Boundary(s, hold=hold), rest_K=3*np.eye(9))

    def evaluate(self, q, direction=None):
        self.space._check(q, (3, 3), 'finite q')
        if np.max(abs(q)) > 1: raise ValueError('illegal material state')
        result = dict(U=1.5*np.sum(q*q)+.25*np.sum(q**4),
            material_U=np.sum(q*q)+.25*np.sum(q**4), stabilization_U=.5*np.sum(q*q),
            force=3*q+q**3, material_force=2*q+q**3, min_detF=1., material_calls=1)
        if direction is not None: result['tangent_action']=(3+3*q*q)*direction
        return result


class DynamicsTests(unittest.TestCase):
    def initialized(self, hold=0.):
        m = Oracle(hold); state = m.rest()
        state.q[0, 0] = .01; state.velocity[2, 1] = -.02
        return AVF(m, state)

    def test_singular_free_mass_is_rejected_before_integrator(self):
        s = Space()
        # Two nonzero diagonal rows but exactly dependent free coordinates.
        mass = np.array([[1., 0., 1.], [0., 1., 0.], [1., 0., 1.]])
        with self.assertRaisesRegex(MassRankError, 'rank 1/2'):
            DynamicModel(s, mass)

    def test_boundary_derivative_and_breaks(self):
        b = Boundary(Space())
        for t in (.1, .4, .55, .7, 1., 1.3):
            np.testing.assert_allclose((b.lift(t+1e-6)-b.lift(t-1e-6))/2e-6,
                                      b.speed(t), atol=1e-10)
        for t in (0., .5, .6, 1.1, 1.6): self.assertEqual(np.linalg.norm(b.speed(t)), 0.)

    def test_full_mass_endpoint_duality(self):
        m = Oracle(); pre = np.arange(9).reshape(3, 3)*.01
        v, impulse, row = m.endpoint(pre, 0.)
        self.assertLess(np.linalg.norm(impulse[m.free]), 1e-14)
        self.assertLess(abs(row['endpoint_identity_error_J']), 1e-14)
        self.assertGreater(np.linalg.norm(v[m.free]-pre[m.free]), 0.)

    def test_fixed_boundary_energy_and_nonlinear_residual(self):
        solver = self.initialized(); s0 = solver.state
        E0 = solver.model.kinetic(s0.velocity)+solver.model.evaluate(s0.q)['U']
        for _ in range(10):
            row = solver.step(.01)
            self.assertLess(row['true_residual'], row['residual_tolerance'])
            self.assertLess(abs(row['budget_defect_J']), 1e-12)
            self.assertNotEqual(row['stabilization_J'], 0.)
        self.assertLess(abs(row['total_J']-E0), 1e-10)

    def test_driven_boundary_nonzero_stabilization_reaction(self):
        solver = AVF(Oracle(hold=None))
        row = solver.step(.01)
        self.assertEqual(row['velocity_constraint'], 0.)
        self.assertGreater(abs(row['stabilization_reaction_N']), 0.)
        self.assertLess(abs(row['energy_balance_J']), 1e-9)

    def test_forward_velocity_reverse(self):
        forward = self.initialized(); initial = forward.state
        forward.step(.01); s = forward.state; s.velocity *= -1; s.predictor = None
        reverse = AVF(forward.model, s); reverse.step(.01)
        np.testing.assert_allclose(reverse.state.q, initial.q, atol=1e-10)
        np.testing.assert_allclose(reverse.state.velocity, -initial.velocity, atol=1e-9)

    def test_failure_rolls_back_every_value_then_retries(self):
        for fault in ('material', 'search', 'boundary', 'postcheck', 'child'):
            with self.subTest(fault=fault):
                solver = self.initialized(); before = solver.state.digest()
                def inject(phase, state):
                    if fault == 'material' and phase == 'before_material': state.q[:] = 2.
                    if fault == 'search' and phase == 'before_search': raise ValueError('line search failure')
                    if fault == 'boundary' and phase == 'after_prepare': state.velocity[1, 0] = 1.
                def prepare(state, children):
                    children['B_proposal'] = {'rule': 'trial-only', 'history': np.ones(3)}
                    return children
                def validate(state):
                    if fault in ('postcheck', 'child'): raise ValueError('post-prepare failure')
                with self.assertRaises(StepRejected):
                    solver.step(.01, inject=inject, prepare_children=prepare, validate_trial=validate)
                self.assertEqual(solver.state.digest(), before)
                solver.step(.01)
                clean = self.initialized(); clean.step(.01)
                np.testing.assert_array_equal(solver.state.q, clean.state.q)
                np.testing.assert_array_equal(solver.state.velocity, clean.state.velocity)
                self.assertNotIn('B_proposal', solver.state.child_states)

    def test_stale_token_and_single_commit(self):
        solver = self.initialized(); tx = solver._transaction
        old = tx.begin_trial(); solver.step(.01)
        with self.assertRaises(ValueError): tx.commit(old)
        self.assertEqual(tx.revision, 1)

    def test_checkpoint_resume_and_wrong_identity(self):
        solver = self.initialized(); solver.step(.01)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoint.json'
            save_checkpoint(path, solver, .01, {'test': 'digest'}, 'parent')
            expected = binding(solver, .01, {'test': 'digest'}, 'parent')
            state = load_checkpoint(path, solver.model, expected)
            resumed = AVF(solver.model, state); resumed.step(.01); solver.step(.01)
            np.testing.assert_array_equal(resumed.state.q, solver.state.q)
            np.testing.assert_array_equal(resumed.state.velocity, solver.state.velocity)
            bad = copy.deepcopy(expected); bad['dt'] *= 2
            with self.assertRaises(ValueError): load_checkpoint(path, solver.model, bad)

    def test_wrong_identity_rejected(self):
        m = Oracle(); s = m.rest(); s.child_states['identity']['mass'] = 'wrong'
        # The state must own its identity instead of mutating the model.
        self.assertNotEqual(m.identity['mass'], 'wrong')
        with self.assertRaises(ValueError): AVF(m, s)


if __name__ == '__main__': unittest.main()
