"""Small state/interface tests; no dynamic cycle or spatial claim."""
import json
import unittest

import numpy as np

from engine.aniso_phase1.research_c.transfer import GridPacket
from engine.aniso_phase1.research_d.common_state import (
    CommonState, StateTrial, StateTransaction, kinetic_state_contract,
)


class CommonStateTests(unittest.TestCase):
    def initial(self):
        q = np.zeros((4, 3)); q[-1, 0] = .005
        return CommonState(q, np.zeros_like(q), predictor=np.ones_like(q),
            transfer={'residual_v': np.zeros((2, 3))},
            child_states={'B': {'rule': 'full', 'history': np.array([1.])},
                          'E': {'pressure': np.array([2.])}})

    def advance(self, trial):
        trial.state.step += 1
        trial.state.time += .01
        return trial

    def test_serializable_explicit_kinetic_contract(self):
        contract = kinetic_state_contract('frozen-A-content', (4, 3), 'full-mass-rule')
        self.assertEqual(json.loads(json.dumps(contract)), contract)
        self.assertEqual(contract['inertia']['density_kg_m3'], 1.)
        self.assertFalse(contract['inertia']['independent_APIC_microinertia'])
        self.assertFalse(contract['inertia']['mass_rule_may_follow_material_compression'])
        self.assertIn('required', contract['inertia']['carrier_local_cross_blocks'])
        self.assertFalse(contract['initial_state']['stress_free'])
        self.assertFalse(contract['scope']['dynamic_acceptance'])
        for shape in [(4,), (4, 2), (0, 3), (True, 3), (4, 3.0)]:
            with self.assertRaises(ValueError):
                kinetic_state_contract('A', shape)

    def test_initial_snapshot_and_trial_have_no_mutable_aliases(self):
        initial = self.initial(); transaction = StateTransaction(initial)
        before = transaction.snapshot().digest()
        initial.q[:] = 55
        snapshot = transaction.committed
        snapshot.child_states['B']['history'][:] = 66
        snapshot.transfer['residual_v'][:] = 77
        trial = transaction.begin_trial()
        trial.state.q[:] = 88
        trial.state.velocity[:] = 99
        trial.state.predictor[:] = 11
        trial.state.transfer['residual_v'][:] = 22
        trial.state.child_states['E']['pressure'][:] = 33
        self.assertEqual(transaction.snapshot().digest(), before)
        transaction.rollback(trial)
        self.assertEqual(transaction.snapshot().digest(), before)
        with self.assertRaisesRegex(ValueError, 'token'):
            transaction.commit(trial)

    def test_atomic_commit_publishes_every_field_and_copies_return(self):
        transaction = StateTransaction(self.initial())
        trial = self.advance(transaction.begin_trial())
        trial.state.q[0] = [.01, .02, .03]
        trial.state.velocity[:] = .04
        trial.state.predictor[:] = .05
        trial.state.transfer['residual_v'][:] = .06
        trial.state.child_states['B']['history'][:] = .07
        trial.state.child_states['E']['pressure'][:] = .08
        expected = trial.state.digest()
        result = transaction.accept(trial)
        self.assertEqual(result.digest(), expected)
        self.assertEqual(transaction.snapshot().digest(), expected)
        result.q[:] = 99
        trial.state.child_states['B']['history'][:] = 99
        self.assertEqual(transaction.snapshot().digest(), expected)
        self.assertEqual(transaction.revision, 1)

    def test_validator_failure_and_mutation_leave_every_committed_field(self):
        def validator(state):
            if state.q[0, 0] < -1:
                state.predictor[:] = 999
                state.child_states['B']['history'][:] = 999
                raise ValueError('detF <= 0')
            # Even an ill-behaved validator cannot mutate accepted data.
            state.q[:] = 999
            state.child_states['E']['pressure'][:] = 999
        transaction = StateTransaction(self.initial(), validator)
        original = transaction.snapshot().digest()
        bad = self.advance(transaction.begin_trial()); bad.state.q[0, 0] = -2
        bad.state.predictor[:] = 12
        bad.state.child_states['B']['rule'] = 'compressed'
        with self.assertRaisesRegex(ValueError, 'detF'):
            transaction.commit(bad)
        self.assertEqual(transaction.snapshot().digest(), original)
        self.assertEqual(transaction.revision, 0)
        with self.assertRaisesRegex(ValueError, 'token'):
            transaction.rollback(bad)
        valid = self.advance(transaction.begin_trial())
        accepted = transaction.commit(valid)
        self.assertEqual(accepted.q[0, 0], 0)
        self.assertEqual(accepted.child_states['E']['pressure'][0], 2)

    def test_foreign_forged_sibling_and_stale_trials_are_rejected(self):
        a = StateTransaction(self.initial()); b = StateTransaction(self.initial())
        trial = self.advance(a.begin_trial()); sibling = self.advance(a.begin_trial())
        foreign = self.advance(b.begin_trial())
        forged = StateTrial(trial.token, trial.base_revision, trial.state.clone())
        for bad in (foreign, forged, None):
            with self.assertRaisesRegex(ValueError, 'token'):
                a.commit(bad)
        a.commit(trial)
        for stale in (trial, sibling):
            with self.assertRaisesRegex(ValueError, 'token'):
                a.commit(stale)
        self.assertEqual(b.revision, 0)

    def test_nonfinite_shape_and_time_fail_atomically(self):
        mutations = [
            lambda s: setattr(s, 'q', np.zeros((3, 3))),
            lambda s: setattr(s, 'velocity', np.full((4, 3), np.nan)),
            lambda s: setattr(s, 'q', np.full((4, 3), 1j)),
            lambda s: setattr(s, 'predictor', np.zeros((2, 3))),
            lambda s: s.child_states['B'].update(history=np.array([np.inf])),
            lambda s: s.transfer.update(residual_v=np.full((2, 3), np.nan)),
            lambda s: setattr(s, 'time', float('nan')),
            lambda s: setattr(s, 'time', 0),
            lambda s: setattr(s, 'step', 0),
            lambda s: setattr(s, 'step', 1.5),
            lambda s: setattr(s, 'step', True),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                transaction = StateTransaction(self.initial()); before = transaction.snapshot().digest()
                trial = self.advance(transaction.begin_trial()); mutate(trial.state)
                with self.assertRaises((ValueError, TypeError)):
                    transaction.commit(trial)
                self.assertEqual(transaction.snapshot().digest(), before)
                self.assertEqual(transaction.revision, 0)
                with self.assertRaises(ValueError):
                    transaction.commit(trial)

    def test_c_grid_packet_is_owned_and_serializable(self):
        packet = GridPacket(np.zeros((1, 3), dtype=int), np.zeros((1, 1), dtype=int),
                            np.ones((1, 1)), np.zeros((1, 1, 3)), np.zeros((1, 3)),
                            np.zeros((1, 3)), np.zeros((1, 3, 3)), np.zeros((1, 3), dtype=int))
        state = self.initial(); state.transfer = packet
        transaction = StateTransaction(state)
        packet.residual_C[:] = 123
        snapshot = transaction.snapshot()
        self.assertTrue(np.all(snapshot.transfer.residual_C == 0))
        np.testing.assert_array_equal(snapshot.transfer.decode()[0], np.zeros((1, 3)))
        self.assertIn('residual_C', json.loads(json.dumps(snapshot.to_dict()))['transfer'])
        trial = self.advance(transaction.trial(state.q))
        trial.state.transfer.residual_C[:] = 44
        transaction.rollback(trial)
        self.assertTrue(np.all(transaction.snapshot().transfer.residual_C == 0))

    def test_detF_validator_uses_owned_full_displacement(self):
        def check_deformation(state):
            F = np.eye(3) + np.diag(state.q[0])
            if np.linalg.det(F) <= 0:
                raise ValueError('nonpositive detF')
        transaction = StateTransaction(self.initial(), check_deformation)
        trial = self.advance(transaction.begin_trial()); trial.state.q[0, 0] = -1.1
        with self.assertRaisesRegex(ValueError, 'detF'):
            transaction.commit(trial)
        self.assertEqual(transaction.snapshot().step, 0)


if __name__ == '__main__':
    unittest.main()
