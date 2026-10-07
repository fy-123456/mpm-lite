import unittest
import numpy as np
from benchmarks.research_b.scenes import source_rule, gradient_map, PARAMS, SPACE_ID, directions
from engine.aniso_phase1.research_b.binding import BoundMaterialFamily


class MaterialOriginTests(unittest.TestCase):
    def family(self, angle):
        return BoundMaterialFamily(source_rule(angle, order=3), gradient_map, PARAMS, SPACE_ID, 9)

    def test_changed_direction_field_rejected_even_at_zero_energy(self):
        original, changed = self.family(45.), self.family(17.)
        candidate = original.compressed(4)
        session = original.session(candidate, 1., 1.)
        q = np.zeros((9,3))
        self.assertLess(abs(original.full.energy(q)-changed.full.energy(q)), 1e-12)
        with self.assertRaises(ValueError):
            session.propose(changed.full, q, directions()['random'])
        self.assertIs(session.operator, candidate)

    def test_same_source_local_fallback_keeps_rule_and_transaction_semantics(self):
        family = self.family(45.)
        compact = family.compressed(4)
        session = family.session(compact, 1., 1.)
        q = np.zeros((9,3)); q[0,0] = .03
        refined = family.fallback(compact, [0,1])
        p = session.propose(refined, q, directions()['random'], fallback_partitions=(0,1))
        self.assertTrue(session.commit_rule(p, q, accepted_step=True))
        self.assertIs(session.operator, refined)
        self.assertEqual(sum(refined.rule.partition == 0), sum(family.full.rule.partition == 0))


if __name__=='__main__':unittest.main()
