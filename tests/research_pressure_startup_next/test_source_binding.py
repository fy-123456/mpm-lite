"""The research entry point must not inherit a nonzero topology source."""
import unittest
from benchmarks.research_pressure_startup_next.coupling import source_for_protocol

class SourceBinding(unittest.TestCase):
    def test_explicit_zero_source(self):
        self.assertEqual(source_for_protocol({'source_density_s_inv':0.}),0.)
    def test_unsupported_nonzero_source_rejected(self):
        with self.assertRaises(ValueError):source_for_protocol({'source_density_s_inv':.001})

if __name__=='__main__':unittest.main()
