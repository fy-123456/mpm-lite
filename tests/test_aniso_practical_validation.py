import tempfile
import unittest
from pathlib import Path
import numpy as np
from benchmarks.aniso_practical_validation import metric
from engine.aniso_phase1.convergence_reference import geometry
from engine.aniso_phase1.history_increment import HistoryField,ReferenceBasis,frozen


class PracticalMetricsTests(unittest.TestCase):
    def test_strain_scale_and_true_band_volume(self):
        source=geometry(9);basis=ReferenceBasis(source)
        A=np.eye(3);A[1,0]=.02
        B=np.eye(3);B[1,0]=.01
        def fields(F):
            return dict(position=HistoryField(((basis,frozen(source.X@F.T)),)),
                        velocity=HistoryField(((basis,frozen(source.X*.01)),)))
        with tempfile.TemporaryDirectory() as folder:
            r=metric(Path(folder),'affine',fields(A),fields(B))
            self.assertAlmostEqual(r['regions']['all']['relative']['F'],1.,places=10)
            self.assertAlmostEqual(r['regions']['band']['relative']['F'],1.,places=10)
            self.assertAlmostEqual(r['regions']['band']['volume']/r['regions']['all']['volume'],.125,places=12)
            self.assertEqual(r['regions']['all']['rms']['v'],0.)
            # Reusing a name with different fields must invalidate cached metrics.
            same=metric(Path(folder),'affine',fields(B),fields(B))
            self.assertEqual(same['regions']['all']['rms']['F'],0.)
            fine=metric(Path(folder),'affine',fields(A),fields(B),5)
            self.assertAlmostEqual(fine['regions']['all']['relative']['F'],1.,places=10)


if __name__=='__main__':unittest.main()
