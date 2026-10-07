"""Spectral reuse must retain exact repeated-eigenvalue and unloading response."""
import unittest
import numpy as np
from engine.aniso_phase1.types import AnisotropicMaterialParams
from engine.aniso_phase1.history_increment import material_tangent
from engine.aniso_phase1.research_sequential_next.cpu_linearization import spectral_prepare,spectral_action


class CPUSpectralTests(unittest.TestCase):
    def test_batched_directions_repeated_spectrum_and_unloading(self):
        params=AnisotropicMaterialParams(10.,20.,200.);a=np.array([1.,1.,0.])/np.sqrt(2);A=np.outer(a,a)[None]
        F=np.array([np.eye(3),np.diag([.98,.98,1.02]),[[1.02,.07,0],[0,.96,0],[0,0,1.01]]])
        A=np.broadcast_to(A,F.shape);prepared=spectral_prepare(F,params);rng=np.random.default_rng(901)
        for sign in [1.,-1.]:
            d=sign*rng.normal(size=F.shape)
            np.testing.assert_allclose(spectral_action(prepared,A,d,params),material_tangent(F,A,d,params),rtol=2e-13,atol=1e-12)
        with self.assertRaises(ValueError):spectral_prepare(-np.eye(3)[None],params)


if __name__=='__main__':unittest.main()
