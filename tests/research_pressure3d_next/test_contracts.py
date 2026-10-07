"""Small asymmetric ownership and shared tensor adjoint checks."""
import unittest
import numpy as np
from engine.aniso_phase1.research_pressure3d_next.geometry import blocks,indices
from engine.aniso_phase1.research_stabilization_boundary_next.reference_topology import ReferenceTopology
from benchmarks.research_restoring_rt0_next.grid_transfer import restriction
from engine.aniso_phase1.research_continuous_geometry_next.reference import prolongation

class Contracts(unittest.TestCase):
    def test_noncontiguous_asymmetric_partition(self):
        points=[np.array([.1,.2,.6,.8]),np.array([.1,.4,.7]),np.array([.05,.2,.45,.7,.9])]
        cuts=[[0.,.5,1.],[0.,.3,1.],[0.,.3,.6,1.]];shape=tuple(map(len,points));b=blocks(points,cuts)
        allids=np.concatenate([indices(v,shape) for v in b]);np.testing.assert_array_equal(np.sort(allids),np.arange(np.prod(shape)))
        grid=np.arange(np.prod(shape)).reshape(shape)
        for v in b:np.testing.assert_array_equal(indices(v,shape),grid[tuple(slice(*x) for x in v)].ravel())
        with self.assertRaises(ValueError):blocks(points,[[0.,.01,.5,1.],*cuts[1:]])

    def test_rt0_transfer_affine_flux(self):
        c=ReferenceTopology([[0.,.3,1.],[0.,1.],[0.,1.]])
        f=ReferenceTopology([[0.,.3,1.],[0.,.4,1.],[0.,.6,1.]])
        P,M,Z=restriction(c,f);E=prolongation(c,f)
        np.testing.assert_allclose(P@np.ones(f.cells),1)
        np.testing.assert_allclose(M@f.V0,c.V0)
        np.testing.assert_allclose(c.B@Z,M@f.B)
        np.testing.assert_allclose(Z@E,np.eye(c.nflux),atol=1e-14)
        z=(1+2*c.centres[np.arange(c.nflux),c.axes])*c.areas
        target=(1+2*f.centres[np.arange(f.nflux),f.axes])*f.areas
        np.testing.assert_allclose(E@z,target,atol=1e-14)

if __name__=='__main__':unittest.main()
