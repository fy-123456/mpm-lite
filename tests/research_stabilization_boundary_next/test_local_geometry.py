import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
import unittest
import numpy as np
from types import SimpleNamespace
from engine.aniso_phase1.research_stabilization_boundary_next.local_geometry import partition,own_weights
from engine.aniso_phase1.research_observable_boundary_next.rt0 import CartesianTopology

def fixture(cuts=None):
    t=CartesianTopology(cuts or [[0,.5,1],[0,1],[0,1]]);points=(np.array([.1,.4,.6,.9]),np.array([.25,.75]),np.array([.25,.75]));shape=tuple(map(len,points));X=np.stack(np.meshgrid(*points,indexing='ij'),axis=-1).reshape(-1,3)
    return SimpleNamespace(topology=t,points=points,shape=shape,count=len(X),cell_ids=t.locate(X),total_weights=np.full(len(X),1/len(X)))

class PartitionTests(unittest.TestCase):
    def test_owned_weights_survive_source_mutation(self):
        source=np.arange(5,dtype=float);owned=own_weights(source);source[:]=0
        np.testing.assert_array_equal(owned,np.arange(5,dtype=float))
        with self.assertRaises(ValueError):owned[0]=8

    def test_zero_weight_elimination_preserves_integral(self):
        g=fixture();v=np.arange(g.count,dtype=float)**2+.3
        for c,(_,_,start,stop) in enumerate(partition(g)):
            self.assertAlmostEqual(float(g.total_weights[start:stop]@v[start:stop]),float((g.total_weights*(g.cell_ids==c))@v))

    def test_foreign_cell_ownership_rejected(self):
        g=fixture();g.cell_ids[2]=1
        with self.assertRaisesRegex(ValueError,'ownership'):partition(g)

    def test_non_x_grid_rejected(self):
        g=fixture([[0,.5,1],[0,.5,1],[0,1]])
        with self.assertRaisesRegex(ValueError,'x-only'):partition(g)

    def test_uncovered_volume_rejected(self):
        g=fixture();g.total_weights[:4]*=2
        with self.assertRaisesRegex(ValueError,'volume'):partition(g)

    def test_outside_quadrature_rejected(self):
        g=fixture();g.points[0][0]=-.1
        with self.assertRaisesRegex(ValueError,'ownership'):partition(g)

if __name__=='__main__':unittest.main()
