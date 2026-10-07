import tempfile
import unittest
from pathlib import Path
import numpy as np

from benchmarks.aniso_axis_reference import CASES,concentration,effect_time
from benchmarks.aniso_directional_reference import location
from engine.aniso_phase1.directional_reference import directional_geometry
from engine.aniso_phase1.history_increment import ReferenceBasis,HistoryField,frozen


def affine(source,F,velocity=0.):
    b=ReferenceBasis(source)
    return dict(position=HistoryField(((b,frozen(source.X@F.T)),)),
                velocity=HistoryField(((b,frozen(np.full_like(source.X,velocity))),)))


class AxisReferenceTests(unittest.TestCase):
    def test_identical_fields_have_zero_shares_and_no_peak_location(self):
        s=directional_geometry((4,2,2));state=affine(s,np.eye(3))
        with tempfile.TemporaryDirectory() as tmp:r=location(Path(tmp),state,state,'zero',order=2)
        self.assertEqual(list(r['rms'].values()),[0.]*4)
        for p in r['profiles']['F']:self.assertEqual(sum(p['squared_share']),0.)
        self.assertEqual(concentration(r)['exploratory_peak_bands'],dict(X=None,Y=None,Z=None))

    def test_equal_cells_original_interfaces_and_affine_maps(self):
        old=directional_geometry((8,2,2))
        points=np.random.default_rng(842).uniform([.25,.4375,.4375],[.75,.5625,.5625],size=(31,3))
        F=np.array([[1.02,.03,-.02],[.01,.99,.02],[.005,-.01,1.01]])
        counts=[]
        for name,shape in CASES.items():
            s=directional_geometry(shape)
            for axis,history in zip(s.axes,old.axes):
                for x in history:self.assertLess(np.min(abs(axis-x)),1e-14)
            x,grad=affine(s,F)['position'].evaluate(points)
            np.testing.assert_allclose(x,points@F.T,atol=2e-15)
            np.testing.assert_allclose(grad,np.broadcast_to(F,grad.shape),atol=1e-12)
            if name!='base':counts.append(np.prod(shape))
        self.assertEqual(counts,[46464]*3)

    def test_profile_resolution_cache_and_uniform_concentration(self):
        s=directional_geometry((4,2,2));a=affine(s,np.eye(3));b=affine(s,np.diag([1.03,1.,1.]),.01)
        with tempfile.TemporaryDirectory() as tmp:
            for bins in (16,8):
                edges=[np.linspace(.25,.75,bins+1),np.linspace(.4375,.5625,bins+1),np.linspace(.4375,.5625,bins+1)]
                r=location(Path(tmp),a,b,'same-cache-name',order=2,edges=edges)
                self.assertEqual(len(r['profiles']['F'][1]['rms']),bins)
                self.assertAlmostEqual(r['rms']['F'],.03,delta=1e-13)
                c=concentration(r)
                self.assertAlmostEqual(c['predeclared_Y_band']['volume_fraction'],.125)
                self.assertAlmostEqual(c['predeclared_Y_band']['squared_share']['F'],.125,delta=1e-12)
                for profile in r['profiles']['F']:self.assertAlmostEqual(sum(profile['squared_share']),1.,delta=1e-12)
                for v in c['exploratory_peak_bands'].values():
                    self.assertAlmostEqual(v['volume_fraction'],.125)
                    self.assertAlmostEqual(v['squared_share']['F'],.125,delta=1e-12)

    def test_axis_effect_temporal_change_matches_affine_ten_percent(self):
        s=directional_geometry((4,2,2));base=affine(s,np.eye(3));d=np.diag([.03,0,0])
        states=[affine(s,np.eye(3)+d),base,affine(s,np.eye(3)+1.1*d),base]
        with tempfile.TemporaryDirectory() as tmp:r=effect_time(Path(tmp),'X',1e-7,states=states)
        self.assertAlmostEqual(r['relative']['F'],.1,delta=1e-12)
        self.assertAlmostEqual(r['fine_rms']['F'],.03,delta=1e-13)
        self.assertFalse(r['passed'])


if __name__=='__main__':unittest.main()
