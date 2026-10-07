import os
import tempfile
import unittest
from pathlib import Path
import numpy as np

from engine.aniso_phase1.directional_reference import directional_geometry
from engine.aniso_phase1.convergence_reference import (geometry,knots,union_knots,tensor_rule,
    sites,state_from_field,save_fields,load_fields)
from engine.aniso_phase1.history_increment import HistoryField,ReferenceBasis,frozen
from engine.aniso_phase1.consistent_transfer import bent_nodes
from engine.aniso_phase1.refinement import TensorReferenceQ1,CachedFrozenQ1


class DirectionalTests(unittest.TestCase):
    def test_direction_effect_time_matches_known_ten_percent_change(self):
        from benchmarks.aniso_directional_reference import effect_time
        source=directional_geometry((4,2,2));basis=ReferenceBasis(source)
        dx=np.zeros((3,3));dx[0,0]=.03
        ds=np.zeros((3,3));ds[0,1]=.04
        groups=[]
        for scale in (1.,1.1):
            group={}
            for name,offset in dict(base=np.zeros((3,3)),axial=dx,cross=ds,both=dx+ds).items():
                F=np.eye(3)+scale*offset
                group[name]=dict(position=HistoryField(((basis,frozen(source.X@F.T)),)),
                                 velocity=HistoryField(((basis,frozen(source.X*0)),)))
            groups.append(group)
        with tempfile.TemporaryDirectory() as tmp:r=effect_time(Path(tmp),*groups,1e-7)
        for name in ('axial_coarse','cross_coarse','axial_fine','cross_fine','joint'):
            self.assertAlmostEqual(r['effects'][name]['relative']['F'],.1,delta=1e-12)
            self.assertFalse(r['passed'][name])

    def test_factorial_statistics_match_affine_norm_and_uniform_region_shares(self):
        from unittest.mock import patch
        from benchmarks.aniso_directional_reference import factorial
        source=directional_geometry((4,2,2));basis=ReferenceBasis(source)
        dx=np.zeros((3,3));dx[0,0]=.03
        ds=np.zeros((3,3));ds[0,1]=.04
        states={}
        for name,F in dict(base=np.eye(3),axial=np.eye(3)+dx,cross=np.eye(3)+ds,both=np.eye(3)+dx+ds).items():
            states[name]=dict(position=HistoryField(((basis,frozen(source.X@F.T)),)),
                              velocity=HistoryField(((basis,frozen(source.X*0)),)))
        pair=[states['base'],dict(position=states['both']['position'],
                                 velocity=HistoryField(((basis,frozen(source.X*.01)),)))]
        with tempfile.TemporaryDirectory() as tmp,patch('benchmarks.aniso_directional_reference.candidate_fields',return_value={'test':pair}):
            r=factorial(Path(tmp),states,1e-7)
        self.assertAlmostEqual(r['effects']['joint']['rms']['F'],.05,delta=1e-13)
        self.assertLess(r['effects']['interaction']['rms']['F'],1e-13)
        self.assertGreater(r['effects']['interaction']['rms']['P'],1e-4)
        self.assertAlmostEqual(r['axial_cross_correlation']['F'],0,delta=1e-12)
        self.assertIsNone(r['axial_cross_correlation']['v'])
        for row in r['effects']['joint']['regions'].values():
            self.assertAlmostEqual(row['squared_share']['F'],row['volume_fraction'],delta=1e-12)
        for p in r['effects']['joint']['profiles']['F']:
            np.testing.assert_allclose(p['rms'],.05,atol=1e-13)

    def test_additive_deformations_can_have_nonlinear_stress_interaction(self):
        from benchmarks.aniso_directional_reference import evaluated,factorial_differences
        source=directional_geometry((4,2,2));basis=ReferenceBasis(source)
        X,_=tensor_rule(knots(source),2)
        dx=np.zeros((3,3));dx[0,0]=.03
        ds=np.zeros((3,3));ds[0,1]=.04
        values={}
        for name,F in dict(base=np.eye(3),axial=np.eye(3)+dx,cross=np.eye(3)+ds,both=np.eye(3)+dx+ds).items():
            f=dict(position=HistoryField(((basis,frozen(source.X@F.T)),)),
                   velocity=HistoryField(((basis,frozen(source.X*0)),)))
            values[name]=evaluated(f,X)
        d=factorial_differences(values)
        np.testing.assert_allclose(d['interaction']['F'],0,atol=2e-14)
        self.assertGreater(np.linalg.norm(d['interaction']['P']),1e-4)
        for k in ('x','F','P','v'):
            np.testing.assert_allclose(d['joint'][k],d['axial_coarse'][k]+d['cross_coarse'][k]+d['interaction'][k],atol=2e-14)

    def test_affine_gradient_mass_and_serialization_on_rectangular_cells(self):
        source=directional_geometry((12,3,2));basis=ReferenceBasis(source)
        X,w=tensor_rule(knots(source),3);q=sites(X,w)
        A=np.array([[1.02,.03,.01],[.02,.99,-.03],[0,.01,1.01]])
        b=np.array([.1,-.2,.3]);field=HistoryField(((basis,frozen(source.X@A.T+b)),))
        x,F=field.evaluate(X)
        np.testing.assert_allclose(x,X@A.T+b,atol=1e-14)
        np.testing.assert_allclose(F,np.broadcast_to(A,F.shape),atol=1e-13)
        direct=source.sample(X,w)
        np.testing.assert_allclose(direct.N@source.X,X,atol=1e-14)
        self.assertAlmostEqual(float(source.rule(2).weight.sum()),.5*.125**2,delta=1e-16)
        state=state_from_field(field,q,q);model=TensorReferenceQ1(source,state)
        dense=CachedFrozenQ1(source,state,knots(source),enrich=False)
        np.testing.assert_allclose(model.M.toarray(),dense.M.toarray(),atol=1e-17)
        rhs=np.random.default_rng(982).normal(size=(model.R.shape[1],3))
        np.testing.assert_allclose(model.mass_solve(rhs),dense.mass_solve(rhs),rtol=1e-10,atol=1e-8)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'state.npz';save_fields(p,position=field,velocity=field)
            recovered=load_fields(p)['position'].evaluate(X)
            np.testing.assert_allclose(recovered[0],x,atol=1e-15)
            np.testing.assert_allclose(recovered[1],F,atol=1e-15)
        model.close()

    def test_original_history_and_uniform_geometry_are_preserved(self):
        original=geometry(17);field=HistoryField(((ReferenceBasis(original),frozen(bent_nodes(original))),))
        source=directional_geometry((12,3,2));axes=union_knots(knots(original),knots(source))
        q=sites(*tensor_rule(axes,3));state=state_from_field(field,q,q)
        model=TensorReferenceQ1(source,state)
        for before,after in zip(field.evaluate(q.X),(state.xq,state.Fq)):
            np.testing.assert_array_equal(before,after)
        x=geometry(33);y=directional_geometry(tuple(x.counts))
        X,w=tensor_rule(knots(x),2)
        a=ReferenceBasis(x).sample(X,w);b=ReferenceBasis(y).sample(X,w)
        np.testing.assert_allclose(a.N.toarray(),b.N.toarray(),atol=1e-14)
        for da,db in zip(a.D,b.D):np.testing.assert_allclose(da.toarray(),db.toarray(),atol=1e-12)
        for counts in ((0,2,2),(2.5,2,2),(2,2)):
            with self.assertRaises(ValueError):directional_geometry(counts)
        model.close()

    @unittest.skipUnless(os.environ.get('ANISO_TEST_CUDA'),'optional real CUDA audit')
    def test_rectangular_gpu_force_and_trajectory_match_cpu(self):
        import warp as wp
        from engine.aniso_phase1.resident_reference import ResidentReference
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        old=geometry(17);source=directional_geometry((12,3,2))
        initial=dict(position=HistoryField(((ReferenceBasis(old),frozen(bent_nodes(old))),)),
                     velocity=HistoryField(((ReferenceBasis(old),frozen(old.X*0)),)))
        axes=union_knots(knots(old),knots(source))
        p=sites(*tensor_rule(axes,2));q=sites(*tensor_rule(axes,5))
        state=state_from_field(initial['position'],p,q)
        cpu=TensorReferenceQ1(source,state);gpu=ResidentReference(source,initial,p,q)
        du=np.random.default_rng(185).normal(size=source.X.shape)*1e-5;du[source.fixed]=0
        U,f=cpu.elastic(state,du);Ug,fg=gpu.elastic(du)
        self.assertAlmostEqual(U,Ug,delta=1e-16)
        np.testing.assert_allclose(f,fg,atol=1e-13,rtol=1e-10)
        vp=np.zeros_like(state.xp)
        for _ in range(8):
            state,vp,_=cpu.step(state,vp,.00003125);gpu.step(.00003125)
        x,F=gpu.fields()['position'].evaluate(q.X)
        np.testing.assert_allclose(x,state.xq,atol=2e-12)
        np.testing.assert_allclose(F,state.Fq,atol=2e-11)
        np.testing.assert_allclose(gpu.fields()['velocity'].evaluate(p.X)[0],vp,atol=2e-10)
        cpu.close();gpu.close()


if __name__=='__main__':unittest.main()
