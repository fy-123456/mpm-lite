import os
import tempfile
import unittest
from pathlib import Path
import numpy as np
from engine.aniso_phase1.directional_reference import cartesian_geometry,directional_geometry
from engine.aniso_phase1.convergence_reference import geometry,knots,union_knots,tensor_rule,sites,state_from_field,save_fields,load_fields
from engine.aniso_phase1.history_increment import HistoryField,ReferenceBasis,frozen
from engine.aniso_phase1.consistent_transfer import bent_nodes
from engine.aniso_phase1.refinement import TensorReferenceQ1,CachedFrozenQ1
from benchmarks.aniso_nonuniform_reference import source,BAND,quality


def small():
    axes=list(knots(directional_geometry((8,3,2))))
    axes[1]=np.array([.4375,.45,.5,.5625])
    return cartesian_geometry(axes)


class NonuniformTests(unittest.TestCase):
    def test_equal_budget_nested_interfaces_and_targeted_bisection(self):
        b,l,u=map(source,('base','localY','Y'))
        self.assertEqual(np.prod(l.counts),np.prod(u.counts))
        self.assertEqual(sum(l.free),sum(u.free))
        for d in (0,2):np.testing.assert_array_equal(l.axes[d],b.axes[d])
        self.assertTrue(set(b.axes[1]).issubset(l.axes[1]))
        extra=np.setdiff1d(l.axes[1],b.axes[1]);self.assertEqual(len(extra),2)
        self.assertTrue(np.all((extra>BAND[0])&(extra<BAND[1])))
        self.assertAlmostEqual(np.diff(l.axes[1]).min()/np.diff(b.axes[1]).min(),.5,delta=1e-13)
        old=directional_geometry((8,2,2))
        for s in (l,u,source('fineY')):
            for axis,history in zip(s.axes,old.axes):
                for x in history:self.assertLess(np.min(abs(axis-x)),1e-14)

    def test_axes_validate_domain_order_and_ownership(self):
        axes=[a.copy() for a in small().axes];s=cartesian_geometry(axes)
        before=s.axes[1].copy();axes[1][1]+=.001
        np.testing.assert_array_equal(s.axes[1],before)
        for bad in ([.4375,.45,.45,.5625],[.4375,np.nan,.5625],[.4,.5,.5625]):
            a=list(s.axes);a[1]=bad
            with self.assertRaises(ValueError):cartesian_geometry(a)

    def test_affine_gradient_consistent_mass_and_serialization(self):
        s=small();q=sites(*tensor_rule(knots(s),3));basis=ReferenceBasis(s)
        A=np.array([[1.02,.03,.01],[.02,.99,-.03],[0,.01,1.01]])
        field=HistoryField(((basis,frozen(s.X@A.T+[.1,-.2,.3])),))
        x,F=field.evaluate(q.X)
        np.testing.assert_allclose(x,q.X@A.T+[.1,-.2,.3],atol=1e-14)
        np.testing.assert_allclose(F,np.broadcast_to(A,F.shape),atol=1e-13)
        state=state_from_field(field,q,q);model=TensorReferenceQ1(s,state)
        dense=CachedFrozenQ1(s,state,knots(s),enrich=False)
        np.testing.assert_allclose(model.M.toarray(),dense.M.toarray(),atol=1e-17)
        rhs=np.random.default_rng(2).normal(size=(model.R.shape[1],3))
        np.testing.assert_allclose(model.mass_solve(rhs),dense.mass_solve(rhs),rtol=1e-10,atol=1e-8)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'field.npz';save_fields(p,position=field,velocity=field)
            xx,FF=load_fields(p)['position'].evaluate(q.X)
            np.testing.assert_array_equal(xx,x);np.testing.assert_array_equal(FF,F)
        model.close()

    def test_potential_gradient_tangent_and_symmetry(self):
        s=small();q=sites(*tensor_rule(knots(s),3))
        field=HistoryField(((ReferenceBasis(s),frozen(bent_nodes(s))),))
        state=state_from_field(field,q,q);m=TensorReferenceQ1(s,state)
        rng=np.random.default_rng(73);v,a,b=[rng.normal(size=s.X.shape)*.01 for _ in range(3)]
        for z in (v,a,b):z[s.fixed]=0
        pred=np.zeros_like(v);dt=.001;eps=1e-4
        energy,res=m.potential(v,state,pred,dt)
        ep,rp=m.potential(v+eps*a,state,pred,dt);em,rm=m.potential(v-eps*a,state,pred,dt)
        Ja=m.tangent(v,state,a,dt);Jb=m.tangent(v,state,b,dt)
        self.assertAlmostEqual((ep-em)/(2*eps),np.sum(res*a),delta=2e-12)
        np.testing.assert_allclose((rp-rm)/(2*eps),Ja,rtol=2e-6,atol=2e-12)
        self.assertAlmostEqual(np.sum(a*Jb),np.sum(b*Ja),delta=1e-16)
        m.close()

    def test_probe_comparison_does_not_reward_closeness_to_base(self):
        # base=0, local=1, uniform=2, probe=3: local is closer to base yet worse.
        def row(value,share):
            return dict(location=dict(rms={k:value for k in ('x','F','P','v')}),
                        concentration=dict(predeclared_Y_band=dict(squared_share={k:share for k in ('x','F','P','v')})))
        r=quality(row(2,.5),row(1,.125))
        for v in r.values():
            self.assertEqual(v['ratio'],2);self.assertEqual(v['gain'],-1)
            self.assertEqual(v['regions']['band']['ratio'],4)
            self.assertAlmostEqual(v['regions']['outside']['ratio'],2*np.sqrt(.5/.875))
        self.assertIsNone(quality(row(1,.125),row(0,0))['F']['ratio'])

    def test_unavailable_cuda_stops_queue_before_any_trajectory(self):
        import json,sys
        from unittest.mock import patch
        from benchmarks import aniso_nonuniform_reference_queue as queue
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            (out/'protocol.json').write_text(json.dumps(dict(initial=queue.fingerprint(queue.initial()),initial_passed=True,equal_cost_passed=True)))
            with patch.object(sys,'argv',['queue','--out',tmp]),patch('warp.init'),patch('warp.get_cuda_devices',return_value=[]),patch.object(queue,'run_queue') as run,patch.object(queue.subprocess,'run'):
                with self.assertRaisesRegex(RuntimeError,'CUDA devices unavailable'):queue.main()
                run.assert_not_called()
            self.assertEqual(json.loads((out/'status.json').read_text())['state'],'blocked_cuda')
            self.assertFalse(json.loads((out/'gpu-preflight.json').read_text())['passed'])

    @unittest.skipUnless(os.environ.get('ANISO_TEST_CUDA'),'optional real CUDA audit')
    def test_nonuniform_gpu_force_and_release_match_cpu(self):
        import warp as wp
        from engine.aniso_phase1.resident_reference import ResidentReference
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        old=geometry(17);s=small();basis=ReferenceBasis(old)
        initial=dict(position=HistoryField(((basis,frozen(bent_nodes(old))),)),
                     velocity=HistoryField(((basis,frozen(old.X*0)),)))
        axes=union_knots(knots(old),knots(s));p=sites(*tensor_rule(axes,2));q=sites(*tensor_rule(axes,5))
        state=state_from_field(initial['position'],p,q);cpu=TensorReferenceQ1(s,state);gpu=ResidentReference(s,initial,p,q)
        du=np.random.default_rng(185).normal(size=s.X.shape)*1e-5;du[s.fixed]=0
        U,f=cpu.elastic(state,du);Ug,fg=gpu.elastic(du)
        self.assertAlmostEqual(U,Ug,delta=1e-16);np.testing.assert_allclose(f,fg,atol=1e-13,rtol=1e-10)
        vp=np.zeros_like(state.xp)
        for _ in range(8):state,vp,_=cpu.step(state,vp,.00003125);gpu.step(.00003125)
        x,F=gpu.fields()['position'].evaluate(q.X)
        np.testing.assert_allclose(x,state.xq,atol=2e-12);np.testing.assert_allclose(F,state.Fq,atol=2e-11)
        np.testing.assert_allclose(gpu.fields()['velocity'].evaluate(p.X)[0],vp,atol=2e-10)
        cpu.close();gpu.close()


if __name__=='__main__':unittest.main()
