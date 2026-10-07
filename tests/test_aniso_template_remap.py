import unittest
import numpy as np
from engine.aniso_phase1.consistent_transfer import MaterialQ1,bent_nodes
from engine.aniso_phase1.template_remap import RemappedQ1,axes_for,audit_remap,audit_mode


class TemplateRemapTests(unittest.TestCase):
    def setUp(self):
        self.old=MaterialQ1(17,ppc_axis=4)
        self.x=bent_nodes(self.old)
        self.xp=self.old.particles.N@self.x
        self.F=self.old.gradient(self.old.particles,self.x)

    def test_identity_and_true_nested_change_preserve_history(self):
        for kind in ('identity','refine_x'):
            new=RemappedQ1(self.old,axes_for(self.old,kind))
            x,r=audit_remap(self.old,new,self.x,self.xp,self.F)
            self.assertTrue(r['dynamic_accepted'])
            self.assertLess(abs(r['remap_energy_relative']),1e-9)
            self.assertLess(r['stress_relative'],1e-8)
            np.testing.assert_allclose(new.particles.N@x,self.xp,atol=1e-12)
            np.testing.assert_array_equal(new.particles.A,self.old.particles.A)
            self.assertAlmostEqual(new.mass.sum(),self.old.mass.sum(),places=15)
            if kind=='identity':
                self.assertAlmostEqual(new.mass_eigenvalues[0]/self.old.mass_eigenvalues[0],1.,places=7)
        self.assertGreater(len(new.X),len(self.old.X))

    def test_shifted_template_is_rejected_without_mutating_particles(self):
        before=self.F.copy();positions=self.xp.copy()
        new=RemappedQ1(self.old,axes_for(self.old,'shift'))
        _,r=audit_remap(self.old,new,self.x,self.xp,self.F)
        self.assertFalse(r['dynamic_accepted'])
        self.assertGreater(r['stress_relative'],1e-3)
        self.assertGreater(r['absolute_local_energy_change_relative'],10*abs(r['remap_energy_relative']))
        np.testing.assert_array_equal(self.F,before);np.testing.assert_array_equal(self.xp,positions)

    def test_unresolved_mass_is_rejected(self):
        old=MaterialQ1(17,ppc_axis=2)
        with self.assertRaisesRegex(ValueError,'rank deficient'):
            RemappedQ1(old,axes_for(old,'refine_x'))

    def test_resolved_modes_retained_by_nested_template(self):
        new=RemappedQ1(self.old,axes_for(self.old,'refine_x'))
        rng=np.random.default_rng(21)
        p=rng.normal(size=self.x.shape);p[self.old.fixed]=0
        self.assertLess(audit_mode(self.old,new,p)['increment_relative'],1e-10)

    def test_accepted_template_can_really_advance_and_keep_budget(self):
        new=RemappedQ1(self.old,axes_for(self.old,'refine_x'))
        xp,F,v,r=new.step(self.xp,self.F,np.zeros_like(self.xp),.0005)
        self.assertGreater(r['min_det'],.9)
        self.assertLess(abs(r['energy_budget_residual']),1e-16)
        self.assertLess(abs(r['history_energy_relative']),1e-8)

    def test_nested_remap_after_exact_pose_change(self):
        new=RemappedQ1(self.old,axes_for(self.old,'refine_x'))
        a=.6;Q=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        x=(self.x-.5)@Q.T+.5+np.array([.14,.03,0])
        xp=(self.xp-.5)@Q.T+.5+np.array([.14,.03,0])
        _,r=audit_remap(self.old,new,x,xp,Q@self.F)
        self.assertTrue(r['dynamic_accepted'])
        self.assertLess(r['current_direction_tensor_relative'],1e-9)
        self.assertLess(abs(self.old.elastic(x)[0]/self.old.elastic(self.x)[0]-1),1e-9)


if __name__=='__main__':unittest.main()
