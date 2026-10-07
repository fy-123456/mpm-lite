import unittest
import numpy as np
from engine.aniso_phase1.convergence_reference import geometry,knots,tensor_rule
from engine.aniso_phase1.history_increment import HistoryField,ReferenceBasis,frozen
from engine.aniso_phase1.consistent_transfer import bent_nodes
from engine.aniso_phase1.smooth_history import fit_smooth_history
from engine.aniso_phase1.initial_controls import match_elastic,energy,smooth_velocity,scaled_position


class InitialControlTests(unittest.TestCase):
    def test_energy_match_preserves_gradient_and_clamp(self):
        s=geometry(17);original=HistoryField(((ReferenceBasis(s),frozen(bent_nodes(s))),))
        X,w=tensor_rule(knots(s),7)
        smooth,_=fit_smooth_history(original,X,w)
        matched,info=match_elastic(original,smooth,X,w,s.params)
        X9,w9=tensor_rule(knots(s),9)
        self.assertLess(abs(energy(matched,X9,w9,s.params)/energy(original,X9,w9,s.params)-1),1e-10)
        x,F=smooth.evaluate(X)
        xm,Fm=matched.evaluate(X)
        np.testing.assert_allclose(xm,X+info['alpha']*(x-X),atol=2e-15)
        np.testing.assert_allclose(Fm,np.eye(3)+info['alpha']*(F-np.eye(3)),atol=2e-14)
        fixed=s.X[s.fixed]
        np.testing.assert_allclose(matched.evaluate(fixed)[0],fixed,atol=1e-15)

    def test_velocity_control_matches_kinetic_energy_without_translation(self):
        s=geometry(17);v=(s.X[:,0]-.25)[:,None]*np.array([.1,.2,-.1])
        velocity=HistoryField(((ReferenceBasis(s),frozen(v)),))
        X,w=tensor_rule(knots(s),7)
        smooth,info=smooth_velocity(velocity,X,w)
        np.testing.assert_allclose(smooth.evaluate(X)[0],velocity.evaluate(X)[0],atol=1e-13)
        np.testing.assert_allclose(smooth.evaluate(s.X[s.fixed])[0],0,atol=1e-15)
        self.assertAlmostEqual(info['matched'],info['target'],delta=1e-17)


if __name__=='__main__':unittest.main()
