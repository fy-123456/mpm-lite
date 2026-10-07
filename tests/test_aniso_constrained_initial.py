import unittest
import numpy as np
from engine.aniso_phase1.convergence_reference import geometry,knots,tensor_rule,directions
from engine.aniso_phase1.history_increment import HistoryField,ReferenceBasis,frozen
from engine.aniso_phase1.consistent_transfer import bent_nodes,material_response
from engine.aniso_phase1.smooth_history import fit_smooth_history
from engine.aniso_phase1.initial_controls import smooth_velocity,match_elastic,energy
from engine.aniso_phase1.constrained_initial import momentum_energy_velocity,StressFit


class ConstrainedInitialTests(unittest.TestCase):
    def test_velocity_energy_momentum_and_clamp_on_independent_rule(self):
        source=geometry(17);s=(source.X[:,0]-.25)/.5
        v=.02*np.sin(3*np.pi*s)[:,None]*np.array([1.,.3,-.2])
        v[:,1]+=.01*s*s
        original=HistoryField(((ReferenceBasis(source),frozen(v)),))
        X,w=tensor_rule(knots(source),7)
        smooth,_=smooth_velocity(original,X,w)
        result,info=momentum_energy_velocity(original,smooth,X,w)
        X,w=tensor_rule(knots(source),9)
        a=original.evaluate(X)[0];b=result.evaluate(X)[0]
        np.testing.assert_allclose(w @ a,w @ b,atol=1e-15,rtol=1e-9)
        self.assertLess(abs(np.sum(w[:,None]*b*b)/np.sum(w[:,None]*a*a)-1),1e-9)
        np.testing.assert_allclose(result.evaluate(source.X[source.fixed])[0],0,atol=1e-15)
        self.assertLess(info['minimum_kinetic'],info['target_kinetic'])

    def fixture(self):
        s=geometry(17);X,w=tensor_rule(knots(s),7)
        original=HistoryField(((ReferenceBasis(s),frozen(bent_nodes(s))),))
        smooth,_=fit_smooth_history(original,X,w)
        initial,_=match_elastic(original,smooth,X,w,s.params)
        return s,original,initial,StressFit(original,initial,X,w,s.params)

    def test_stress_objective_and_constraint_derivatives(self):
        _,_,_,problem=self.fixture()
        d=np.random.default_rng(9841).normal(size=problem.start.shape);d/=np.linalg.norm(d)
        y=problem.start+1e-4*d;epsilon=1e-6
        r=problem.evaluate(y);plus=problem.evaluate(y+epsilon*d);minus=problem.evaluate(y-epsilon*d)
        for value,gradient in [('objective','gradient'),('energy','energy_gradient'),('shape','shape_gradient'),('deformation','deformation_gradient')]:
            analytic=r[gradient] @ d;fd=(plus[value]-minus[value])/(2*epsilon)
            np.testing.assert_allclose(analytic,fd,atol=2e-7,rtol=2e-5)

    def test_stress_fit_preserves_energy_and_improves_stress(self):
        source,original,initial,problem=self.fixture()
        final,info=problem.solve()
        X,w=tensor_rule(knots(source),9);A=directions(X)
        U=energy(original,X,w,source.params)
        self.assertLess(abs(energy(final,X,w,source.params)/U-1),1e-8)
        P0=material_response(original.evaluate(X)[1],A,source.params)[1]
        error=[]
        for f in (initial,final):
            P=material_response(f.evaluate(X)[1],A,source.params)[1]
            error.append(float(np.sum(w[:,None,None]*(P-P0)**2)))
        self.assertLess(error[1],error[0])
        self.assertLessEqual(info['displacement_relative'],info['displacement_cap']+1e-8)
        self.assertLessEqual(info['deformation_relative'],info['deformation_cap']+1e-8)
        np.testing.assert_allclose(final.evaluate(source.X[source.fixed])[0],source.X[source.fixed],atol=1e-14)


if __name__=='__main__':unittest.main()
