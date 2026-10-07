"""Independent checks for modal attribution and exact-time controls."""
import unittest
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v17_modes import ModalModel,controlled_case,snapshot_model,oscillator,trig_average
from engine.aniso_phase1.unresolved_velocity import pack

class ModalDiagnosticsTests(unittest.TestCase):
    def test_exact_oscillator_and_midpoint_amplitude(self):
        w=np.array([7.,500.]);eq=np.array([.2,-.4]);v=np.array([.3,.6]);dt=.001;t=np.arange(301)*dt
        q=oscillator(w,eq,v,t,dt);qn=np.zeros(2);vn=v.copy();direct=[qn.copy()]
        for _ in t[1:]:
            mid=(vn-.5*dt*w*w*(qn-eq))/(1+.25*dt*dt*w*w)
            qn=qn+dt*mid;vn=2*mid-vn;direct.append(qn.copy())
        np.testing.assert_allclose(q.T,np.array(direct),rtol=1e-11,atol=1e-12)
        exact=oscillator(w,eq,v,t)
        np.testing.assert_allclose(exact[:,0],0,atol=1e-15)
        dense=np.linspace(0,.3,100001);qosc=oscillator(w,eq,v,dense)-eq[:,None]
        integral=np.trapezoid(qosc*qosc,dense,axis=1)/.3
        np.testing.assert_allclose(integral,trig_average(w,eq,v,.3),rtol=1e-7)

    def test_stress_linearization_and_snapshot_initial_projection(self):
        model=snapshot_model(1.6);e=model.e;s=model.s
        self.assertLess(model.eigen_residual,1e-6);self.assertEqual(model.null_count,0)
        i=int(model.order[0]);p=model.phi[:,i];dy=model.Q@p.reshape(3,model.Q.shape[1]).T
        eps=1e-7/max(la.norm(dy),1.)
        fd=(e.evaluate(s.Y+eps*dy)['P']-e.evaluate(s.Y-eps*dy)['P'])/(2*eps)
        self.assertLess(la.norm(fd-model.D[i])/la.norm(model.D[i]),1e-6)
        q,v=model.project(s.Y,pack(s.v,s.C))
        np.testing.assert_allclose(q,0,atol=1e-12);np.testing.assert_allclose(v,model.v0,rtol=1e-8,atol=1e-12)
        np.testing.assert_allclose(model.stress([0])[0],model.P0,atol=1e-12)

    def test_singular_rest_uses_constraints_without_added_mass(self):
        s,e,m,h,meta=controlled_case();model=ModalModel(s,e,m,h,meta['particle_reference'],meta['carrier_reference'])
        self.assertEqual(model.null_count,9);self.assertTrue(model.gate['passed'])
        self.assertLess(model.eigen_residual,1e-9)
        np.testing.assert_allclose(model.null_basis.T@model.K@model.phi,0,atol=1e-9)
        self.assertLess(la.norm(model.static_shift),1e-10)
        self.assertLess(la.norm(model.M@model.null_basis),1e-15)
        self.assertTrue(np.all(la.eigvalsh(model.null_basis.T@model.K@model.null_basis)>0))

    def test_modal_energy_and_zone_partition(self):
        model=snapshot_model(1.6);r=model.record();self.assertEqual(len(r['modes']),225)
        for m in r['modes']:
            self.assertAlmostEqual(m['modal_mass'],1.,places=6)
            self.assertAlmostEqual(m['kinetic_translation_fraction']+m['kinetic_C_fraction'],1.,places=12)
            for key in ('material_zones','stress_zones','kinetic_zones','stabilization_row_zones'):
                self.assertAlmostEqual(sum(m[key].values()),1.,places=8)
        self.assertEqual(r['modes'][r['stress_ranking'][0]]['stress_rank'],1)

if __name__=='__main__':unittest.main()
