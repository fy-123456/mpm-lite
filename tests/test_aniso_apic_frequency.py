import unittest
import numpy as np
from benchmarks.aniso_apic_frequency import Probe, initial_field, run_case, SOURCE


class ApicFrequencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(SOURCE) as z:cls.probe=Probe(z['particle_x_before'],z['particle_F_before'])

    def test_affine_and_constant_full_chain(self):
        for field in ('constant','affine'):
            rows,_=run_case(self.probe,field,'coupled',.9,3)
            for r in rows:
                self.assertLess(r['velocity_relative_change'],1e-12)
                self.assertLess(r['gradient_relative_change'],1e-12)
                self.assertEqual(r['x_error'],0.);self.assertEqual(r['F_error'],0.)

    def test_independent_stage_oracle_and_holds(self):
        for mode in ('coupled','velocity_only','gradient_only'):
            for beta in (0.,.9,1.):
                rows,_=run_case(self.probe,'sine_45',mode,beta,1);r=rows[0]
                self.assertLess(max(r['oracle_errors'].values()),1e-12)
                self.assertLess(r['stage_momentum_error'],1e-14)
                self.assertLess(r['candidate_momentum_error'],1e-14)
                self.assertEqual(r['held_error'],0.)
                self.assertGreater(r['candidate_gradient_increment_rms'],1e-5)

    def test_gradient_return_independent_of_beta_on_same_input(self):
        outputs=[]
        for beta in (0.,.9,1.):
            _,states=run_case(self.probe,'sine_x2','coupled',beta,1);outputs.append(states)
        np.testing.assert_array_equal(outputs[0]['G_1'],outputs[1]['G_1'])
        np.testing.assert_array_equal(outputs[1]['G_1'],outputs[2]['G_1'])
        v0,_=initial_field(self.probe.x,'sine_x2')
        np.testing.assert_array_equal(outputs[2]['v_1'],v0)
        self.assertGreater(np.max(abs(outputs[0]['v_1']-v0)),1e-6)


if __name__=='__main__':unittest.main()
