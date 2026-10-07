"""Check time-error norms and phase signs against analytic oscillators."""
import unittest
from types import SimpleNamespace
import numpy as np
from benchmarks.aniso_v17_analysis import comparison,phase_amplitude,time_rms

class TimeMetricTests(unittest.TestCase):
    def test_tensor_volume_norm_and_terminal_difference(self):
        t=np.linspace(0,1,1001);V=np.array([1.,3.]);P=np.zeros((len(t),2,3,3));P[:,:,0,0]=2
        Q=P.copy();Q[:,0,1,2]=t
        # Trapezoidal quadrature on t^2 has a known positive discretization error.
        actual=np.sqrt(np.trapezoid(.25*t*t,t))
        self.assertAlmostEqual(time_rms(Q-P,V,t),actual,places=14)
        c=comparison(Q,P,V,t);self.assertAlmostEqual(c['relative'],actual/2,places=14)
        self.assertAlmostEqual(c['terminal_relative'],.25,places=14)

    def test_amplitude_phase_separation_without_alignment(self):
        w=100.;dt=.001;t=np.arange(101)*dt;rate=2*np.arctan(w*dt/2)/dt
        a=SimpleNamespace(omega=np.array([w]),eq=np.array([-.1]),v0=np.array([.2]),score=np.array([1.]),order=np.array([0]))
        q=-.1+.1*np.cos(rate*t)+.002*np.sin(rate*t)
        v=-.1*w*np.sin(rate*t)+.2*np.cos(rate*t)
        r=phase_amplitude(a,dict(time=t,modal_q=q[:,None],modal_v=v[:,None]),dt)[0]
        self.assertLess(r['amplitude_max_relative_error'],1e-14)
        self.assertAlmostEqual(r['terminal_phase_error_rad'],(rate-w)*t[-1],places=12)
        self.assertAlmostEqual(r['terminal_phase_error_rad'],r['midpoint_predicted_terminal_phase_error_rad'],places=12)
        self.assertLess(r['terminal_phase_error_rad'],0.)

if __name__=='__main__':unittest.main()
