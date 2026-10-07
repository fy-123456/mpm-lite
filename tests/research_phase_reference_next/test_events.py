import copy,unittest
import numpy as np
from benchmarks.research_phase_reference_next.events import events,difference,roundoff_only
class EventTests(unittest.TestCase):
    def test_roundoff_boundary_and_true_excess(self):
        def signal(lo,hi):return dict(resolved=True,events=[dict(kind='maximum',time_s=(lo+hi)/2,bracket_s=[lo,hi])],raw_times=[lo,hi])
        a=signal(1.,1.0031250000000002);b=signal(1.,1.)
        self.assertEqual(roundoff_only(a,b,.003125)['status'],'passed_scoped')
        a['events'][0]['bracket_s'][1]+=1e-8
        self.assertEqual(roundoff_only(a,b,.003125)['status'],'unresolved_or_shifted')
    def test_known_fine_shift_and_repeated_cycles(self):
        t=np.linspace(.0004,.0129,129);a=events(t,np.sin(2*np.pi*t/.0028));b=events(t,np.sin(2*np.pi*(t-.00003)/.0028))
        d=difference(a,b);self.assertEqual(d['status'],'passed_scoped')
    def test_missing_peak_is_not_deleted(self):
        t=np.linspace(.0004,.0129,129);a=events(t,np.sin(2*np.pi*t/.0028));b=copy.deepcopy(a);b['events'].pop(0)
        self.assertFalse(difference(a,b)['same_event_counts']);self.assertNotEqual(difference(a,b)['status'],'passed_scoped')
    def test_wide_aliased_brackets_remain_ambiguous(self):
        t=np.linspace(.0004,.0129,129);a=events(t,np.sin(2*np.pi*t/.0028));b=copy.deepcopy(a)
        for x in a['events']:x['bracket_s']=[0.,.0125]
        self.assertFalse(difference(a,b)['unambiguous'])
    def test_near_constant_and_invalid_budget(self):
        a=events([0.,.1,.2],[1e-17,0.,-1e-17]);self.assertNotEqual(difference(a,a)['status'],'passed_scoped')
        with self.assertRaises(ValueError):difference(a,a,0.)
if __name__=='__main__':unittest.main()
