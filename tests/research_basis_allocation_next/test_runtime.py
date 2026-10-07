"""Risks: path adjacency, provenance, sensitive scope and event resolution."""
import copy, unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
from engine.aniso_phase1.research_basis_allocation_next.rules import contiguous_slice,permission,rule_branch
from benchmarks.research_basis_allocation_next.time_study import events,difference
from benchmarks.research_basis_allocation_next import run
from benchmarks.research_basis_allocation_next.provenance import write,read,sha
from engine.aniso_phase1.research_d.common_state import CommonState

class ScopeTests(unittest.TestCase):
    def test_adjacent_slice_rejects_skipped_nodes_even_under_max_dt(self):
        grid=[.6,.603125,.60625,.6125,.625]
        self.assertTrue(contiguous_slice(grid[1:4],grid))
        self.assertFalse(contiguous_slice([.6,.60625,.6125],grid))
        self.assertFalse(contiguous_slice([.6,.6],grid))
        self.assertFalse(contiguous_slice([.6,float('nan')],grid))
    def test_entry_scope_initial_and_source_boundaries(self):
        with TemporaryDirectory() as d,patch.object(run,'verify',return_value={'energy_scale_J':1.}),patch.object(run,'source_files',return_value={'code':'a'}):
            p=Path(d);write(p/'selected-space.json',dict(package={'path':'x','sha256':'space'},mass_order=7,full_order=7));write(p/'state.json',{})
            cert=dict(schema='basis-allocation-q5-qualification-v1',qualified=True,numerical_source_sha256={'code':'a'},
                scope=dict(physical_space_sha256='space',mass_order=7,full_order=7,peak_m=.0075,fiber_angle_degrees=45.,time_grid_s=[.6,.6125,.625],rest_start=False,
                    initial_states=[dict(time_s=.6,sha256=sha(p/'state.json'))]))
            write(p/'cert.json',cert)
            def make(name,**kw):return run.create_config(p,name,peak_m=.0075,start=.6,end=.625,initial=p/'state.json',qualification_path=p/'cert.json',rule_policy='q5_with_full_retry',**kw)
            self.assertEqual(make('okay')['post_release']['rule_policy'],'q5_with_full_retry')
            self.assertEqual(make('skipped',times=[.6,.625])['post_release']['rule_policy'],'full_only')
            cert['numerical_source_sha256']={'code':'changed'};write(p/'cert.json',cert)
            self.assertEqual(make('source')['post_release']['rule_policy'],'full_only')
            cert['numerical_source_sha256']={'code':'a'};write(p/'cert.json',cert);write(p/'state.json',{'different':1})
            self.assertEqual(make('initial')['post_release']['rule_policy'],'full_only')
    def test_fixed_rule_window_builds_full_model_for_authenticated_initial(self):
        from types import SimpleNamespace
        with TemporaryDirectory() as d:
            path=Path(d)/'cert.json';full=SimpleNamespace(identity={'full':'model'});write(path,dict(full_model=full.identity))
            cfg={'post_release':{'rule_policy':'q5_fixed','full_order':7,'qualification':{'path':str(path),'sha256':sha(path)}}}
            with patch.object(run,'load_model',return_value=(full,{})),patch.object(run,'ValidatedAVF',return_value=SimpleNamespace()):
                c=run.make_stepper(Path(d),cfg,SimpleNamespace(),None)
                self.assertIs(c.full,full)
    def test_branch_preserves_owned_history_and_accounts_energy(self):
        class Model:
            M=np.eye(1)
            def __init__(self,order):self.identity={'model':'same','material':str(order)};self.order=order
            def validate(self,s,material=False):
                if s.child_states['identity']!=self.identity:raise ValueError('foreign model')
            def evaluate(self,q):return {'U':float(np.sum(q*q))+self.order*1e-7}
        full,compact=Model(7),Model(5);s=CommonState(np.array([[.1,0.,0.]]),np.zeros((1,3)),predictor=np.ones((1,3)),child_states={'identity':full.identity,'fluid':{'pressure':[.2]}});before=s.digest()
        cfg={'acceptance':{'energy_fraction':.01,'energy_scale_J':1.}}
        result=rule_branch(s,compact,full,cfg,{'sha256':'source'})
        self.assertEqual(s.digest(),before);np.testing.assert_array_equal(result.predictor,s.predictor)
        self.assertEqual(result.child_states['fluid'],s.child_states['fluid']);self.assertAlmostEqual(result.child_states['cumulative_abs_rule_switch_error_J'],2e-7)
        result.child_states['fluid']['pressure'][0]=9;self.assertEqual(s.child_states['fluid']['pressure'][0],.2)
        compact.M=np.eye(1)*2
        with self.assertRaisesRegex(ValueError,'mass'):rule_branch(s,compact,full,cfg,{})

class EventTests(unittest.TestCase):
    def test_exact_zero_single_crossing(self):
        e=events([0.,1.,2.,3.,4.],[-1.,-.5,0.,.5,1.]);self.assertEqual(len(e['events']),1);self.assertEqual(e['events'][0]['kind'],'up_zero')
    def test_near_zero_is_unresolved(self):
        e=events([0.,1.,2.],[1e-16,0.,-1e-16]);self.assertFalse(e['resolved']);self.assertNotEqual(difference(e,e)['status'],'passed_scoped')
    def test_coarse_identical_peaks_do_not_prove_subsample_accuracy(self):
        e=events([0.,.02,.04],[0.,1.,0.]);d=difference(e,e);self.assertEqual(d['status'],'unresolved_or_shifted');self.assertEqual(d['pairs'][0]['offset_estimate_s'],0.)
    def test_fine_known_peak_brackets(self):
        t=np.linspace(.001,.099,99);a=events(t,np.sin(2*np.pi*t/.04));d=difference(a,a)
        self.assertEqual(d['status'],'passed_scoped')
if __name__=='__main__':unittest.main()
