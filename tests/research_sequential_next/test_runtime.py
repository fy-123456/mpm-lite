"""Risk-focused checks for provenance, atomic publication and physical gates."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np
import scipy.linalg as la
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.config import protocol,identity,validate
from benchmarks.research_sequential_next.compare import impulse_average,metric
from benchmarks.research_sequential_next.provenance import serial_lock
from engine.aniso_phase1.research_d.common_state import CommonState
from engine.aniso_phase1.research_c.stage2.dynamics import AVF
from engine.aniso_phase1.research_c.stage2.model import DynamicModel
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF,StepRejected


class StoreTests(unittest.TestCase):
    def test_interruptions_publish_only_whole_generations(self):
        for point in ('after_state','after_ledger','after_frame','before_pointer','after_pointer'):
            with self.subTest(point=point),TemporaryDirectory() as directory:
                store=GenerationStore(directory,{'source':'A','protocol':'P','model':'M'})
                initial=CommonState(np.zeros((2,3)),np.zeros((2,3)))
                store.save(initial,[],frame={'x':initial.q})
                trial=initial.clone();trial.q[0,0]=.01;trial.step=1;trial.time=.1
                rows=[dict(step=1,time=.1,dt=.1,reaction_N=2.)]
                def fail(where):
                    if where==point:raise OSError('simulated crash')
                with self.assertRaises(OSError):store.save(trial,rows,frame={'x':trial.q},inject=fail)
                loaded=store.load()
                expected=trial if point=='after_pointer' else initial
                self.assertEqual(loaded['state'].digest(),expected.digest())
                self.assertEqual(len(store.history()),2 if point=='after_pointer' else 1)
                if point!='after_pointer':
                    store.save(trial,rows,frame={'x':trial.q})
                    self.assertEqual(len(store.history()),2)
                with self.assertRaises(ValueError):store.save(trial,rows)

    def test_changed_protocol_source_and_corruption_rejected(self):
        with TemporaryDirectory() as directory:
            store=GenerationStore(directory,{'source':'A','protocol':'P'})
            state=CommonState(np.zeros((2,3)),np.zeros((2,3)));store.save(state,[])
            for change in ({'source':'B','protocol':'P'},{'source':'A','protocol':'Q'}):
                with self.assertRaises(ValueError):GenerationStore(directory,change).load()
            current=store.load();p=current['folder']/'state.json'
            value=json.loads(p.read_text());value['payload']['q'][0][0]=1.;p.write_text(json.dumps(value))
            with self.assertRaises(ValueError):store.load()

    def test_shared_lock_rejects_other_output(self):
        with TemporaryDirectory() as one,TemporaryDirectory() as two:
            with serial_lock(one):
                with self.assertRaises(RuntimeError):
                    with serial_lock(two):pass
            with serial_lock(two):pass


class ToyModel:
    def __init__(self):
        self.M=np.array([[2.,.2],[.2,1.]])
        self.K=np.array([[2.,-1.],[-1.,2.]])
        self.free=np.array([0]);self.fixed=np.array([1]);self.ids=np.arange(3)
        self.Mff=self.M[:1,:1];self.M3ff=np.kron(self.Mff,np.eye(3));self.mass_factor=la.cho_factor(self.Mff)
        self.rest_K=np.kron(self.K,np.eye(3));self.rule=SimpleNamespace(signature='toy')
        unit=np.zeros((2,3));unit[1,0]=1.
        self.boundary=SimpleNamespace(unit=unit,hold=0.,lift=lambda t:.005*t*t*unit,
            speed=lambda t:.01*t*unit,validate=self.boundary_validate)
        self.bad_det=False
    def boundary_validate(self,state):
        errors=[float(np.max(abs((a-b)[self.fixed]))) for a,b in
                ((state.q,self.boundary.lift(state.time)),(state.velocity,self.boundary.speed(state.time)))]
        if max(errors)>1e-8:raise ValueError('boundary')
        return errors
    def rest(self):return CommonState(np.zeros((2,3)),np.zeros((2,3)))
    def validate(self,state,material=False):
        self.boundary_validate(state)
        if not np.isfinite(state.q).all() or not np.isfinite(state.velocity).all():raise ValueError('nonfinite')
        if material and self.evaluate(state.q)['min_detF']<=.1:raise ValueError('det')
    def evaluate(self,q,direction=None):
        force=self.K@q;energy=.5*float(np.sum(q*force))
        result=dict(U=energy,material_U=energy,stabilization_U=0.,force=force,material_force=force,
                    min_detF=.05 if self.bad_det and np.max(abs(q))>0 else 1.,material_calls=1)
        if direction is not None:result['tangent_action']=self.K@direction
        return result
    def kinetic(self,v):return .5*float(np.sum(v*(self.M@v)))
    def wrench(self,generalized,q=None):return generalized.sum(axis=0),np.zeros(3)
    endpoint=DynamicModel.endpoint


class PhysicalGateTests(unittest.TestCase):
    def test_unchanged_equations_and_parent_child_rollback(self):
        model=ToyModel();cfg=protocol(1.,device='cpu',dt=.1,end=.2)
        new=ValidatedAVF(model,cfg);old=AVF(model,path_order=2,residual_atol=1e-7,residual_rtol=1e-5,ledger_atol=1e-7)
        for h in (.05,.075):
            a=new.step(h);b=old.step(h)
            np.testing.assert_allclose(new.state.q,old.state.q,rtol=0,atol=1e-13)
            np.testing.assert_allclose(new.state.velocity,old.state.velocity,rtol=0,atol=1e-13)
            self.assertAlmostEqual(a['reaction_N'],b['reaction_N'])
        before=new.state.digest()
        def child(candidate,values):values['E']={'p':[1.,2.],'time':candidate.time};return values
        def fail(where,candidate):
            if where=='before_commit':raise ValueError('failure after prepared child')
        with self.assertRaises(StepRejected):new.step(.025,prepare_children=child,inject=fail)
        self.assertEqual(new.state.digest(),before)
        new.step(.025,prepare_children=child)
        self.assertEqual(new.state.child_states['E']['time'],new.state.time)

    def test_path_det_gate_rolls_back_even_without_fault_hook(self):
        model=ToyModel();model.bad_det=True;stepper=ValidatedAVF(model,protocol(1.,device='cpu'))
        before=stepper.state.digest()
        with self.assertRaises(StepRejected):stepper.step(.05)
        self.assertEqual(before,stepper.state.digest())

    def test_protocol_identity_binds_time_and_solver(self):
        base=protocol(1.,device='cpu');a=copy.deepcopy(base);a['path_order']=3
        b=copy.deepcopy(base);b['solver']['residual_atol']=1e-6
        self.assertNotEqual(identity(base),identity(a));self.assertNotEqual(identity(base),identity(b))
        c=copy.deepcopy(base);c['times']=[0,.55,1.6]
        with self.assertRaises(ValueError):validate(c)

    def test_reaction_is_integrated_over_the_whole_interval(self):
        rows=[dict(time=.025,dt=.025,reaction_N=2.),dict(time=.1,dt=.075,reaction_N=4.)]
        self.assertAlmostEqual(impulse_average(rows,0,.1),3.5)
        with self.assertRaises(ValueError):impulse_average(rows,.01,.1)
        self.assertTrue(metric(1e-5,0.,1e-4,.05)['passed'])
        w=np.array([1.,3.]);a=np.array([[1.,0,0],[3.,0,0]])
        self.assertAlmostEqual(metric(a,np.zeros_like(a),10.,0.,w)['absolute'],np.sqrt(7.))


if __name__=='__main__':unittest.main()
