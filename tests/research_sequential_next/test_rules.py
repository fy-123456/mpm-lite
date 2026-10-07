"""Material retries must preserve all physical state, rule provenance and mass."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np
from .test_runtime import ToyModel
from benchmarks.research_sequential_next.config import protocol
from engine.aniso_phase1.research_sequential_next.rules import FixedRuleRetry
from engine.aniso_phase1.research_sequential_next.integrator import StepRejected


class RuleToy(ToyModel):
    def __init__(self,order):
        super().__init__();self.rule=SimpleNamespace(signature='q'+str(order),orders=(order,))
        self.identity=dict(model='same-space',boundary='same-boundary',device='cpu',dtype='float64',material=self.rule.signature)
        self.signature='model-'+self.rule.signature
    def rest(self):
        s=super().rest();s.child_states['identity']=copy.deepcopy(self.identity);return s
    def validate(self,state,material=False):
        super().validate(state,material=material)
        if state.child_states.get('identity')!=self.identity:raise ValueError('wrong toy rule')


class RuleTests(unittest.TestCase):
    def test_failure_rolls_back_then_one_full_rule_commit(self):
        a,b=RuleToy(5),RuleToy(7);cfg=protocol(1.,device='cpu');control=FixedRuleRetry(a,b,cfg)
        initial=control.state
        def children(candidate,values):values['E']={'p':[1.,2.],'time':candidate.time};return values
        def fail(attempt,where,state):
            if attempt==0 and where=='after_prepare':raise ValueError('compressed trial failure')
        row=control.step(.025,prepare_children=children,inject=fail)
        self.assertEqual(control.state.step,1);self.assertEqual(control.state.child_states['identity'],b.identity)
        self.assertEqual(row['material_attempts'],2);self.assertEqual(row['rule_proposal']['parent_state_sha256'],initial.digest())
        self.assertEqual(control.state.child_states['E']['time'],control.state.time)
        second=control.step(.025);self.assertEqual(second['material_attempts'],1)

    def test_double_failure_keeps_source_state_and_predictor(self):
        a,b=RuleToy(5),RuleToy(7);c=FixedRuleRetry(a,b,protocol(1.,device='cpu'));before=c.state.digest()
        def fail(attempt,where,state):
            if where=='after_prepare':raise ValueError('both attempts fail')
        with self.assertRaises(StepRejected):c.step(.025,inject=fail)
        self.assertEqual(c.state.digest(),before);self.assertEqual(len(c.failures),2)
        b.M=b.M.copy();b.M[0,0]+=.1
        with self.assertRaises(ValueError):FixedRuleRetry(a,b,protocol(1.,device='cpu'))


if __name__=='__main__':unittest.main()
