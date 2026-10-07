"""Separate immutable initial-condition identity from inherited equation identity."""
import copy
import numpy as np
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_d.common_state import StateTransaction

CASES={'Y64':('y',1.,0.),'Y128':('yz',1.,0.),'YZ128':('yz',1.,.5),'LEGACY128':('yz',0.,0.)}
TAG='transverse_initial_contract'

def pressure_profile(case,topology):
    if case not in CASES:raise ValueError('unregistered initial case')
    grid,ay,az=CASES[case]
    expected=(32,2,1) if grid=='y' else (32,2,2)
    if tuple(topology.shape)!=expected:raise ValueError('case grid differs')
    centres=np.mean(np.asarray(topology.cell_bounds),axis=2)
    eta=2*(centres[:,1:]-0.5)/.25
    p=.2*(1+.2*(ay*eta[:,0]+az*eta[:,1]));p.setflags(write=False)
    definition=dict(case_id=case,shape=list(topology.shape),grid=grid,
        continuous='0.2*(1+0.2*(ay*eta_y+az*eta_z)); eta=2*(coordinate-.5)/.25',
        ay=ay,az=az,projection='exact reference cell-volume average of affine field',
        cuts=[np.asarray(x).tolist() for x in topology.cuts],pressure_Pa=p.tolist(),
        pressure_sha256=digest(p.tolist()),grid_sha256=digest([np.asarray(x).tolist() for x in topology.cuts]))
    return p,definition

def contract_for(definition,state):
    # No full-state digest here: the completed tagged initial digest is recorded
    # later, avoiding a recursive hash. All physical t=0 fields are bound here.
    return dict(schema='transverse-initial-v1',**copy.deepcopy(definition),
        q0_sha256=digest(state.q.tolist()),v0_sha256=digest(state.velocity.tolist()),
        initial_physical_digest=state.digest(),q0_source='authenticated selected-space rest',
        initial_flux_semantics='zero means no accepted interval, not a Darcy equilibrium')

class InitialCoupling:
    def __init__(self,coupling,definition):
        self.inner=coupling;self.core=coupling.core
        original=coupling.state;coupling.validate(original)
        self._contract=contract_for(definition,original);self.contract_id=digest(self._contract)
        self.identity=dict(schema='transverse-initial-coupling-v1',equations=coupling.identity,
                           initial_contract=self._contract,initial_contract_sha256=self.contract_id)
        owned=original.clone();owned.child_states[TAG]=self.contract_id
        self.initial_digest=owned.digest();self.initial=owned.clone()
        self.core._transaction=StateTransaction(owned,validator=self.validate)

    def __getattr__(self,name):return getattr(self.inner,name)

    def validate(self,state):
        if state.child_states.get(TAG)!=self.contract_id:raise ValueError('foreign initial-condition history')
        self.inner.validate(state)
        if state.step==0 and state.digest()!=self.initial_digest:raise ValueError('initial payload differs from bound contract')

    def step(self,dt=None,*,inject=None,external_force=None):
        self.validate(self.state)
        def guard(where,state):
            if inject:inject(where,state)
            self.validate(state)
        return self.inner.step(dt,inject=guard,external_force=external_force)

    def restore(self,state):
        self.validate(state)
        self.core._transaction=StateTransaction(state,validator=self.validate)
        self.geometry.cache.clear()

    def bridge_legacy(self,state,expected_digest):
        if self._contract['case_id']!='LEGACY128' or state.digest()!=expected_digest or TAG in state.child_states:
            raise ValueError('explicit authenticated legacy bridge required')
        self.inner.validate(state);owned=state.clone();owned.child_states[TAG]=self.contract_id
        self.restore(owned);return owned
