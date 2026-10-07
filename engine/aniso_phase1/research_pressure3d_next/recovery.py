"""Restore only previously device-validated commits without device allocation.

Pending restoration is not permission to advance. A full device validation and
the original transaction validator must be reinstated before the next step.
"""
from engine.aniso_phase1.research_d.common_state import StateTransaction


class SafePublication:
    def __init__(self,coupling,store):
        self.c=coupling;self.store=store;self.pending=False;self.status='ready_to_advance'
        record=store.load(validator=coupling.validate)
        if record is None or record['state'].digest()!=coupling.state.digest():raise ValueError('authenticated current commit required')
        store.history();self.authorized={record['state'].digest():record['state'].clone()};self.last=record

    def _restore_known(self,record):
        state=record['state'];key=state.digest()
        if key not in self.authorized or self.authorized[key].to_dict()!=state.to_dict():
            raise ValueError('restoration requires an unchanged previously validated state')
        self.pending=True;self.status='restored_pending_device_validation';self.c.geometry.cache.clear()
        def guard(candidate):
            if self.pending:
                if candidate.digest()!=key:raise ValueError('pending restoration cannot commit a new state')
            else:self.c.validate(candidate)
        # StateTransaction still checks owned finite arrays, shape and histories.
        self.c.core._transaction=StateTransaction(state,validator=guard);self.last=record

    def prepare(self):
        record=self.store.load()
        if record is None or record['state'].digest()!=self.last['state'].digest():raise ValueError('published state changed outside this owner')
        if self.c.state.digest()!=record['state'].digest():raise ValueError('in-memory state differs from authenticated commit')
        if self.pending:
            self.c.validate(record['state'])
            transaction=StateTransaction(record['state'],validator=self.c.validate)
            self.c.core._transaction=transaction;self.pending=False;self.status='ready_to_advance'
        return record

    def advance(self,*,inject_step=None,inject_store=None):
        old=self.prepare();candidate=None;row=None
        try:
            row=self.c.step(inject=inject_step);candidate=self.c.state
            # c.step returned only after full physical commit validation.
            self.authorized[candidate.digest()]=candidate.clone()
            self.store.save(candidate,[*old['rows'],row],inject=inject_store)
        except Exception:
            record=self.store.load()
            if record is None:raise ValueError('lost authenticated persistent commit')
            self._restore_known(record)
            if candidate is not None and record['state'].digest()==candidate.digest() and candidate.step==old['state'].step+1:
                return record['rows'][-1]
            raise
        self.last=self.store.load();self.authorized={candidate.digest():candidate.clone()};return row
