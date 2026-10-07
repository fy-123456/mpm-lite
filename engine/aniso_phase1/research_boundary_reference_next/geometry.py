"""Exact Simpson volume gradient reuses endpoint and midpoint geometry evaluations.

For fixed displacement bases F(q) is affine and cof(F(q)) is quadratic on a
segment. This changes neither det(F), current H, material quadrature nor state.
"""
import copy
from collections import OrderedDict
import numpy as np
from engine.aniso_phase1.research_pressure_startup_next.reduced_geometry import ReducedGeometry
from engine.aniso_phase1.research_restoring_rt0_next.fixture import physical_payload
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_d.common_state import StateTransaction

class SharedMidpointGeometry(ReducedGeometry):
    @classmethod
    def from_reduced(cls,original):
        if not isinstance(original,ReducedGeometry):raise ValueError('authenticated bounded geometry required')
        obj=cls.__new__(cls);obj.__dict__=original.__dict__.copy();obj.cache=OrderedDict()
        obj.identity=dict(schema='shared-midpoint-exact-volume-gradient-v1',parent=copy.deepcopy(original.identity),discrete_gradient='Simpson exact for affine F and cubic detF; shared with residual midpoint H',cache='owned copies, original six-entry q-key cache',additional_static_device_bytes=0)
        return obj

    def discrete(self,q0,q1):
        # No interpolation of H or frozen F: each missing q is evaluated normally.
        g0=self.evaluate(q0)['gradient'];gm=self.evaluate(.5*(q0+q1))['gradient'];g1=self.evaluate(q1)['gradient']
        return (g0+4*gm+g1)/6


def install(coupling,previous_bridge=None):
    c=coupling;state=c.state;c.validate(state);before=physical_payload(state);old=copy.deepcopy(c.identity);core=c.core
    g=SharedMidpointGeometry.from_reduced(c.geometry);core.geometry=g;core.identity=dict(core.identity,geometry=g.identity,implementation='shared-midpoint-exact-volume-gradient-v1');core.signature=digest(core.identity)
    c.identity=dict(schema='boundary-reference-shared-geometry-fixture-v1',physical_parent=old,core=core.identity);c.signature=digest(c.identity)
    state.child_states['fluid']['model']=core.signature;state.child_states['explicit_pressure_grid']=c.signature
    if physical_payload(state)!=before:raise ValueError('implementation changed physical history')
    core._transaction=StateTransaction(state,validator=c.validate)
    return dict(status='passed_scoped',prior_bridge=previous_bridge,old_identity=old,new_identity=c.identity,physical_payload_exact=True,physical_payload_sha256=digest(before),new_digest=state.digest(),additional_static_device_bytes=0,metadata_bytes=g.local_bytes)
