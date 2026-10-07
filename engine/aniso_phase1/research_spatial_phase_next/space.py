"""Compact combinations of a qualified reference basis; exact full operators."""
import copy
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.research_d.identity import digest
from engine.aniso_phase1.research_d.stage2.contracts import array_digest
from engine.aniso_phase1.research_sequential.condensation import Condensation


def combine(reference,C,label):
    s=reference.parent;C=np.asarray(C,dtype=np.float64)
    if C.shape!=(s.ndof-s.n,144) or not np.isfinite(C).all():raise ValueError('finite reference-local x144 transform required')
    obj=copy.copy(s);obj.transform=np.array(s.transform@C,copy=True);obj.ndof=s.n+144
    obj.reference=np.vstack((s.carrier_X,np.zeros((144,3))));obj.free_scalar_ids=np.r_[s.free_scalar_ids[s.free_scalar_ids<s.n],np.arange(s.n,obj.ndof)]
    obj.fixed_scalar_ids=s.fixed_scalar_ids.copy();obj.q_shape=(len(obj.free_scalar_ids),3)
    obj.lift=np.vstack((s.lift[:s.n],np.zeros((144,3))));obj.q0=np.vstack((s.q0[:s.nfree_carrier],np.zeros((144,3))))
    obj.signature=digest(dict(parent=s.signature,combination=array_digest(C),budget=144,label=label))
    obj.metadata=dict(s.metadata,derived_space_label=label,parent_sha256=s.signature,q_shape=list(obj.q_shape))
    for v in [obj.transform,obj.reference,obj.free_scalar_ids,obj.fixed_scalar_ids,obj.lift,obj.q0]:v.setflags(write=False)
    T=la.block_diag(np.eye(s.n),C);T3=np.kron(T,np.eye(3))
    M=T.T@reference.original_mass@T;K=T3.T@reference.original_stiffness@T3
    return Condensation(obj,M,K),T
