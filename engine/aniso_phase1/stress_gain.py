"""Exact stress-error reduction score for physically derived local corrections.

The H.T H operator is used ONLY to rank spaces. The equilibrium energy,
residual, and tangent always use the original H and original carrier patch.
"""
import numpy as np
from . import local_reference as ref
from .compatible_carrier import shape_matrices
from benchmarks.aniso_local_q3 import gradient,chunks


def reference_stress_load(edges,reference,H):
    er,pr,ur=reference;common=[np.union1d(a,b) for a,b in zip(edges,er)];n=np.prod([2*(len(e)-1)+1 for e in edges]);force=np.zeros((n,3));norm=0.
    for X,V in chunks(common,order=max(3,pr+1),size=16):
        P=gradient(X,er,pr,ur).reshape(-1,9)@H.T;dual=(P@H).reshape(-1,3,3);norm+=float(V@np.sum(P*P,axis=1));_,*B=shape_matrices(X,edges)
        for j,b in enumerate(B):force+=b.T@(V[:,None]*dual[:,:,j])
    return force.T.ravel(),norm


def predicted_reduction(ids,correction,stress_residual,Kstress_local):
    return float(-2*stress_residual[ids]@correction-correction@(Kstress_local@correction))
