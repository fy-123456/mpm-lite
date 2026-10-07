"""Re-evaluate novelty with mass-scaled coordinates without modifying the candidate."""
from pathlib import Path
import argparse
import numpy as np
import scipy.linalg as la
from .provenance import *
from .spaces import load_selected

def review(run):
    run=Path(run);verify(run);r,_=load_selected(read(run/'baseline-space.json')['package'])
    with np.load(REFERENCE/'Q1/R3/data.npz') as z:M=z['M'].copy()
    with np.load(LOCAL/'S1/candidates/global-snapshot6/space.npz') as z:T0=z['T'].copy()
    with np.load(run/'S1/candidates/cross-direction-snapshot6/space.npz') as z:T=z['T'].copy()
    B=T0@r.P[:,r.free];G=B.T@M@B;scale=np.sqrt(np.diag(G));B=B/scale;G=B.T@M@B
    Z=T[:,-6:];coef=la.solve(G,B.T@M@Z,assume_a='pos');perp=Z-B@coef;angles=la.eigvalsh(perp.T@M@perp,Z.T@M@Z)
    ortho=float(la.norm(B.T@M@perp)/max(la.norm(B.T@M@Z),1e-30));condition=float(np.linalg.cond(G))
    if ortho>1e-8 or not np.any(angles>1e-6):raise ValueError('scaled novelty inconclusive')
    write(run/'S1/conditioning-review.json',dict(status='passed_scoped',cause='unscaled independent columns have very different mass norms; normalized physical mass remains positive',scaled_Gram_condition=condition,mass_orthogonality_relative=ortho,principal_sine_squared=angles.tolist(),robust_new_directions_over_1e_6=int(sum(angles>1e-6)),candidate_unchanged=True,no_mass_regularization=True))
    print('SCALED_NOVELTY',condition,ortho,angles,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):review(a.run)
