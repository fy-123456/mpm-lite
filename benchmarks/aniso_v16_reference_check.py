"""Cross-check ill-conditioned generalized modes via weighted-J SVD whitening."""
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_carrier_joint import load_case,OUT,write
from benchmarks.aniso_v16_analysis import arrays,rms,PARAMS
from engine.aniso_phase1.carrier_joint import Geometry,gradient
from engine.aniso_phase1.unresolved_velocity import pack
from engine.aniso_phase1.history_increment import material_tangent
from benchmarks.aniso_dynamic_check import pk1


def main():
    assert not (OUT/'modal-reference-check.json').exists();records=[]
    for label,t in [('early_hold',1.1),('late_hold',1.6)]:
        s,e,m,h,_=load_case(t);g=Geometry(s,e,m,h);Q=g.Q;Jr=g.J@Q;K=e.tangent(s.Y,Q)
        _,sigma,Vh=la.svd(np.sqrt(g.metric)[:,None]*Jr,full_matrices=False);assert sigma[-1]>1e-12*sigma[0]
        W=Vh.T/sigma;white=la.block_diag(W,W,W);H=white.T@K@white;lam,V=la.eigh((H+H.T)/2);Phi=white@V;omega=np.sqrt(lam)
        f=(Q.T@e.evaluate(s.Y)['force']).T.ravel();b=(Jr.T@(g.metric[:,None]*pack(s.v,s.C))).T.ravel()
        eq=-(Phi.T@f)/lam;c=-eq-1j*(Phi.T@b)/omega;B=[bb@Q for bb in e.B];F=gradient(e.B,s.Y);P=pk1(F,e.A,200.);values=[]
        for tau in np.arange(11)*.005:
            u=(Phi@(eq+np.real(c*np.exp(1j*omega*tau)))).reshape(3,Q.shape[1]).T
            values.append(P+material_tangent(F,e.A,gradient(B,u),PARAMS))
        exact=arrays(OUT/f'modal-{label}.npz')['P_exact_linear'];difference=rms(np.array(values)-exact)/rms(exact)
        assert difference<1e-5,difference
        records.append(dict(time=t,exact_time_stress_relative_difference=difference,
            minimum_kinetic_singular=float(sigma[-1]),positive_modes=len(lam),smallest_frequency_rad_s=float(omega.min()),largest_frequency_rad_s=float(omega.max())))
    write(OUT/'modal-reference-check.json',dict(passed=True,records=records,
        scope='Original generalized-eigenvalue exact time reference cross-checked without normal-equation mass formation, using weighted-J SVD. Same local linear model.'))
    print(records,flush=True)

if __name__=='__main__':main()
