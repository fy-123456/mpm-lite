"""Independent smallest local eigenvalues for enlarged static enrichment."""
import numpy as np
from scipy.sparse.linalg import eigsh
from benchmarks.aniso_v20_common import *
from benchmarks.aniso_boundary_reference import hessian
from engine.aniso_phase1 import local_reference as ref

def main():
    rows=[]
    for resolution in (16,32):
        with np.load(BASE/'v19/space'/f'reconstruction{resolution}.npz') as z:edges=[z[f'axis{k}'] for k in range(3)]
        X,K=ref.assemble(edges,2,hessian('F45'))
        for width in (.125,.1875,.25):
            local=(X[:,0]>.25+1e-12)&(X[:,0]<.75-1e-12)&(((X[:,0]<.25+width-1e-12)|(X[:,0]>.75-width+1e-12)) if width<.25 else True);ids=np.flatnonzero(np.tile(local,3));A=K[ids][:,ids];w,v=eigsh(A,k=1,which='SA',tol=1e-7);res=float(np.linalg.norm(A@v[:,0]-w[0]*v[:,0]));r=dict(resolution=resolution,width=width,dofs=len(ids),smallest_eigenvalue=float(w[0]),eigen_residual=res,mass_included=False,stiffness_shift=0.,passed=bool(w[0]>1e-8 and res<1e-6));assert r['passed'];rows.append(r);print(r,flush=True)
    write(OUT/'local-stiffness-checks.json',dict(completed=True,records=rows,all_passed=all(r['passed'] for r in rows)))
if __name__=='__main__':main()
