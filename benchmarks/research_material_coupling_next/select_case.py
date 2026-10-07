"""Cheap linearized candidate selection, not a nonlinear qualification."""
import argparse
from pathlib import Path
import numpy as np
import scipy.linalg as la
from .common import MaterialStateModel,unit_boundary,write
from engine.aniso_phase1.research_material_coupling_next.budget import Budget
from engine.aniso_phase1.research_material_coupling_next.operators import Geometry,Material
from engine.aniso_phase1.research_material_coupling_next.solver import Solver

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();m=MaterialStateModel();budget=Budget();g=Geometry(m,budget);op=Material(m,7,budget);unit,right=unit_boundary(m)
    q0=m.rest().q;geo=g.evaluate(q0);G=geo['G'].reshape(2,-1);rows=[]
    for h in (.01,.025):
        cases={}
        for drained in (False,True):
            sol=Solver(m,g,op,drained=drained);A=2*sol.M/h**2+.5*sol.K
            d=unit.ravel()*(-1e-5/16);J=np.block([[A[np.ix_(sol.free,sol.free)],-.5*G[:,sol.free].T],
                [G[:,sol.free]/h,np.diag(sol.C/h)+.5*sol.L]])
            rhs=-np.r_[(A@d)[sol.free],G@d/h];y=la.solve(J,rhs);d[sol.free]=y[:-2];press=y[-2:]
            residual=A@d-.5*G.T@press;R=residual.reshape(-1,3)[right,0].sum()
            cases['drained' if drained else 'closed']=dict(reaction=float(R),pressure=press)
        eta=abs(cases['closed']['reaction']-cases['drained']['reaction'])/max(abs(cases['closed']['reaction']),1e-8)
        rows.append(dict(dt=h,amplitude_coefficient=-1e-5/(4*h)**2,eta_reaction=float(eta),cases=cases))
    eligible=[x for x in rows if x['eta_reaction']>=.01];choice=eligible[0] if eligible else rows[-1]
    write(a.run/'S3/selection.json',dict(candidates=rows,selected=choice,scope='linearized selection only; original solid, unchanged fluid constants; at most four final nonlinear steps',budget=budget.report()))
    print('selected',choice,flush=True)
