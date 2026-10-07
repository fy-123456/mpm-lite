"""Diagnose observed time sensitivity with rest tangent, no material traversal."""
import argparse,json
from pathlib import Path
import numpy as np
import scipy.linalg as la
from .common import MaterialStateModel,unit_boundary,write
from engine.aniso_phase1.research_material_coupling_next.budget import Budget
from engine.aniso_phase1.research_material_coupling_next.operators import Geometry,Material
from engine.aniso_phase1.research_material_coupling_next.solver import Solver

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();read=lambda name:json.loads((a.run/name).read_text())
    choice=read('S3/selection.json')['selected'];h=choice['dt'];coef=choice['amplitude_coefficient'];m=MaterialStateModel();budget=Budget();g=Geometry(m,budget)
    sol=Solver(m,g,Material(m,7,budget));unit,right=unit_boundary(m);G=g.evaluate(m.rest().q)['G'].reshape(2,-1);M=sol.M;K=sol.K
    def advance(dt,n):
        q=np.zeros(M.shape[0]);v=q.copy();p=np.zeros(2);rows=[];A=2*M/dt**2+.5*K
        J=np.block([[A[np.ix_(sol.free,sol.free)],-.5*G[:,sol.free].T],[G[:,sol.free]/dt,np.diag(sol.C/dt)+.5*sol.L]])
        fac=la.lu_factor(J)
        for i in range(n):
            q1=q.copy();q1[sol.fixed]=(unit*(coef*((i+1)*dt)**2)).ravel()[sol.fixed];p1=p.copy()
            def residual(q1,p1):
                d=q1-q;v1=2*d/dt-v;pm=(p+p1)/2
                return M@(v1-v)/dt+K@(q+q1)/2-G.T@pm,sol.C*(p1-p)+G@d+dt*sol.L@pm
            f,b=residual(q1,p1);dy=la.lu_solve(fac,-np.r_[f[sol.free],b/dt]);q1[sol.free]+=dy[:-2];p1+=dy[-2:];f,b=residual(q1,p1)
            rows.append(dict(reaction=float(f.reshape(-1,3)[right,0].sum()),pressure=p1.copy(),free_residual=float(la.norm(f[sol.free]))))
            v=2*(q1-q)/dt-v;q,p=q1,p1
        return rows
    coarse=advance(h,1);fine=advance(h/2,2);ic=h*coarse[0]['reaction'];ih=sum(h/2*r['reaction'] for r in fine)
    actual=read('S7/assessment.json')['time_comparison'];error=abs((ic-ih)/abs(ih));gap=abs(error-actual['impulse_relative'])
    write(a.run/'S3/linear-time-diagnosis.json',dict(coarse=coarse,half_steps=fine,linear_impulse_relative=error,actual_impulse_relative=actual['impulse_relative'],difference=gap,
        budget=budget.report(),scope='rest-linearized original space with full M/K and same hydraulic network; identifies time-discretization sensitivity without added physical frames; no modal convergence certificate'))
    print('linear/actual time impulse differences',error,actual['impulse_relative'],'gap',gap)
