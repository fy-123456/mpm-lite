"""Check midpoint boundary momentum reflection directly in saved states."""
import json
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v18_runs import ROOT,OUT,load,write,sha,initial
from engine.aniso_phase1.carrier_joint import State,Geometry
from engine.aniso_phase1.unresolved_velocity import pack

def main():
    assert not (OUT/'boundary-reflection.json').exists();p=load(OUT/'protocol.json');records=[]
    for level in range(4):
        name=f'cycle-L{level}';spec=p['cases'][name];dt=spec['dt'];_,e,m,h,_=initial(spec);rows=[json.loads(s) for s in (OUT/'cases'/name/'steps.jsonl').read_text().splitlines()]
        for path in sorted((OUT/'cases'/name).glob('audit-*.npz')):
            with np.load(path) as z:d={k:z[k] for k in z.files}
            step=int(path.stem.split('-')[-1]);row=rows[step-1];s=State(d['x_before'],d['Y_before'],d['v_before'],d['C_before']);g=Geometry(s,e,m,h);q=g.metric;fixed=(g.nodes[:,0]*h<=.25)|(g.nodes[:,0]*h>=.75);bc=np.zeros(len(g.nodes));bc[g.nodes[:,0]*h>=.75]=1.;lift=la.lstsq(g.E[fixed],bc[fixed],cond=1e-11)[0]
            U,sv,_=la.svd(np.sqrt(q)[:,None]*(g.J@g.Q),full_matrices=False);U=U[:,sv>sv[0]*1e-12];a=np.sqrt(q)*(g.J@lift);b=a-U@(U.T@a)
            z0=pack(s.v,s.C);z1=pack(d['v'],d['C']);p0=float(b@(np.sqrt(q)*z0[:,0]));p1=float(b@(np.sqrt(q)*z1[:,0]));bb=float(b@b);expected=2*row['loading_speed']*bb
            err=abs(p0+p1-expected);assert err<1e-12
            records.append(dict(case=name,step=step,time=row['time'],loading_speed=row['loading_speed'],boundary_momentum_before=p0,boundary_momentum_after=p1,
                expected_sum=expected,reflection_identity_error=err,orthogonal_inertial_reaction_N=(p1-p0)/dt,boundary_metric=bb,free_orthogonality=float(la.norm(U.T@b))))
    write(OUT/'boundary-reflection.json',dict(completed=True,source_sha256=sha(ROOT/'benchmarks/aniso_v18_boundary_reflection.py'),records=records,
        formula='b=(I-Pi_(sqrt(M)JQ))*sqrt(M)J*lift; p=b^T sqrt(M)z_x; p1+p0=2*speed*||b||^2. At zero speed p1=-p0.',
        scope='Identity demonstrates reflected boundary momentum in the actual scheme. It does not by itself identify when/how that component is excited as geometry moves.'))
    print('reflection max error',max(r['reflection_identity_error'] for r in records),flush=True)
    print('terminal records',[r for r in records if abs(r['time']-1.6)<1e-10],flush=True)
if __name__=='__main__':main()
