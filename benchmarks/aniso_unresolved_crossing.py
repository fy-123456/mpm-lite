"""Extra geometry stress screen: joint velocity filtering across cell boundaries.

Positions are prescribed for this isolated intervention. This is not a free
flight integrator and does not claim an energy bound for particle advection.
"""
import hashlib
import numpy as np
from benchmarks.aniso_unresolved_history import OUT,write
from engine.aniso_phase1.unresolved_velocity import VelocityFilter,pack
from tests.test_aniso_unresolved_velocity import geometry,angular


def main():
    x,m=geometry();h=.125;rng=np.random.default_rng(785);v=rng.normal(size=x.shape)*.01;C=rng.normal(size=(len(x),3,3))*.1
    omega=np.array([[0.,-.3,.1],[.3,0.,-.2],[-.1,.2,0.]])
    shifts=sorted(set(np.linspace(0,2*h,33))|{h/4-1e-10,h/4+1e-10,3*h/4-1e-10,3*h/4+1e-10})
    rows=[]
    for mode in ('null','weak'):
        for shift in shifts:
            pos=x+.5+shift*np.array([1.,.5,-.25]);p=VelocityFilter(pos,m,h,.0005,np.sqrt(10)/h,mode,(25,25,25));a,b,d=p.apply(v,C)
            va=pos@omega.T+np.array([.01,-.02,.03]);Ca=np.broadcast_to(omega,C.shape);u,G,_=p.apply(va,Ca)
            affine=max(float(np.max(abs(va-u))),float(np.max(abs(Ca-G))))
            momentum=float(np.linalg.norm(m@(a-v)));spin=float(np.linalg.norm(angular(pos,m,p.data['D'],a,b)-angular(pos,m,p.data['D'],v,C)))
            grid=float(np.max(abs(p.data['momentum_map']@(pack(a,b)-pack(v,C)))))
            assert d['dissipation_delta']<=1e-14 and d['dissipation_identity_error']<1e-14
            assert max(affine,momentum,spin)<1e-12
            if mode=='null':assert grid<1e-12
            rows.append(dict(mode=mode,shift=shift,nodes=len(p.data['nodes']),rank=len(p.singular),energy_delta=d['dissipation_delta'],energy_identity_error=d['dissipation_identity_error'],affine_error=affine,momentum_error=momentum,angular_momentum_error=spin,node_momentum_change=grid))
    write(OUT/'crossing-filter-check.json',dict(passed=True,cases=len(rows),records=rows,source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),scope='isolated filters at prescribed translating positions, including exact and near support transitions; total advection energy and full cross-cell dynamics not certified'))
    print('passed',len(rows),'max affine',max(r['affine_error'] for r in rows),'ranks',sorted(set(r['rank'] for r in rows)))

if __name__=='__main__':main()
