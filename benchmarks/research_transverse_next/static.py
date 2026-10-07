"""Four geometry states per grid; independent rest RT0 and pressure-work tests."""
import argparse,time,resource
import numpy as np
import scipy.linalg as la
from .provenance import *
from .fixture import setup
from .runtime import update
from .observables import modes,face_signals
from engine.aniso_phase1.research_transverse_next.initial import pressure_profile
from benchmarks.research_sequential_next.compare import metric

def cpu_rest_H(top,mobility):
    xi,wi=np.polynomial.legendre.leggauss(2);points=[];weights=[];cells=[]
    for c,b in enumerate(top.cell_bounds):
        axes=[(lo+hi)/2+(hi-lo)/2*xi for lo,hi in b]
        X=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
        w=(wi[:,None,None]*wi[None,:,None]*wi[None,None,:]).ravel()*top.V0[c]/8
        points.extend(X);weights.extend(w);cells.extend([c]*8)
    X=np.asarray(points);F=np.broadcast_to(np.eye(3),(len(X),3,3)).copy()
    return top.assemble(X,np.asarray(weights),np.asarray(cells),F,mobility)[0]

def main(run,case):
    run=Path(run);mutable(run);tick=time.perf_counter();c,m,cfg,ident=setup(run,case);g=c.geometry;t=g.topology
    folder=run/'S1'/case;write(folder/'identity.json',ident);snapshot(folder/'source',ident['numerical_sources'])
    q=c.state.q;rest=g.evaluate(q);H=cpu_rest_H(t,g.mobility)
    rng=np.random.default_rng(817);z=rng.normal(size=t.nflux);z*=.2/np.linalg.norm(H@z)
    checks=dict(rest_volume=metric(rest['volume'],t.V0,1e-12,2e-5),H_action=metric(rest['H']@z,H@z,1e-6,2e-5))
    directions={}
    for name in (['Y64'] if case=='Y64' else ['Y128','YZ128']):
        p,d=pressure_profile(name,t);rhs=t.B.T@p-c.core.gb
        a=la.solve(rest['H'],rhs,assume_a='pos');b=la.solve(H,rhs,assume_a='pos')
        directions[name]=dict(check=metric(a,b,1e-10,2e-5),modes=modes(t,p),darcy_diagnostic=face_signals(t,b),diagnostic_is_not_committed_flux=True,initial_flux_zero=not np.any(c.state.child_states['fluid']['flux_interval_m3_s']))
    d=np.zeros_like(q);d[m.free]=rng.normal(size=d[m.free].shape);d*=1e-6/np.linalg.norm(d)
    plus=g.evaluate(q+d);minus=g.evaluate(q-d);bar=g.discrete(q,q+d)
    p=np.asarray(c.state.child_states['fluid']['pressure_Pa']);fd=.5*c.core.alpha*float(p@(plus['volume']-minus['volume']))
    analytic=c.core.alpha*float(np.einsum('k,kij,ij',p,rest['gradient'],d))
    checks['pressure_potential_direction']=metric(fd,analytic,1e-12,2e-5)
    checks['Simpson_volume_chain']=metric(np.einsum('kij,ij->k',bar,d),plus['volume']-rest['volume'],1e-12,2e-5)
    owned=g.evaluate(q+d);owned['gradient'][:]=111
    checks['owned_cache']=dict(passed=np.array_equal(g.evaluate(q+d)['gradient'],plus['gradient']))
    status='passed_scoped' if all(x['passed'] for x in checks.values()) and all(x['check']['passed'] for x in directions.values()) else 'failed'
    write(folder/'operator-check.json',dict(status=status,checks=checks,directions=directions,geometry=g.identity,static_state_groups=4,CPU_rest_points=t.cells*8,min_detF=min(rest['min_detF'],plus['min_detF'],minus['min_detF']),profile=g.profile,seconds=time.perf_counter()-tick,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,mass_sha256=digest(m.M.tolist()),initial_digest=c.initial_digest))
    if status!='passed_scoped':raise ValueError('directional operator check failed')
    update(run,f'S1 {case}：静止完整张量CPU H、Y/YZ方向流量、压力势差分、Simpson体积链和缓存所有权通过；四个几何状态，元数据{g.local_bytes/2**20:.1f}MiB。')
    print(case,status,checks,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--case',choices=['Y64','Y128'],required=True);a=p.parse_args()
    with serial_lock(a.run):main(a.run,a.case)
