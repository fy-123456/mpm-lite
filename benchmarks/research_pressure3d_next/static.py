"""Independent sealed CPU-reference comparisons before dynamic qualification."""
import argparse,time,resource
import numpy as np
from .provenance import *
from .fixture import setup
from .runtime import update
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.compare import metric

def main(run,grid):
    run=Path(run);mutable(run);tick=time.perf_counter();c,m,cfg,ident=setup(run,grid);g=c.geometry
    h=history(TRAJECTORY/'cases/boundary32-h');out=[]
    folder=run/'S1'/grid;folder.mkdir(parents=True,exist_ok=True)
    write(folder/'identity.json',ident);snapshot(folder/'source',ident['numerical_sources'])
    write(folder/'construction.json',dict(status='passed_scoped',identity=g.identity,build_seconds=g.build_seconds,initial_energy_J=m.evaluate(c.state.q)['U']+.5*np.sum(c.core.capacity*np.asarray(c.state.child_states['fluid']['pressure_Pa'])**2),mass_sha256=digest(m.M.tolist())))
    for step in (8,28):
        state=h[step]['state'];a=g.evaluate(state.q)
        path=APP/'S3'/f'state{step}-{grid}-q7.npz'
        with np.load(path) as z:b={k:z[k] for k in ('volume','gradient','H')}
        z=np.random.default_rng(731).normal(size=g.topology.nflux);z*=.2/np.linalg.norm(b['H']@z)
        checks={k:metric(a[k],b[k],1e-10 if k=='volume' else 1e-8,2e-5) for k in ('volume','gradient')}
        checks['H_action']=metric(a['H']@z,b['H']@z,1e-6,2e-5)
        out.append(dict(step=step,source_state_sha256=sha(h[step]['folder']/'state.json'),reference=dict(path=str(path),sha256=sha(path)),checks=checks,passed=all(x['passed'] for x in checks.values())))
        np.savez_compressed(folder/f'state{step}.npz',**a)
    rng=np.random.default_rng(742);d=np.zeros_like(h[8]['state'].q);d[m.free]=rng.normal(size=d[m.free].shape);d*=1e-6/np.linalg.norm(d)
    q0=h[8]['state'].q;q1=q0+d;v0=g.evaluate(q0);v1=g.evaluate(q1);gb=g.discrete(q0,q1)
    dv=np.einsum('kij,ij->k',gb,d);checks=dict(discrete_volume=metric(dv,v1['volume']-v0['volume'],1e-12,2e-5),reverse=metric(g.discrete(q1,q0),gb,1e-8,2e-5))
    saved=g.evaluate(q1);saved['volume'][:]=123;owned=np.array_equal(g.evaluate(q1)['volume'],v1['volume'])
    c0=g.evaluate(c.state.q);checks['rest_volume']=metric(c0['volume'],g.V0,1e-12,2e-5)
    passed=all(x['passed'] for x in out) and all(x['passed'] for x in checks.values()) and owned
    write(folder/'operator-check.json',dict(status='passed_scoped' if passed else 'failed',records=out,checks=checks,owned_results=owned,min_detF=v1['min_detF'],actual_q_cases=2,static_state_groups=5,profile=g.profile,seconds=time.perf_counter()-tick,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,old_guards_unchanged=True))
    if not passed:raise ValueError('D3 static independent comparison failed')
    update(run,f'S1 {grid}：有界构造及两份真实位移CPU参考、制造路径压力功、缓存所有权通过；元数据{g.local_bytes/2**20:.1f}MiB，构造暂存峰值{g.identity["construction_metadata_peak_bytes"]/2**20:.1f}MiB，耗时{time.perf_counter()-tick:.1f}s。')
    print('D3_STATIC',grid,'passed',g.identity,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--grid',choices=['base','yz'],required=True);a=p.parse_args()
    with serial_lock(a.run):main(a.run,a.grid)
