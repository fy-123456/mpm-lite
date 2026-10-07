"""Four transverse CPU references, then bounded actual-q geometry checks."""
import argparse,time,resource
import numpy as np
import scipy.linalg as la
from .provenance import *
from .runtime import update
from .spaces import load_selected
from benchmarks.research_stabilization_boundary_next.pressure import algebra,exact,comparison
from benchmarks.research_restoring_rt0_next.grid_transfer import restriction
from benchmarks.research_sequential_next.compare import metric
from benchmarks.research_phase_stress_next.time_study import history
from engine.aniso_phase1.research_startup_substeps_next.schedule import OBSERVATIONS
from engine.aniso_phase1.research_continuous_geometry_next.reference import prolongation,integrate
from engine.aniso_phase1.research_local_span_next.rt0 import MOBILITY

PAIRS=(('base','y'),('base','z'),('y','yz'),('z','yz'))

def main(run):
    run=Path(run);mutable(run);tick=time.perf_counter();deadline=tick+900;p=read(run/'S0/coupled-protocol.json');base=p['cuts'];grids={key:[base[0],np.linspace(base[1][0],base[1][-1],ny+1).tolist(),np.linspace(base[2][0],base[2][-1],nz+1).tolist()] for key,ny,nz in [('base',1,1),('y',2,1),('z',1,2),('yz',2,2)]}
    register(run,'S3/grid-protocol.json',dict(cuts=grids,parameters=p['parameters'],times_s=OBSERVATIONS.tolist(),max_static_seconds=900,max_assemblies=12,max_spectral_evaluations=4,geometry_orders=[7,9],source_steps=[8,28],gradient_direction='fixed seed 742, free coefficients normalized to 1e-6 m; full G retained',action_directions='BASE actual face flux and seed 731 mixed; each H action normalized to 0.2 Pa',GPU_guard_unchanged=True))
    models={};values={};meta=[]
    for k,cuts in grids.items():
        a=algebra(cuts,p['parameters']);models[k]=a;v=exact(a,p['parameters'],OBSERVATIONS);values[k]=v;t=a['top'];meta.append(dict(grid=k,shape=list(t.shape),cells=t.cells,faces=t.nflux,H_bytes=a['H'].nbytes,factorization_cubic_units=t.nflux**3,quadrature_points=27*t.cells,min_eigenvalue=float(a['lam'][0]),symmetry_defect=float(la.norm(a['H']-a['H'].T))))
        np.savez_compressed(run/'S3'/('fixed-'+k+'.npz'),times=OBSERVATIONS,H=a['H'],B=t.B,V0=t.V0,**v)
    transfers=[];comparisons=[]
    for a,b in PAIRS:
        ca,cb=models[a]['top'],models[b]['top'];P,M,Z=restriction(ca,cb);E=prolongation(ca,cb);checks=dict(constants=float(np.max(abs(P@np.ones(cb.cells)-1))),volumes=float(np.max(abs(M@cb.V0-ca.V0))),divergence=float(np.max(abs(ca.B@Z-M@cb.B))),injection=float(np.max(abs(Z@E-np.eye(ca.nflux)))),fixed_H_relative=float(la.norm(models[a]['H']-E.T@models[b]['H']@E)/la.norm(models[a]['H'])))
        if max(checks.values())>1e-8:raise ValueError('reference transfer failed')
        transfers.append(dict(coarse=a,fine=b,checks=checks,passed=True));comparisons.append(dict(coarse=a,fine=b,comparison=comparison(models[a],models[b],values[a],values[b],OBSERVATIONS)))
    write(run/'S3/topology-transfer-check.json',dict(status='passed_scoped',grids=meta,records=transfers));write(run/'S3/fixed-skeleton-comparison.json',dict(status='diagnostic',records=comparisons,continuous_space_accuracy=False,spectral_evaluations=4))
    update(run,'S3：四种横向CPU网格已完成固定骨架参考，拓扑、守恒限制及RT0延拓检查通过；保留全部方向差异，不据此宣称三维空间收敛。')
    r,_=load_selected(read(run/'selected-space.json')['package']);s=r.parent;expected=[]
    for k in ('base','yz'):
        for order in (7,9):
            n=[order*(len(np.unique(np.r_[s.edges[j],grids[k][j]]))-1) for j in range(3)];expected.append(dict(grid=k,order=order,shape=n,points=int(np.prod(n)),unstreamed_F_bytes=int(np.prod(n))*72,nodal_dual_bytes=int(np.prod(s.shape))*24))
    write(run/'S3/frozen-preflight.json',dict(status='registered',records=expected,parent_shape=list(s.shape),full_G_shape=[128,r.P.shape[1],3],dense_node_basis_used=False,streaming='one x quadrature slab, one cell nodal dual; no node-by-coefficient table',seconds_remaining=deadline-time.perf_counter()))
    h=history(APP/'cases/boundary32-h');states=[h[8],h[28]];rng=np.random.default_rng(742);direction=np.zeros_like(states[0]['state'].q);direction[r.free]=rng.normal(size=direction[r.free].shape);direction*=1e-6/la.norm(direction);outputs={};completed=[];limitation=None
    for item in states:
        for key in ('base','yz'):
            for order in (7,9):
                name=f"state{item['state'].step}-{key}-q{order}"
                try:
                    v=integrate(r,item['state'].q,models[key]['top'],p['parameters']['mobility_scale']*MOBILITY,order,deadline);outputs[name]=v;np.savez_compressed(run/'S3'/(name+'.npz'),**v);completed.append(dict(name=name,seconds=float(v['seconds']),points=int(v['points']),min_detF=float(v['min_detF'])));print('FROZEN',completed[-1],flush=True)
                    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20>16:raise MemoryError('S3 RSS cap')
                except (TimeoutError,MemoryError) as e:limitation=repr(e);break
            if limitation:break
        if limitation:break
    checks=[];spatial=[]
    for item in states:
        step=item['state'].step
        for k in ('base','yz'):
            names=[f'state{step}-{k}-q{q}' for q in (7,9)]
            if not all(n in outputs for n in names):continue
            a,b=[outputs[n] for n in names];g=models[k]['top'];E=prolongation(models['base']['top'],g);z=E@np.asarray(item['state'].child_states['fluid']['flux_interval_m3_s']);mixed=np.random.default_rng(731).normal(size=g.nflux);directions=[]
            for d in (z,mixed):
                d=d*(.2/la.norm(b['H']@d));directions.append(metric(a['H']@d,b['H']@d,.001,.05))
            dg=[np.einsum('kij,ij->k',x['gradient'],direction) for x in (a,b)];pdir=np.full(g.cells,.2);cc=dict(H=metric(a['H'],b['H'],1e-8,.05),V=metric(a['volume'],b['volume'],1e-10,.05),G_direction=metric(*dg,1e-10,.05),pressure_work=metric(.8*pdir@dg[0],.8*pdir@dg[1],1e-10,.05));checks.append(dict(step=step,grid=k,checks=cc,actions=directions,passed=all(x['passed'] for x in [*cc.values(),*directions])))
        for order in (7,9):
            names=[f'state{step}-{k}-q{order}' for k in ('base','yz')]
            if not all(n in outputs for n in names):continue
            a,b=[outputs[n] for n in names];P,M,Z=restriction(models['base']['top'],models['yz']['top']);E=prolongation(models['base']['top'],models['yz']['top']);cc=dict(H=metric(a['H'],E.T@b['H']@E,1e-8,.05),V=metric(a['volume'],M@b['volume'],1e-10,.05),G=metric(a['gradient'],np.einsum('ab,bij->aij',M,b['gradient']),1e-8,2e-5));spatial.append(dict(step=step,order=order,checks=cc,passed=all(x['passed'] for x in cc.values())))
    status='passed_scoped' if len(completed)==8 and all(x['passed'] for x in checks+spatial) else 'limited';write(run/'S3/frozen-geometry-reference.json',dict(status=status,completed=completed,quadrature_checks=checks,subspace_checks=spatial,limitation=limitation,seconds=time.perf_counter()-tick,new_dynamic_steps=0,actual_coupled_spatial_accuracy=False,full_G=True,scope='two frozen true displacements; preliminary q7/q9 sufficient integration and RT0 subspace only'))
    write(run/'S3/reference-status.json',dict(status='limited',topology='passed_scoped',fixed_skeleton='transverse_change_diagnostic',frozen_geometry=status,actual_coupled_spatial_accuracy=False,assemblies=4+len(completed),spectral_evaluations=4,seconds=time.perf_counter()-tick,peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20));update(run,f'S3真实位移冻结算子：{status}，完成{len(completed)}/8组静态积分，耗时{time.perf_counter()-tick:.1f}秒；实际动态空间精度仍未认证。')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):main(a.run)
