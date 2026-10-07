"""Design one 138+6 candidate; release reference memory before separate validation."""
from pathlib import Path
import argparse,gc,time
import numpy as np
import scipy.linalg as la
from .provenance import *
from .spaces import load_selected
from benchmarks.research_phase_boundary_next.solver import bounded_static
from benchmarks.research_cross_direction_next.spatial_study import material,mass_project,nodal
from benchmarks.research_reference_next.reference_study import reopen
from benchmarks.research_phase_reference_next.spatial_study import reload_level
from benchmarks.research_local_span_next.spatial_study import independent_basis,reference4
from benchmarks.research_reference_next.field_audit import compare_nodal,invariants
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.compare import metric,regions
from benchmarks.research_sequential_next.run import probe_frame
from engine.aniso_phase1.research_spatial_phase_next.space import combine
from engine.aniso_phase1.research_sequential.condensation import CondensedModel
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_d.common_kinetic import PointInertia
PB=ROOT/'docs/results/phase-boundary/20261001T183820Z-phase-boundary'
CROSS=ROOT/'docs/results/cross-direction/20261001T170848Z-cross-direction'
OP=ROOT/'docs/results/observable-pressure/20261001T154616Z-observable-pressure'
NAME='origin-restoring-snapshot6'

def study(run):
    import warp as wp
    run=Path(run);verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');started=time.perf_counter()
    audit=read(PB/'S2/reference-audit.json');reserved=read(PB/'S2/reserved-direction-check.json');old=read(PB/'S2/training-comparison.json')['records'];refok=all(v['reliable'] for v in audit['records'].values()) and all(v['absolute']<=.25*v['budget'] for row in reserved['adjacent'].values() for v in row.values())
    eligible=refok and read(run/'S3/reference-scope-decision.json')['local_prefix_time_passed'] and not read(run/'S1/diagnosis-decision.json')['unexplained_numeric_anomaly']
    register(run,'S4/design-eligibility.json',dict(status='passed_scoped' if eligible else 'not_triggered',eligible=eligible,static_reference_reliable=refok,reference_audit=dict(path=str(PB/'S2/reference-audit.json'),sha256=sha(PB/'S2/reference-audit.json')),historical_regression=dict(path=str(PB/'S2/reserved-direction-check.json'),sha256=sha(PB/'S2/reserved-direction-check.json')),angles=[45,52.5,60,48.75],independent_hidden_set=False,no_new_reference_solves=True))
    if not eligible:
        write(run/'S4/space-decision.json',dict(status='not_triggered',reason='reference/diagnostic prerequisite',new_spaces=0,new_static_solves=0,new_dynamic_attempts=0));return
    weights={str(a):1+max(v['absolute']/v['budget'] for row in old[str(a)]['errors'].values() for v in row.values())+old[str(a)]['reaction']['absolute']/old[str(a)]['reaction']['budget'] for a in (45,52.5,60)}
    register(run,'S4/design-protocol.json',dict(status='registered',name=NAME,keep=138,replace=6,budget=144,rank_cutoff=1e-9,static_weights=weights,origin_restoring_weight=.2,method='complete mass Schur covariance; unit block norms, static regional stress and full reaction weights; origin stabilization acceleration residual; leading six directions',dynamic_indices=[0],late_diagnostic_nodes_used_for_training=False,primary_goal=dict(angle=60,region='interior',quantity='fiber_PK1',minimum_relative_improvement=.1,minimum_improvement_over_reference=2.,absolute_engineering_scale_Pa=.02,small_error_percentages_not_claimed_as_benefit=True),regression='increase <= max(2 adjacent-reference error, .1 engineering budget)',max_candidates=1,max_static_solves=4,max_new_dynamic_attempts=28,formal_space_unchanged=True))
    r,rm,rs,err=reopen(REFERENCE/'Q1/R3');del rm,rs;gc.collect();r4,f4,_,_=reference4(r);r5,f5,_,_=reload_level(PHASE_REFERENCE,5);refs={45:f5};projections={};adjacent={float(k):v['adjacent'] for k,v in audit['records'].items()};adjacent[48.75]=reserved['adjacent'];old_errors={float(k):v['errors'] for k,v in old.items()};old_errors[48.75]=reserved['new']
    for a,path in [(52.5,CROSS/'S1/reserved/R5/state.npz'),(60,OP/'S2/direction/R5/state-fields.npz'),(48.75,PB/'S2/reserved/R5/state.npz')]:
        with np.load(path) as z:refs[a]=z['full'].copy()
    for a in refs:projections[a]=mass_project(r,r5,refs[a])
    cfg=read(APP/'cases/candidate-quarter/execution-protocol.json');baseline,_=load_selected(cfg['physical_space']);folder=Path(cfg['physical_space']['path']).parent
    with np.load(folder/'space.npz') as z:C0=z['C'].copy();Tc=z['T'].copy()
    fcfg=read(OBS/'cases/phase-quarter/execution-protocol.json');formal,_=load_selected(fcfg['physical_space'])
    with np.load(Path(fcfg['physical_space']['path']).parent/'space.npz') as z:Tf=z['T'].copy()
    qc=history(APP/'cases/candidate-quarter')[0]['state'].q;qf=history(OBS/'cases/phase-quarter')[0]['state'].q
    def restoring(rr,q,T):
        s=rr.parent;f=rr.P[:s.n].T@s.Ks@(s.reference[:s.n]+rr.expand(q)[:s.n]);a=np.zeros_like(q);a[rr.free]=la.solve(rr.M[np.ix_(rr.free,rr.free)],-f[rr.free],assume_a='pos');return T@rr.velocity(a)
    response=restoring(formal,qf,Tf)-restoring(baseline,qc,Tc)
    s=r.parent;M=r.original_mass;keep=np.zeros((s.ndof-s.n,138));keep[:138]=np.eye(138);B=independent_basis(r,keep);ids=np.arange(s.n+138,s.ndof);S=M[np.ix_(ids,ids)]-(M[ids]@B)@la.solve(B.T@M@B,B.T@M[:,ids],assume_a='pos');L=la.cholesky((S+S.T)/2,lower=True)
    blocks=[]
    for a in (45,52.5,60):
        v=L.T@projections[a][ids];blocks.append(np.sqrt(weights[str(a)])*v/max(la.norm(v),1e-30))
    v=L.T@response[ids];blocks.append(np.sqrt(.2)*v/max(la.norm(v),1e-30));U,sv,_=la.svd(np.column_stack(blocks),full_matrices=False);rank=int(sum(sv>sv[0]*1e-9))
    if rank<6:raise ValueError('new candidate has fewer than six independent directions')
    C=np.zeros_like(C0);C[:138,:138]=np.eye(138);C[138:,138:]=la.solve_triangular(L.T,U[:,:6],lower=False)
    oldB=independent_basis(r,C0);oldB/=np.sqrt(np.diag(oldB.T@M@oldB));Z=np.zeros((s.ndof,6));Z[s.n:]=C[:,138:];perp=Z-oldB@la.solve(oldB.T@M@oldB,oldB.T@M@Z,assume_a='pos');angles=la.eigvalsh(perp.T@M@perp,Z.T@M@Z)
    if not np.any(angles>1e-6):raise ValueError('no resolved new span')
    from engine.aniso_phase1.tensor_reference import coordinates
    xs=coordinates(s.edges,s.p)[0];supports=[]
    for j in range(144):
        v=np.asarray(s.raw@(s.transform@C[:,j])).reshape(s.shape);active=np.where(np.max(abs(v),axis=(1,2))>np.max(abs(v))*1e-10)[0];supports.append(dict(column=j,x=xs[active[[0,-1]]].tolist(),boundary_max=float(max(np.max(abs(v[0])),np.max(abs(v[-1]))))))
    cr,T=combine(r,C,NAME);target=run/'S4/candidates'/NAME;target.mkdir(parents=True);initial=la.lstsq(T,projections[45])[0];np.savez_compressed(target/'space.npz',C=C,T=T,M=cr.original_mass,K=cr.original_stiffness,initial=initial)
    package=dict(schema='spatial-phase-combination-v1',name=NAME,budget=144,reference_sha256=sha(REFERENCE/'Q1/R3/space-package.json'),data_sha256=sha(target/'space.npz'),reduction_sha256=cr.signature,space_sha256=cr.parent.signature,mass_order=7,full_order=7,supports=supports,rank=cr.audit,promoted=False)
    write(target/'space-package.json',package);write(run/'S4/candidate-package.json',dict(status='passed_scoped',path=str(target/'space-package.json'),sha256=sha(target/'space-package.json'),snapshot_singular_values=sv.tolist(),principal_sine_squared=angles.tolist(),keep=138,replace=6,origin_only=True))
    print('CANDIDATE_FROZEN',target,'validate in separate serial processes with spatial_validate',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
