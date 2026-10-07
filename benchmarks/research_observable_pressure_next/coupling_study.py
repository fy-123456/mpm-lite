"""Separate pressure-time and 2/4-cell qualification on a frozen physical solid."""
from pathlib import Path
import argparse,time,copy,inspect
import numpy as np
import scipy.linalg as la
from .provenance import APP,read,write,sha,digest,register,verify,source_files,serial_lock,resources
from . import config
from .spaces import load_selected
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish
from engine.aniso_phase1.research_local_span_next.rt0 import TwoCellTopology,TensorCellGeometry,MOBILITY
from engine.aniso_phase1.research_observable_pressure_next.rt0 import BoundedTopology,BoundedGeometry,BoundedAVF,BoundedGridCoupling
from benchmarks.research_phase_reference_next.coupling_study import quadrature

def setup(run):
    import warp as wp
    run=Path(run);verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');choice=read(run/'selected-space.json');r,_=load_selected(choice['package'])
    cfg=config.make(read(run/'input-lock.json')['energy_scale_J'],end=.05,space=choice['package'],mass_order=7,full_order=7,reuse_transpose_buffers=False)
    return SegmentedModel(r,order=7,device='cuda:0',hold=0.),cfg

def source(geometry):return geometry.V0*np.array([.001 if i<geometry.cells//2 else 0. for i in range(geometry.cells)])

def affine_check(top,F):
    X,w,ids=quadrature(top);H,J=top.assemble(X,w,ids,np.broadcast_to(F,(len(X),3,3)));B=top.B;n=top.nflux;c=top.cells
    grad=np.array([.03,-.02,.04]);boundary=.1+top.centres@grad;g=top.boundary_term(boundary)
    A=np.block([[H,-B.T],[B,np.zeros((c,c))]]);answer=la.solve(A,np.r_[-g,np.zeros(c)]);z,p=answer[:n],answer[n:]
    K=la.det(F)*la.inv(F)@MOBILITY@la.inv(F).T;expected=-top.areas*(K@grad)[top.axes];expected_p=np.array([.1+b.mean(axis=1)@grad for b in top.cell_bounds])
    constant=la.solve(A,np.r_[-top.boundary_term(.1),np.zeros(c)])
    err=max(float(np.max(abs(z-expected))),float(np.max(abs(p-expected_p))),float(np.max(abs(constant[:n]))),float(np.max(abs(constant[n:]-.1))))
    closure=float(z@H@z+g@z)
    if err>1e-9 or abs(closure)>1e-10 or la.eigvalsh(H)[0]<=0:raise ValueError('full tensor RT0 manufactured field failed')
    return dict(cells=c,faces=n,error=err,energy_closure_J_s=closure,min_H_eigenvalue=float(la.eigvalsh(H)[0]),min_detF=J,all_axes_tested=True)

def topology(run):
    run=Path(run);records=[];bounds=[[.25,.75],[-.125,.125],[-.1875,.1875]]
    for cells in (2,4):
        top=BoundedTopology(bounds,cells)
        if not np.all(top.B[:,top.internal].sum(axis=0)==0) or abs(sum(top.V0)-.046875)>1e-12:raise ValueError('face or volume ownership failed')
        for F in (np.eye(3),np.array([[1.05,.12,0.],[.02,.98,.07],[0.,.03,1.02]])):records.append(affine_check(top,F))
    old=TwoCellTopology(bounds);new=BoundedTopology(bounds,2)
    for key in ('B','V0','faces','axes','areas','centres'):
        if not np.array_equal(getattr(old,key),getattr(new,key)):raise ValueError('two-cell topology changed')
    write(run/'S3/four-cell-topology.json',dict(status='passed_scoped',records=records,two_cell_layout_identical=True,mixed_system_SPD=False))
    from engine.aniso_phase1.research_phase_reference_next.scaled_rt0 import ScaledTensorCellAVF
    if BoundedAVF._compute is not ScaledTensorCellAVF._compute:raise ValueError('mixed residual was replaced')
    write(run/'S3/equation-inheritance.json',dict(status='passed_scoped',compute_inherited_unchanged=True,source=inspect.getsourcefile(BoundedAVF._compute),source_sha256=sha(inspect.getsourcefile(BoundedAVF._compute))))

def fixed(run,cells):
    run=Path(run);m,cfg=setup(run);geometry=BoundedGeometry(m,cells);rest=geometry.evaluate(m.rest().q);H=rest['H'];B=geometry.B;C=.0002*geometry.V0;L=B@la.solve(H,B.T,assume_a='pos');rate=L/C[:,None];rates=la.eigvalsh(L/np.sqrt(C[:,None]*C[None,:]));s=source(geometry);g=geometry.topology.boundary_term(.002)
    if cells==2:
        hmax=.2/rates[-1];T=min(1/rates[0],24*hmax);N=max(4,int(np.ceil(T/hmax)));times=np.linspace(0,T,N+1).tolist()
        old_end=read(APP/'S4/coupled-settings.json')['times_s'][-1]
        if T<=old_end or N>24:raise ValueError('pressure time extension unavailable in budget')
        register(run,'S3/pressure-time-protocol.json',dict(times_s=times,steps=N,lambda_positive=rates.tolist(),tau_fast_s=1/rates[-1],tau_slow_s=1/rates[0],
            T_over_tau_fast=T*rates[-1],T_over_tau_slow=T*rates[0],old_end_s=old_end,storage=.0002,alpha=.8,source_density_s=.001,source_region='left half of physical domain',fixed_pressure_budget_Pa=.001,max_steps=24))
        old=TensorCellGeometry(m);v=old.evaluate(m.rest().q);error=max(float(np.max(abs(v[k]-rest[k]))) for k in ('H','volume','gradient'))
        if error>1e-10:raise ValueError('two-cell geometry differs from parent')
        write(run/'S3/two-cell-geometry-equivalence.json',dict(status='passed_scoped',max_absolute=error))
    else:times=read(run/'S3/pressure-time-protocol.json')['times_s']
    core=BoundedGridCoupling(m,cfg,times,cells=cells,fixed_solid=True,source_m3_s=s);rhs=s+B@la.solve(H,g);pinf=la.solve(L,rhs);p0=np.full(cells,.01);rows=[];errors=[];start=time.perf_counter()
    for t in times[1:]:
        row=core.step();rows.append(row);expected=pinf+la.expm(-rate*t)@(p0-pinf);errors.append(float(np.max(abs(np.array(row['pressure_Pa'])-expected))))
    f=core.state.child_states['fluid'];mass=float(np.sum(np.array(f['content_m3'])-C*p0)+f['cumulative_boundary_m3']-sum(f['cumulative_source_m3']))
    passed=max(errors)<=.001 and abs(mass)<=1e-10 and min(x['darcy_dissipation_J'] for x in rows)>=0
    report=dict(status='passed_scoped' if passed else 'time_resolution_limited',cells=cells,faces=core.nflux,times_s=times,rows=rows,matrix_exponential_max_error_Pa=max(errors),mass_defect_m3=mass,
        seconds=time.perf_counter()-start,lambda_positive=rates.tolist(),max_h_lambda_max=max(np.diff(times))*rates[-1],source_m3_s=s.tolist(),reference_volumes=geometry.V0.tolist())
    write(run/f'S3/fixed-{cells}.json',report);print('FIXED_PRESSURE',cells,report['status'],len(rows),times[-1],max(errors),flush=True)
    if not passed:raise ValueError('fixed pressure time gate failed')

def coupled(run,cells,state=None):
    m,cfg=setup(run);ts=read(Path(run)/'S3/pressure-time-protocol.json')['times_s'];geometry=BoundedTopology([[e[0],e[-1]] for e in m.parent.edges],cells)
    src=geometry.V0*np.array([.001 if i<cells//2 else 0 for i in range(cells)])
    return BoundedGridCoupling(m,cfg,ts,cells=cells,source_m3_s=src,state=state),m,cfg

def start(run,cells):
    run=Path(run)
    if read(run/f'S3/fixed-{cells}.json')['status']!='passed_scoped':raise ValueError('fixed solid gate required')
    if cells==4 and (read(run/'S3/two-cell-coupled.json')['status']!='passed_scoped' or not read(run/'S3/pressure-scope-decision.json')['four_cell_fixed']):raise ValueError('two-cell time and complete fixed-grid qualification required')
    c,m,cfg=coupled(run,cells);folder=run/'cases'/f'pressure-{cells}';cache=CachedProbes(m);identity=dict(schema='observable-pressure-case-v1',coupling=c.identity,numerical_source_sha256=source_files(),fixture_source_sha256=sha(Path(__file__)),input_lock_sha256=sha(run/'input-lock.json'))
    if (folder/'identity.json').exists():raise ValueError('pressure case exists')
    write(folder/'identity.json',identity);write(folder/'execution-protocol.json',dict(config=cfg,times=c.times.tolist(),coupling=c.identity));store=GenerationStore(folder,identity);store.save(c.state,[],frame=c.frame(cache));rows=[]
    count=2 if cells==2 else 4
    for i in range(count):
        tick=time.perf_counter();row=advance_publish(c,store,rows,frame_builder=(lambda st:c.frame(cache)) if i==count-1 else None);rows.append(row);print('PRESSURE_STEP',cells,c.state.step,time.perf_counter()-tick,flush=True)
    if cells==2:
        before=c.state.digest();pointer=read(store.pointer);faults=[]
        def fail_before(stage,state):raise ValueError('controlled pressure precommit')
        def fail_pointer(stage):
            if stage=='before_pointer':raise OSError('controlled pressure pointer failure')
        for label,kw in [('before_commit',dict(inject_step=fail_before)),('before_pointer',dict(inject_store=fail_pointer))]:
            try:advance_publish(c,store,rows,**kw)
            except (ValueError,OSError):pass
            else:raise AssertionError('fault was not observed')
            if before!=c.state.digest() or read(store.pointer)!=pointer:raise ValueError('partial pressure rollback')
            faults.append(dict(stage=label,complete_state_unchanged=True))
        for _ in range(2):c.step()
        final=c.state;np.savez_compressed(run/'S3/restart-witness.npz',q=final.q,v=final.velocity,predictor=final.predictor)
        write(run/'S3/restart-witness.json',dict(fluid=final.child_states['fluid'],step=final.step,time_s=final.time,base_digest=before,extra_witness_steps=2,faults=faults))
    else:closure(run,cells)

def resume(run):
    run=Path(run);folder=run/'cases/pressure-2';identity=read(folder/'identity.json')
    if identity['numerical_source_sha256']!=source_files() or identity['fixture_source_sha256']!=sha(Path(__file__)):raise ValueError('pressure source drift')
    store=GenerationStore(folder,identity);saved=store.load();c,m,cfg=coupled(run,2,saved['state']);rows=saved['rows'];cache=CachedProbes(m);witness=read(run/'S3/restart-witness.json')
    if c.state.step!=2 or c.state.digest()!=witness['base_digest']:raise ValueError('wrong complete restart state')
    errors=None
    while c.state.step<len(c.times)-1:
        def after(stage):
            if stage=='after_pointer':raise OSError('controlled postcommit observer')
        i=c.state.step+1;tick=time.perf_counter();rows.append(advance_publish(c,store,rows,frame_builder=(lambda st:c.frame(cache)) if i in (4,len(c.times)-1) else None,inject_store=after if i==4 else None));print('PRESSURE_STEP',2,c.state.step,time.perf_counter()-tick,flush=True)
        if c.state.step==4:
            actual=c.state;errors={}
            with np.load(run/'S3/restart-witness.npz') as z:
                for key,name in [('q','q'),('velocity','v'),('predictor','predictor')]:errors[key]=float(np.max(abs(getattr(actual,key)-z[name])))
            for key in ('pressure_Pa','flux_interval_m3_s','content_m3','cumulative_boundary_m3','cumulative_source_m3'):errors[key]=float(np.max(abs(np.asarray(actual.child_states['fluid'][key])-witness['fluid'][key])))
            if max(errors.values())>1e-8:raise ValueError('pressure restart differs')
    write(run/'S3/rollback-restart.json',dict(status='passed_scoped',actual_new_process=True,errors=errors,after_pointer_accepted_once=True,faults=witness['faults']))
    closure(run,2)

def closure(run,cells):
    run=Path(run);folder=run/'cases'/f'pressure-{cells}';store=GenerationStore(folder,read(folder/'identity.json'));h=store.history();last=h[-1];rows=last['rows'];f=last['state'].child_states['fluid'];f0=h[0]['state'].child_states['fluid']
    mass=float(np.sum(np.array(f['content_m3'])-f0['content_m3'])+f['cumulative_boundary_m3']-sum(f['cumulative_source_m3']))
    if abs(mass)>1e-10 or max(r['true_scaled_residual'] for r in rows)>1 or min(r['darcy_dissipation_J'] for r in rows)<0:raise ValueError('pressure original physical closure failed')
    report=dict(status='passed_scoped',cells=cells,faces=5*cells+1,steps=len(rows),end_s=last['state'].time,cumulative_mass_defect_m3=mass,max_true_residual_fraction=max(r['true_scaled_residual'] for r in rows),max_energy_defect_J=max(abs(r['energy_balance_J']) for r in rows),max_pressure_work_defect_J=max(max(abs(np.asarray(r['pressure_work_defect_J']))) for r in rows),min_detF=min(r['min_detF'] for r in rows),actual_linear_calls=sum(len(r['linear_scaling']) for r in rows))
    write(run/('S3/two-cell-coupled.json' if cells==2 else 'S3/four-cell-coupled.json'),report);write(folder/'ledger.json',rows);print('PRESSURE_CLOSURE',report,flush=True)

def compare_grids(run):
    from benchmarks.research_sequential_next.compare import metric
    run=Path(run);a,b=(read(run/f'S3/fixed-{c}.json') for c in (2,4));records=[]
    if a['times_s']!=b['times_s']:raise ValueError('different physical pressure times')
    boundary={2:0.,4:0.}
    signs={n:BoundedTopology([[0,1],[0,1],[0,1]],n).boundary_sign for n in (2,4)}
    for x,y in zip(a['rows'],b['rows']):
        coarse=np.asarray(x['pressure_Pa']);fine=np.asarray(y['pressure_Pa']).reshape(2,2).mean(axis=1)
        for n,row in ((2,x),(4,y)):boundary[n]+=row['dt']*float(signs[n]@np.array(row['flux_interval_m3_s']))
        records.append(dict(time_s=x['time'],pressure=metric(coarse,fine,.001,.05),content=metric(sum(x['content_m3']),sum(y['content_m3']),1e-10,.05),boundary_volume=metric(boundary[2],boundary[4],1e-10,.05)))
    passed=all(x['pressure']['passed'] and x['content']['passed'] and x['boundary_volume']['passed'] for x in records)
    write(run/'S3/grid-comparison.json',dict(status='passed_scoped' if passed else 'grid_sensitive',records=records,spatial_convergence_proof=False))
    write(run/'S3/pressure-scope-decision.json',dict(status='limited_research_scope',two_cell_time=read(run/'S3/two-cell-coupled.json'),four_cell_fixed=passed,four_cell_coupled=False,pressure_spatial_accuracy=False,production_C_E_integration=False,coupled_q5=False,pure_solid_default=True))
    print('GRID_COMPARISON',passed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['topology','fixed','start','resume','compare']);p.add_argument('--cells',type=int,choices=[2,4],default=2);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase in ('fixed','start'):globals()[a.phase](a.run,a.cells)
        else:{'topology':topology,'resume':resume,'compare':compare_grids}[a.phase](a.run)
