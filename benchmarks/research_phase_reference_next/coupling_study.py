"""Bounded global RT0 assembly, four-step transients, actual solid transactions."""
from pathlib import Path
import argparse,copy
import numpy as np
import scipy.linalg as la
from .provenance import read,write,sha,digest,register,verify,source_files,serial_lock
from . import config
from .spaces import load_selected
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_phase_stress_next.coupled import advance_publish
from engine.aniso_phase1.research_local_span_next.rt0 import TwoCellTopology,TensorCellGeometry,MOBILITY
from engine.aniso_phase1.research_phase_reference_next.scaled_rt0 import ScaledTensorGridCoupling


def quadrature(topology,order=3):
    x,w=np.polynomial.legendre.leggauss(order);points=[];weights=[];cells=[]
    for cell,b in enumerate(topology.cell_bounds):
        axes=[.5*(hi-lo)*x+.5*(hi+lo) for lo,hi in b];ww=[.5*(hi-lo)*w for lo,hi in b]
        points.append(np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3));weights.append((ww[0][:,None,None]*ww[1][None,:,None]*ww[2][None,None,:]).ravel());cells.extend([cell]*order**3)
    return np.concatenate(points),np.concatenate(weights),np.array(cells)


def affine_check(topology,F):
    X,w,c=quadrature(topology);FF=np.broadcast_to(F,(len(w),3,3));H,minJ=topology.assemble(X,w,c,FF);B=topology.B;grad=np.array([.03,-.02,.04]);pb=.1+topology.centres@grad;gb=topology.boundary_term(pb)
    A=np.block([[H,-B.T],[B,np.zeros((2,2))]]);rhs=np.r_[-gb,[0.,0.]];answer=la.solve(A,rhs);z,p=answer[:11],answer[11:];K=la.det(F)*la.inv(F)@MOBILITY@la.inv(F).T
    expected=topology.areas*(-K@grad)[topology.axes];pmean=np.array([.1+b.mean(axis=1)@grad for b in topology.cell_bounds]);err=float(np.max(abs(z-expected)));perr=float(np.max(abs(p-pmean)));res=float(la.norm(A@answer-rhs));D=float(z@H@z);power=float(gb@z)
    if max(err,perr,res,abs(D+power),float(la.norm(B@z)))>1e-10 or D<0:raise ValueError('full-tensor manufactured closure failed')
    constant=la.solve(A,np.r_[-topology.boundary_term(.1),[0.,0.]])
    if np.max(abs(constant[:11]))>1e-10 or np.max(abs(constant[11:]-.1))>1e-10:raise ValueError('constant pressure generated flow')
    if not np.array_equal(B[:,topology.internal].sum(axis=0),np.zeros(len(topology.internal))):raise ValueError('internal orientation mismatch')
    return dict(F=F.tolist(),K=K.tolist(),H=H.tolist(),flux=z.tolist(),expected_flux=expected.tolist(),pressure=p.tolist(),flux_error=err,pressure_error=perr,true_residual=res,mass_error=(B@z).tolist(),dissipation=D,boundary_power=power,closure=D+power,min_H_eigenvalue=float(la.eigvalsh(H)[0]),constant_flux_max=float(np.max(abs(constant[:11]))),min_detF=minJ)


def setup(run):
    import warp as wp
    run=Path(run);lock=verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');choice=read(run/'selected-space.json');r,_=load_selected(choice['package']);m=SegmentedModel(r,order=7,device='cuda:0',hold=0.)
    cfg=config.make(lock['energy_scale_J'],end=.05,space=choice['package'],mass_order=7,full_order=7,reuse_transpose_buffers=False)
    return m,cfg


def independent(run):
    run=Path(run);verify(run)
    register(run,'S4/manufactured-protocol.json',dict(cells=2,faces=11,pressures=2,full_tensor=True,affine_gradient=[.03,-.02,.04],geometry_states=2,solver='general dense saddle block',production=False,atol=1e-10))
    top=TwoCellTopology([[0,2],[0,1],[0,1]]);shear=np.array([[1.05,.12,0.],[.02,.98,.07],[0.,.03,1.02]])
    records=[affine_check(top,F) for F in [np.eye(3),shear]]
    write(run/'S4/multicell-manufactured.json',dict(status='passed_scoped',records=records,continuous_spatial_accuracy=False))
    write(run/'S4/global-face-audit.json',dict(status='passed_scoped',faces=top.faces.tolist(),B=top.B.tolist(),boundary=top.boundary.tolist(),internal=top.internal.tolist(),internal_flux_single_owned=True,all_three_directions_nonzero=True))
    write(run/'S4/block-layout.json',dict(flux=11,pressure=2,local_order=['-x','+x','-y','+y','-z','+z'],global_normal='+axis',solver='general dense; mixed block not SPD'))
    (run/'S4/three-dimensional-interface.md').write_text('''# 两单元完整张量RT0接口

每个单元有六个局部向外面，内部共享一个全局通量。全局法向取正坐标方向，局部符号为[-,+,-,+,-,+]。H由完整Fᵀ(k/μ)⁻¹F/J和单位面通量RT0基函数积分构造，不使用旧x向标量导通近似。

Hz-Bᵀp=-b_boundary；Δm+hBz=hs；m=α(V-V0)+diag(S V0)p。全部体积、面面积与源项取实际参考几何。储存矩阵固定于参考体积。固体沿用完整M7和原AVF。两Gauss体积离散梯度精确满足detF的三次多项式链式关系，压力两侧功抵消；H取实际中点F并在每次残差评价重建。

通用准Newton解线性系统、线搜索、真实混合残差和质量／能量账沿用既有实现；新的方法、面布局、H和状态身份独立命名。仅两单元四步资格，不是一般生产三维耦合、压力空间精度或单调性证明。
''')
    print('RT0_MANUFACTURED',[x['flux_error'] for x in records],flush=True)


def geometry_and_fixed(run):
    from .provenance import APP
    run=Path(run);settings=read(APP/'S5/coupled-settings.json');times=settings['times_s']
    register(run,'S4/scaling-protocol.json',dict(strategy='row/column max positive equilibration; general LU; inverse column transform',
        original_residual=True,original_energy=True,original_physics=True,linear_residual_budget='1% original block tolerance plus backward-error term',
        pressure_storages=[.2,.0002],common_times_s=times,steps_each=4,coupled_initial_steps=4,coupled_conditional_max_steps=8,grid_registered_before_stepping=True))
    m,cfg=setup(run);records=[]
    for storage in [.2,.0002]:
        c=ScaledTensorGridCoupling(m,cfg,times,storage=storage,fixed_solid=True);geo=c.geometry.evaluate(m.rest().q);top=c.geometry.topology
        C=storage*top.V0;L=top.B@la.solve(geo['H'],top.B.T,assume_a='pos');rate=L/C[:,None];rows=[];errors=[]
        for k in range(4):
            row=c.step();rows.append(row);ref=np.full(2,.002)+la.expm(-rate*times[k+1])@np.full(2,.008);errors.append(float(np.max(abs(np.array(row['pressure_Pa'])-ref))))
        f=c.state.child_states['fluid'];mass=float(np.sum(np.array(f['content_m3'])-C*.01)+f['cumulative_boundary_m3'])
        if max(errors)>1e-3 or abs(mass)>1e-10 or min(x['darcy_dissipation_J'] for x in rows)<0 or not all(x['linear_scaling'] for x in rows):raise ValueError('scaled actual fixed-solver gate failed')
        records.append(dict(storage=storage,times_s=times,rows=rows,matrix_exponential_max_error_Pa=max(errors),cumulative_mass_defect_m3=mass))
    write(run/'S4/fixed-common-time.json',dict(status='passed_scoped',records=records,identical_time_nodes=True,pressure_spatial_accuracy=False))
    extended=[float(k*times[1]) for k in range(9)]
    write(run/'S4/coupled-settings.json',dict(**{k:v for k,v in settings.items() if k!='times_s'},times_s=extended))
    write(run/'S4/scaling-validation.json',dict(status='passed_scoped',actual_solve_calls=sum(len(x['linear_scaling']) for r in records for x in r['rows']),
        records=[dict(storage=r['storage'],linear=[x for row in r['rows'] for x in row['linear_scaling']]) for r in records],original_nonlinear_residual_used=True,mixed_system_SPD=False))
    print('SCALED_FIXED',[x['matrix_exponential_max_error_Pa'] for x in records],flush=True)


def coupled(run,state=None):
    m,cfg=setup(run);settings=read(Path(run)/'S4/coupled-settings.json');c=ScaledTensorGridCoupling(m,cfg,settings['times_s'],storage=settings['storage'],source_m3_s=settings['source_m3_s'],state=state)
    return c,m,cfg


def seed(run):
    run=Path(run);register(run,'S4/coupled-protocol.json',dict(steps=4,conditional_extension_steps=4,restart_after=2,method='full tensor RT0, two pressure cells, 11 face fluxes',mass=7,full_material=7,
        source='actual V0 per cell times [0.001,0] per second',reference_extra_steps=2,no_q5=True))
    c,m,cfg=coupled(run);folder=run/'cases/coupled-rt0-small2';cache=CachedProbes(m);identity=dict(schema='phase-reference-scaled-rt0-case-v1',coupling=c.identity,numerical_source_sha256=source_files(),fixture_source_sha256=sha(Path(__file__)),input_lock_sha256=sha(run/'input-lock.json'))
    write(folder/'identity.json',identity);write(folder/'execution-protocol.json',dict(config=cfg,times=c.times.tolist(),coupling=c.identity));store=GenerationStore(folder,identity);store.save(c.state,[],frame=c.frame(cache));rows=[]
    for i in range(2):rows.append(advance_publish(c,store,rows,frame_builder=(lambda st:c.frame(cache)) if i==1 else None))
    before=c.state.digest();pointer=read(store.pointer);faults=[]
    def before_commit(stage,state):raise ValueError('controlled RT0 precommit failure')
    try:advance_publish(c,store,rows,inject_step=before_commit)
    except ValueError:pass
    else:raise AssertionError('expected precommit fault')
    if c.state.digest()!=before or read(store.pointer)!=pointer:raise ValueError('RT0 precommit partial state')
    faults.append(dict(stage='before_commit',complete_digest_unchanged=True))
    def before_pointer(stage):
        if stage=='before_pointer':raise OSError('controlled RT0 publication error')
    try:advance_publish(c,store,rows,inject_store=before_pointer)
    except OSError:pass
    else:raise AssertionError('expected prepublication fault')
    if c.state.digest()!=before or read(store.pointer)!=pointer:raise ValueError('RT0 publication rollback failed')
    faults.append(dict(stage='before_pointer',complete_digest_unchanged=True))
    # Two continuation steps are a numerical restart witness, not a second committed cycle.
    for _ in range(2):c.step()
    final=c.state;np.savez_compressed(run/'S4/uninterrupted-continuation.npz',q=final.q,v=final.velocity,predictor=final.predictor)
    write(run/'S4/uninterrupted-continuation.json',dict(fluid=final.child_states['fluid'],step=final.step,time_s=final.time,base_committed_digest=before,extra_reference_steps=2))
    write(run/'S4/rollback-check.json',dict(status='passed_scoped',faults=faults,new_process_expected_start=before,source=source_files()))
    print('RT0_SEEDED',rows[-1]['true_scaled_residual'],flush=True)


def resume(run):
    run=Path(run);folder=run/'cases/coupled-rt0-small2';identity=read(folder/'identity.json')
    if identity['numerical_source_sha256']!=source_files() or identity['fixture_source_sha256']!=sha(Path(__file__)):raise ValueError('source changed across RT0 restart')
    store=GenerationStore(folder,identity);saved=store.load();c,m,cfg=coupled(run,saved['state']);cache=CachedProbes(m);rows=saved['rows']
    if c.state.step!=2 or c.state.digest()!=read(run/'S4/rollback-check.json')['new_process_expected_start']:raise ValueError('incorrect resumed full state')
    rows.append(advance_publish(c,store,rows))
    def after_pointer(stage):
        if stage=='after_pointer':raise OSError('controlled postcommit observer error')
    rows.append(advance_publish(c,store,rows,frame_builder=lambda st:c.frame(cache),inject_store=after_pointer))
    witness=read(run/'S4/uninterrupted-continuation.json');actual=c.state;errors={}
    with np.load(run/'S4/uninterrupted-continuation.npz') as z:
        for key,name in [('q','q'),('velocity','v'),('predictor','predictor')]:errors[key]=float(np.max(abs(getattr(actual,key)-z[name])))
    for key in ['pressure_Pa','flux_interval_m3_s','content_m3','cumulative_source_m3','cumulative_boundary_m3']:
        errors[key]=float(np.max(abs(np.asarray(actual.child_states['fluid'][key])-witness['fluid'][key])))
    h=store.history();first=h[0]['state'].child_states['fluid'];last=actual.child_states['fluid'];mass=float(np.sum(np.asarray(last['content_m3'])-first['content_m3'])+last['cumulative_boundary_m3']-sum(last['cumulative_source_m3']))
    if len(h)!=5 or max(errors.values())>1e-8 or abs(mass)>1e-10 or max(r['true_scaled_residual'] for r in rows)>1 or min(r['darcy_dissipation_J'] for r in rows)<0:raise ValueError('RT0 coupled restart/physics failed')
    result=dict(status='passed_scoped',steps=4,end_s=actual.time,actual_new_process=True,errors=errors,cumulative_mass_defect_m3=mass,min_dissipation_J=min(r['darcy_dissipation_J'] for r in rows),max_energy_defect_J=max(abs(r['energy_balance_J']) for r in rows),max_pressure_work_defect_J=max(float(np.max(np.abs(r['pressure_work_defect_J']))) for r in rows),max_scaled_residual=max(r['true_scaled_residual'] for r in rows),after_pointer_accepted_once=True,numeric_sources=source_files(),extra_uncommitted_reference_steps=2)
    write(run/'S4/coupled-closure.json',result);write(run/'S4/rollback-restart.json',result);write(folder/'ledger.json',rows)
    write(run/'S4/coupling-decision.json',dict(status='passed_scoped_two_cell_RT0_coupling',production_C_E_integration=False,pressure_spatial_accuracy=False,pressure_general_monotonicity=False,coupled_q5=False,pure_solid_default=True,physical_scope='two reference-aligned hexahedra, all 11 full-tensor face fluxes, four steps'))
    print('RT0_RESTART',errors,mass,flush=True)

def extend(run):
    run=Path(run)
    if read(run/'S4/coupled-closure.json')['status']!='passed_scoped':raise ValueError('first four coupled steps not qualified')
    register(run,'S4/extension-decision.json',dict(trigger='first four steps and actual restart passed original residual/energy/mass gates',new_steps=4,new_cells=0,new_materials=0,scope='same two-cell fixture through eight steps'))
    folder=run/'cases/coupled-rt0-small2';identity=read(folder/'identity.json')
    if identity['numerical_source_sha256']!=source_files() or identity['fixture_source_sha256']!=sha(Path(__file__)):raise ValueError('source drift')
    store=GenerationStore(folder,identity);saved=store.load();c,m,cfg=coupled(run,saved['state']);rows=saved['rows'];cache=CachedProbes(m)
    if c.state.step!=4:raise ValueError('extension requires exactly four committed steps')
    for i in range(4):rows.append(advance_publish(c,store,rows,frame_builder=(lambda st:c.frame(cache)) if i==3 else None))
    history=store.history();f=c.state.child_states['fluid'];f0=history[0]['state'].child_states['fluid']
    mass=float(np.sum(np.array(f['content_m3'])-f0['content_m3'])+f['cumulative_boundary_m3']-sum(f['cumulative_source_m3']))
    if len(history)!=9 or abs(mass)>1e-10 or max(r['true_scaled_residual'] for r in rows)>1 or min(r['min_detF'] for r in rows)<=.1 or min(r['darcy_dissipation_J'] for r in rows)<0:raise ValueError('coupled extension failed')
    write(folder/'ledger.json',rows)
    write(run/'S4/coupled-eight-step.json',dict(status='passed_scoped',steps=8,end_s=c.state.time,cumulative_mass_defect_m3=mass,max_energy_defect_J=max(abs(r['energy_balance_J']) for r in rows),max_pressure_work_defect_J=max(max(abs(np.array(r['pressure_work_defect_J']))) for r in rows),max_scaled_residual=max(r['true_scaled_residual'] for r in rows),min_detF=min(r['min_detF'] for r in rows),actual_linear_calls=sum(len(r['linear_scaling']) for r in rows),source_sha256=source_files()))
    write(run/'S4/coupling-decision.json',dict(status='passed_scoped_two_cell_scaled_RT0_eight_steps',pressure_scaling_adopted=True,production_C_E_integration=False,pressure_spatial_accuracy=False,pressure_general_monotonicity=False,coupled_q5=False,pure_solid_default=True,physical_scope='same two aligned reference hexahedra, 11 full-tensor fluxes, eight microsecond-scale steps'))
    print('SCALED_COUPLED_EIGHT',mass,rows[-1]['true_scaled_residual'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['fixed','seed','resume','extend']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):{'fixed':geometry_and_fixed,'seed':seed,'resume':resume,'extend':extend}[a.phase](a.run)
