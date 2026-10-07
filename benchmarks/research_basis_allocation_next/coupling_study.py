"""Common-time pressure evidence and an independent RT0 full-tensor fixture."""
from pathlib import Path
import argparse,copy
import numpy as np
import scipy.linalg as la
from .provenance import APP,read,write,sha,register,verify,source_files,snapshot,serial_lock
from . import config
from .spaces import load_selected
from benchmarks.research_phase_stress_next.coupling_study import matrices,independent
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_phase_stress_next.coupled import ExplicitGridCoupling,advance_publish
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from benchmarks.research_sequential_next.checkpoint import GenerationStore


def setup(run):
    import warp as wp
    run=Path(run);lock=verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');choice=read(run/'selected-space.json')
    cfg=config.make(lock['energy_scale_J'],dt=.01,end=.04,space=choice['package'],mass_order=7,full_order=7)
    r,_=load_selected(choice['package']);m=SegmentedModel(r,order=7,device='cuda:0',hold=0.)
    return m,cfg


def common_time(run):
    run=Path(run);verify(run)
    register(run,'S5/common-time-protocol.json',dict(cells=[2,4],steps=4,dt=9.375e-6,end_s=3.75e-5,storage=.0002,
        initial_pressure=.01,reservoir=.002,source=0.,fixed_solid=True,compare='volume averages at identical time; each versus own matrix exponential'))
    m,cfg=setup(run);records=[]
    for cells in (2,4):
        h=9.375e-6;ref=independent(cells,.0002,h);c=ExplicitGridCoupling(m,cfg,[k*h for k in range(5)],cells=cells,storage=.0002,drained=True,fixed_solid=True,reservoir=.002)
        C,H,B,L,b=matrices(cells,.0002);geometry=c.geometry.evaluate(m.rest().q)
        if np.max(abs(geometry['H'][np.ix_(c.active,c.active)]-H))>1e-8:raise ValueError('analytic pressure matrix mismatch')
        rows=[];cumulative=0.
        for k in range(4):
            row=c.step();rows.append(row)
            if np.max(abs(np.asarray(row['pressure_Pa'])-ref['records'][k]['p']))>1e-8:raise ValueError('actual grid differs from independent midpoint')
            cumulative+=h*np.asarray(row['flux_interval_m3_s'])[-1]
        records.append(dict(cells=cells,times=c.times.tolist(),rows=rows,analytic=ref,cumulative_boundary_m3=float(cumulative)))
    p2=np.array(records[0]['rows'][-1]['pressure_Pa']);p4=np.array(records[1]['rows'][-1]['pressure_Pa']);restricted=p4.reshape(2,2).mean(axis=1)
    write(run/'S5/pressure-time-vs-space.json',dict(status='passed_scoped_time_comparison',records=records,
        spatial_sensitivity=dict(two_cell=p2.tolist(),four_cell_volume_mean=restricted.tolist(),difference_Pa=(p2-restricted).tolist(),
            volume_weighted_rms_Pa=float(la.norm(p2-restricted)/np.sqrt(2))),
        continuous_spatial_accuracy=False,scope='two levels at same time; matrix exponential is each half-discrete time reference'))
    print('PRESSURE_COMMON',p2,restricted,flush=True)


def coupled(run,state=None):
    m,cfg=setup(run);times=[k*3.75e-5 for k in range(5)]
    # Same x-half-cell physics; obtain source normalization from current reference geometry.
    probe=ExplicitGridCoupling(m,cfg,times,storage=.0002,drained=True,reservoir=.002)
    volume=float(probe.geometry.V0[0]);del probe
    c=ExplicitGridCoupling(m,cfg,times,storage=.0002,drained=True,reservoir=.002,source_m3_s=[volume*.001,0.],state=state)
    return c,m,cfg


def seed(run):
    run=Path(run);register(run,'S5/coupled-protocol.json',dict(steps=4,restart_after=2,storage=.0002,dt=3.75e-5,source_unit='reference volume times 0.001 per second',
        full_order=7,mass_order=7,pressure_scheme='unchanged x-half-cell',normal_fixtures='authenticated inheritance; same space and solver'))
    c,m,cfg=coupled(run);folder=run/'cases/coupled-small2';cache=CachedProbes(m)
    ident=dict(schema='basis-allocation-coupled-case-v1',input_lock_sha256=sha(run/'input-lock.json'),coupling=c.identity,numerical_source_sha256=source_files(),fixture_source_sha256=sha(Path(__file__)))
    write(folder/'identity.json',ident);snapshot(folder/'source',source_files());store=GenerationStore(folder,ident);store.save(c.state,[],frame=c.frame(cache));rows=[]
    for _ in range(2):rows.append(advance_publish(c,store,rows,frame_builder=lambda st:c.frame(cache)))
    write(folder/'execution-protocol.json',dict(config=cfg,steps=4,times=c.times.tolist(),source=c.identity))
    # Failure before publication must restore the complete pressure and solid state.
    before=c.state.digest();pointer=read(store.pointer)
    def fail(stage):
        if stage=='before_pointer':raise OSError('controlled pressure prepublication fault')
    try:advance_publish(c,store,rows,inject_store=fail)
    except OSError:pass
    else:raise AssertionError('expected failure')
    if c.state.digest()!=before or read(store.pointer)!=pointer:raise ValueError('pressure publication rollback failed')
    write(run/'S5/rollback-check.json',dict(status='passed_scoped',complete_state_digest=before,before_pointer_unchanged=True,numeric_sources=source_files()))


def resume(run):
    run=Path(run);folder=run/'cases/coupled-small2';ident=read(folder/'identity.json')
    if ident['numerical_source_sha256']!=source_files() or ident['fixture_source_sha256']!=sha(Path(__file__)):raise ValueError('source changed during pressure restart')
    store=GenerationStore(folder,ident);saved=store.load();c,m,cfg=coupled(run,saved['state']);cache=CachedProbes(m);rows=saved['rows']
    if c.state.step!=2:raise ValueError('expected exactly two committed steps')
    for _ in range(2):rows.append(advance_publish(c,store,rows,frame_builder=lambda st:c.frame(cache)))
    parent=APP/'cases/coupled-small2';expected=GenerationStore(parent,read(parent/'identity.json')).load()['state'];actual=c.state
    errors={k:float(np.max(abs(getattr(actual,k)-getattr(expected,k)))) for k in ('q','velocity','predictor')}
    for key in ('pressure_Pa','flux_interval_m3_s','content_m3','cumulative_source_m3','cumulative_boundary_m3'):
        errors[key]=float(np.max(abs(np.asarray(actual.child_states['fluid'][key])-np.asarray(expected.child_states['fluid'][key]))))
    f0=store.history()[0]['state'].child_states['fluid'];f=actual.child_states['fluid'];closure=float(sum(np.array(f['content_m3'])-f0['content_m3'])+f['cumulative_boundary_m3']-sum(f['cumulative_source_m3']))
    if max(errors.values())>1e-8 or abs(closure)>1e-10 or min(r['darcy_dissipation_J'] for r in rows)<0 or max(r['true_scaled_residual'] for r in rows)>1:raise ValueError('coupled closure failed')
    summary=dict(status='passed_scoped',errors=errors,steps=4,time_s=actual.time,cumulative_closure_m3=closure,
        min_dissipation_J=min(r['darcy_dissipation_J'] for r in rows),max_energy_defect_J=max(abs(r['energy_balance_J']) for r in rows),
        max_pressure_work_defect_J=max(np.max(np.abs(r['pressure_work_defect_J'])) for r in rows),max_scaled_residual=max(r['true_scaled_residual'] for r in rows),
        actual_new_process=True,numeric_sources=source_files(),complete_histories_checked=True)
    write(run/'S5/coupled-closure.json',summary);write(run/'S5/rollback-restart.json',summary);write(folder/'ledger.json',rows)
    print('PRESSURE_RESTART',errors,closure,flush=True)


def report_committed(run):
    """Recover reporting after a Python list/abs error; no additional integration."""
    import ast
    run=Path(run);folder=run/'cases/coupled-small2';ident=read(folder/'identity.json')
    previous=run/'S5/coupled-protocol-source/benchmarks/research_basis_allocation_next/coupling_study.py'
    if sha(previous)!=ident['fixture_source_sha256'] or ident['numerical_source_sha256']!=source_files():raise ValueError('old fixture provenance mismatch')
    old={n.name:n for n in ast.parse(previous.read_text()).body if isinstance(n,ast.FunctionDef)}
    new={n.name:n for n in ast.parse(Path(__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    for name in ('setup','coupled','seed'):
        if ast.dump(old[name])!=ast.dump(new[name]):raise ValueError('physical fixture function changed')
    def numeric_prefix(node):
        out=[]
        for n in node.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='summary' for t in n.targets):break
            out.append(ast.dump(n))
        return out
    if numeric_prefix(old['resume'])!=numeric_prefix(new['resume']):raise ValueError('resume integration changed')
    store=GenerationStore(folder,ident);h=store.history()
    if len(h)!=5 or h[-1]['state'].step!=4:raise ValueError('completed four-step history required')
    c,m,cfg=coupled(run,h[-1]['state']);actual=c.state;rows=h[-1]['rows'];parent=APP/'cases/coupled-small2'
    expected=GenerationStore(parent,read(parent/'identity.json')).load()['state']
    errors={k:float(np.max(abs(getattr(actual,k)-getattr(expected,k)))) for k in ('q','velocity','predictor')}
    for k in ('pressure_Pa','flux_interval_m3_s','content_m3','cumulative_source_m3','cumulative_boundary_m3'):
        errors[k]=float(np.max(abs(np.asarray(actual.child_states['fluid'][k])-expected.child_states['fluid'][k])))
    f0=h[0]['state'].child_states['fluid'];f=actual.child_states['fluid'];closure=float(sum(np.array(f['content_m3'])-f0['content_m3'])+f['cumulative_boundary_m3']-sum(f['cumulative_source_m3']))
    if max(errors.values())>1e-8 or abs(closure)>1e-10 or min(r['darcy_dissipation_J'] for r in rows)<0 or max(r['true_scaled_residual'] for r in rows)>1:raise ValueError('published pressure physics failed')
    summary=dict(status='passed_scoped',errors=errors,steps=4,time_s=actual.time,cumulative_closure_m3=closure,
        min_dissipation_J=min(r['darcy_dissipation_J'] for r in rows),max_energy_defect_J=max(abs(r['energy_balance_J']) for r in rows),
        max_pressure_work_defect_J=max(float(np.max(np.abs(r['pressure_work_defect_J']))) for r in rows),max_scaled_residual=max(r['true_scaled_residual'] for r in rows),
        actual_new_process=True,numeric_sources=source_files(),complete_histories_checked=True,
        reporting_recovery=dict(cause='built-in abs on deserialized list',fix='numpy abs',old_fixture_sha256=sha(previous),new_fixture_sha256=sha(Path(__file__)),
            physical_function_AST_unchanged=True,additional_steps=0,old_checkpoint_identity_unchanged=True))
    write(run/'S5/coupled-closure.json',summary);write(run/'S5/rollback-restart.json',summary);write(folder/'ledger.json',rows)
    print('PRESSURE_COMMITTED_REPORT',errors,closure,flush=True)


def manufactured(run):
    run=Path(run);verify(run)
    text='''# 一般三维压力的最小接口（独立制造夹具）

现有 x 向半单元算法保持独立。本夹具命名 rt0-cube-full-tensor-v1，只检验一个参考立方体上的六面 RT0 通量与单元常数压力，不接入生产固体循环。

面顺序为 -x,+x,-y,+y,-z,+z；本单元流量为向外体积通量，全局网格需共享面定向和关联 B，内部两侧符号相反。参考体积 V0=1。H_ij=积分(phi_i^T K^{-1} phi_j)dX，K=J F^{-1}(k/mu)F^{-T}；不可用 1/(K^{-1})xx 代替整个张量。

混合方程 Hz-B^T p=-p_boundary，Bz=s-Delta m/h。H 正定保证 z^T H z>=0，但混合块矩阵并非固体 SPD；本夹具用通用稠密求解，并检查真实残差。生产推广还缺少多单元全局面组装、变形随时间变化的H、压力功离散梯度和完整状态事务。

制造解 p=p0+g dot X，常量通量 z_face=n dot(-K g)，每面面积1；Bz=0，边界压力功 p_boundary dot z=-z^T H z。检验交叉流量、面方向及边界功，不证明一般网格精度或单调性。
'''
    (run/'S5/three-dimensional-interface.md').write_text(text)
    register(run,'S5/manufactured-protocol.json',dict(schema='rt0-cube-full-tensor-v1',cells=1,faces=6,pressure='affine exact',quadrature=3,
        full_tensor=True,fixed_solid=True,production=False,atol=1e-10,rtol=2e-5))
    F=np.array([[1.05,.12,0.],[.02,.98,.07],[0.,.03,1.02]]);k=np.array([[.1,.02,.01],[.02,.08,-.015],[.01,-.015,.06]])
    invF=la.inv(F);K=la.det(F)*invF@k@invF.T;H=np.zeros((6,6));x,w=np.polynomial.legendre.leggauss(3);x=(x+1)/2;w=w/2
    for i in range(3):
        for j in range(3):
            for l in range(3):
                X=np.array([x[i],x[j],x[l]]);phi=np.zeros((6,3))
                for d in range(3):phi[2*d,d]=X[d]-1;phi[2*d+1,d]=X[d]
                H+=w[i]*w[j]*w[l]*(phi@la.solve(K,phi.T,assume_a='pos'))
    normals=np.vstack([-np.eye(3)[d] if side==0 else np.eye(3)[d] for d in range(3) for side in range(2)])
    centres=np.full((6,3),.5)
    for d in range(3):centres[2*d,d]=0;centres[2*d+1,d]=1
    grad=np.array([.03,-.02,.04]);pb=.1+centres@grad;B=np.ones((1,6));A=np.block([[H,-B.T],[B,np.zeros((1,1))]])
    rhs=np.r_[-pb,0.];solution=la.solve(A,rhs);z=solution[:6];expected=normals@(-K@grad);pmean=.1+np.sum(grad)/2
    err=float(np.max(abs(z-expected)));res=float(la.norm(A@solution-rhs));diss=float(z@H@z);power=float(pb@z)
    if err>1e-10 or abs(solution[-1]-pmean)>1e-10 or res>1e-10 or diss<0 or abs(diss+power)>1e-10:raise ValueError('3D tensor fixture failed')
    write(run/'S5/three-dimensional-manufactured.json',dict(status='manufactured_3D_scoped',schema='rt0-cube-full-tensor-v1',F=F.tolist(),K=K.tolist(),
        H=H.tolist(),flux=z.tolist(),analytic_flux=expected.tolist(),pressure_mean=float(solution[-1]),flux_error=err,true_residual=res,
        mass_defect=float(sum(z)),dissipation=diss,boundary_pressure_power=power,closure=diss+power,H_eigen_min=float(la.eigvalsh(H)[0]),
        offdiagonal_K_norm=float(la.norm(K-np.diag(np.diag(K)))),production_integrated=False))
    write(run/'S5/block-layout.json',dict(flux=6,pressure=1,face_order=['-x','+x','-y','+y','-z','+z'],B=B.tolist(),solver='general dense; mixed matrix is not SPD'))


def finish(run):
    run=Path(run);verify(run)
    inherited=[dict(path=str(APP/p),sha256=sha(APP/p)) for p in ['S4/coupled-checks.json','S4/publication-check.json','S4/coupling-decision.json']]
    write(run/'S5/coupling-decision.json',dict(status='passed_scoped',normal_evidence=inherited,new_common_time=[2,4],
        small_coupled_steps=4,small_coupled_dt=3.75e-5,pressure_general_monotonicity=False,pressure_spatial_accuracy=False,
        manufactured_3D_scoped=True,production_C_E_integration=False,coupled_q5=False,pure_solid_default=True))
    write(run/'S5/scope-matrix.json',dict(fixed_solid_small_common_time='passed 2/4 cells at 3.75e-5s',
        current_3D_solid_x_pressure='passed small coupled restart; normal inherited',independent_3D='one RT0 affine fixture only',
        general_3D_production=False,continuous_pressure_convergence=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['common-time','seed','resume','manufactured','finish','report-committed']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):{'common-time':common_time,'seed':seed,'resume':resume,'manufactured':manufactured,'finish':finish,'report-committed':report_committed}[a.phase](a.run)
