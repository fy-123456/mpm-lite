"""Small-storage pressure time diagnosis and actual explicit-grid transactions."""
from pathlib import Path
import argparse, copy
import numpy as np
import scipy.linalg as la
from .provenance import APP,read,write,sha,digest,register,source_files,snapshot,serial_lock,verify
from . import config
from .spaces import load_selected
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric,regions
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_phase_stress_next.coupled import ExplicitGridCoupling,advance_publish
from engine.aniso_phase1.research_post_release.fields import CachedProbes


def model_config(run):
    import warp as wp
    run=Path(run);verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    choice=read(run/'selected-space.json');cfg=config.make(read(run/'input-lock.json')['energy_scale_J'],dt=.01,end=.04,
        space=choice['package'],mass_order=choice['mass_order'],full_order=choice['full_order'])
    r,_=load_selected(choice['package'])
    return SegmentedModel(r,order=choice['full_order'],device='cuda:0',hold=0.),cfg


def matrices(cells,storage):
    length=.75/cells;area=.0625;H=np.diag(np.r_[np.full(cells-1,length/(area*.1)),.5*length/(area*.1)])
    B=np.zeros((cells,cells))
    for k in range(cells):
        B[k,k]=1.
        if k:B[k,k-1]=-1.
    C=storage*length*area*np.eye(cells);gb=np.zeros(cells);gb[-1]=.002
    L=B@la.solve(H,B.T,assume_a='pos');b=B@la.solve(H,gb,assume_a='pos')
    return C,H,B,L,b


def independent(cells,storage,h):
    C,H,B,L,b=matrices(cells,storage);A=-la.solve(C,L);peq=la.solve(L,b);p=np.full(cells,.01)
    G=la.solve(C+.5*h*L,C-.5*h*L);boundary=la.solve(C+.5*h*L,h*b);records=[]
    for k in range(1,5):
        p=G@p+boundary;exact=peq+la.expm(A*k*h)@(np.full(cells,.01)-peq)
        records.append(dict(time=k*h,p=p.tolist(),exact=exact.tolist(),error_Pa=float(np.max(abs(p-exact)))))
    off=A.copy();np.fill_diagonal(off,0)
    return dict(cells=cells,storage=storage,dt=h,bound=float(2*np.min(np.diag(C)/np.diag(L))),
        min_C=float(np.min(np.diag(C))),min_H_eigen=float(la.eigvalsh(H)[0]),min_generator_offdiag=float(off.min()),
        min_G=float(G.min()),min_boundary=float(boundary.min()),max_error_Pa=max(x['error_Pa'] for x in records),
        raw_min_pressure_Pa=min(min(x['p']) for x in records),records=records,
        positive_update=bool(G.min()>=-1e-14 and boundary.min()>=-1e-14))


def diagnose(run):
    run=Path(run);verify(run)
    register(run,'S4/time-protocol.json',dict(cells=[2,4],normal_storage=.2,small_storage=.0002,
        steps=4,initial_pressure_Pa=.01,reservoir_Pa=.002,fixed_solid_source=0.,
        method='unchanged midpoint; h=min(.01, .2*2*min(Cii/Lii))',factor=.2,
        reason='single conservative positivity candidate with margin for transient accuracy',no_pressure_clipping=True,
        exact_reference='small matrix exponential of independent half-cell discretization'))
    records=[]
    for cells in (2,4):
        for S in (.2,.0002):
            old=independent(cells,S,.01);h=min(.01,.2*old['bound']);new=independent(cells,S,h)
            if not new['positive_update'] or new['raw_min_pressure_Pa']<.002-1e-5 or new['max_error_Pa']>5e-4:raise ValueError('chosen pressure time grid fails independent check')
            records.append(dict(cells=cells,storage=S,old=old,selected=new,times=[i*h for i in range(5)]))
    write(run/'S4/small-storage-diagnostic.json',dict(status='diagnosed',records=records,
        scope='fixed solid, no source, x-only diffusion; mixed rank does not imply midpoint positivity'))
    write(run/'S4/monotonicity-and-time.json',dict(status='passed_scoped',records=records,
        small_storage_selected_dt_s=records[1]['selected']['dt'],normal_storage_dt_s=.01,general_coupled_positivity=False))
    print('PRESSURE_GRID',[(x['cells'],x['storage'],x['selected']['dt'],x['old']['min_G'],x['selected']['max_error_Pa']) for x in records],flush=True)


def spec(run,name):
    small=name=='coupled-small2';h=read(Path(run)/'S4/monotonicity-and-time.json')['small_storage_selected_dt_s'] if small else .01
    return dict(times=[i*h for i in range(5)],storage=.0002 if small else .2,drained=name!='coupled-closed2',reservoir=.002 if name!='coupled-closed2' else 0.)


def build(m,cfg,specification,state=None):
    spec=copy.deepcopy(specification);times=spec.pop('times');volume=.75*.0625/2
    return ExplicitGridCoupling(m,cfg,times,source_m3_s=[volume*.001,0.],state=state,**spec)


def advance(c,store,rows,cache,steps):
    for _ in range(steps):rows.append(advance_publish(c,store,rows,frame_builder=lambda state:c.frame(cache)))
    return rows


def prepare(run):
    run=Path(run);register(run,'S4/transaction-protocol.json',dict(steps=4,restart_after=2,
        owned='q/v/predictor, p/content/flux/source/boundary history and explicit pressure grid identity',
        faults=['corrupted owned fields before commit','before pointer','after pointer'],coupled_q5=False))
    m,cfg=model_config(run);cache=CachedProbes(m)
    fixed=[]
    for cells in (2,4):
        C,H,B,L,b=matrices(cells,.0002);h=min(.01,.2*2*np.min(np.diag(C)/np.diag(L)))
        c=ExplicitGridCoupling(m,cfg,[i*h for i in range(5)],cells=cells,storage=.0002,drained=True,fixed_solid=True,reservoir=.002)
        real=c.geometry.evaluate(m.rest().q);actualH=real['H'][np.ix_(c.active,c.active)]
        assert np.max(abs(actualH-H))<1e-8 and np.max(abs(c.capacity-np.diag(C)))<1e-14
        expected=independent(cells,.0002,h);rr=[]
        for k in range(4):
            rr.append(c.step());assert np.max(abs(np.array(rr[-1]['pressure_Pa'])-expected['records'][k]['p']))<1e-8
        fixed.append(dict(cells=cells,times=c.times.tolist(),rows=rr,analytic=expected))
    write(run/'S4/explicit-grid-check.json',dict(status='passed_scoped',actual_driver_matches_independent_midpoint=True,records=fixed))
    for name in ['coupled-closed2','coupled-drained2','coupled-small2']:
        ss=spec(run,name);c=build(m,cfg,ss);folder=run/'cases'/name
        identity=dict(schema='phase-stress-coupled-case-v1',input_lock_sha256=sha(run/'input-lock.json'),coupling=c.identity,
            numerical_source_sha256=source_files(),fixture_source_sha256=sha(Path(__file__)))
        write(folder/'identity.json',identity);write(folder/'execution-protocol.json',dict(config=cfg,spec=ss,steps=4))
        snapshot(folder/'source',source_files());store=GenerationStore(folder,identity);store.save(c.state,[],frame=c.frame(cache))
        rows=advance(c,store,[],cache,2 if name=='coupled-small2' else 4);write(folder/'ledger.json',rows)
        print('COUPLED_PREPARED',name,c.state.step,c.state.time,flush=True)
    c=build(m,cfg,spec(run,'coupled-small2'));before=c.state.digest()
    def fault(where,state):
        state.q[:]=99;state.velocity[:]=98;state.predictor[:]=97
        f=state.child_states['fluid']
        for key in ('pressure_Pa','content_m3','flux_interval_m3_s','cumulative_source_m3'):f[key][0]=999.
        f['cumulative_boundary_m3']=999.;raise RuntimeError('controlled all-owned-field failure')
    try:c.step(inject=fault)
    except RuntimeError:pass
    else:raise AssertionError('expected precommit fault')
    assert c.state.digest()==before
    folder=run/'S4/publication-fixture';store=GenerationStore(folder,dict(grid=c.identity));store.save(c.state,[]);pointer=read(store.pointer)
    def disk_fail(where):
        if where=='before_pointer':raise OSError('controlled prepublication failure')
    try:advance_publish(c,store,[],inject_store=disk_fail)
    except OSError:pass
    else:raise AssertionError('expected disk failure')
    assert c.state.digest()==before and read(store.pointer)==pointer and len(store.history())==1
    def observer(where):
        if where=='after_pointer':raise OSError('controlled postpublication observer')
    row=advance_publish(c,store,[],inject_store=observer)
    assert c.state.step==1 and len(store.history())==2
    for key in ('times','source'):
        changed=spec(run,'coupled-small2');changed['times']=[x*2 for x in changed['times']]
        try:build(m,cfg,changed,state=c.state)
        except ValueError:pass
        else:raise AssertionError('foreign grid accepted')
        break
    write(run/'S4/publication-check.json',dict(status='passed_scoped',before_pointer_rollback=True,after_pointer_once=True,foreign_time_grid_rejected=True))
    write(run/'S4/rollback-check.json',dict(status='passed_scoped',all_owned_fields=True,unchanged_digest=before))


def cycle(run):
    run=Path(run);folder=run/'cases/coupled-small2';identity=read(folder/'identity.json')
    if identity['numerical_source_sha256']!=source_files() or identity['fixture_source_sha256']!=sha(Path(__file__)):raise ValueError('coupling source changed')
    m,cfg=model_config(run);store=GenerationStore(folder,identity);data=store.load();c=build(m,cfg,spec(run,'coupled-small2'),data['state'])
    assert c.state.step==2
    rows=advance(c,store,data['rows'],CachedProbes(m),2);write(folder/'ledger.json',rows)
    print('SMALL_STORAGE_NEW_PROCESS',c.state.step,c.state.time,flush=True)


def final_check(run):
    run=Path(run);m,cfg=model_config(run);cache=CachedProbes(m);records=[]
    for name in ['coupled-closed2','coupled-drained2','coupled-small2']:
        folder=run/'cases'/name;h=GenerationStore(folder,read(folder/'identity.json')).history();saved=h[-1]['state'];rows=h[-1]['rows']
        if name=='coupled-small2':
            direct=build(m,cfg,spec(run,name))
            for _ in range(4):direct.step()
            expected=direct.state
        else:
            parent=APP/'cases'/name;expected=GenerationStore(parent,read(parent/'identity.json')).load()['state']
        errors={key:float(np.max(abs(getattr(saved,key)-getattr(expected,key)))) for key in ('q','velocity','predictor')}
        for key in ('pressure_Pa','flux_interval_m3_s','content_m3','cumulative_source_m3','cumulative_boundary_m3'):
            errors[key]=float(np.max(abs(np.array(saved.child_states['fluid'][key])-expected.child_states['fluid'][key])))
        assert max(errors.values())<1e-8 and len(h)==5
        initial=h[0]['state'].child_states['fluid'];f=saved.child_states['fluid']
        closure=float(sum(np.array(f['content_m3'])-initial['content_m3'])+f['cumulative_boundary_m3']-sum(f['cumulative_source_m3']))
        assert abs(closure)<1e-10
        report=dict(status='passed_scoped',case=name,steps=4,final_time_s=saved.time,errors=errors,cumulative_closure_m3=closure,
            min_detF=min(x['min_detF'] for x in rows),max_mass_defect_m3=max(np.max(np.abs(x['mass_defect_m3'])) for x in rows),
            max_energy_balance_J=max(abs(x['energy_balance_J']) for x in rows),min_darcy_J=min(x['darcy_dissipation_J'] for x in rows),
            max_pressure_work_defect_J=max(np.max(np.abs(x['pressure_work_defect_J'])) for x in rows),
            min_raw_pressure_Pa=min(min(x['pressure_Pa']) for x in rows),max_true_scaled_residual=max(x['true_scaled_residual'] for x in rows))
        assert report['min_detF']>0 and report['min_darcy_J']>=0 and report['max_true_scaled_residual']<=1
        with np.load(h[-1]['folder']/'frame.npz') as z:
            report['regional_stress']={region:{key:float(np.sqrt(np.sum(w*np.sum(z[key]**2,axis=(-2,-1)))/np.sum(w)))
                for key in ('solid_PK1','total_PK1')} for region,w in regions(z['X']).items()}
        write(folder/'summary.json',report);records.append(report)
    inherited=['P4/coupling-kernel-check.json','P4/mixed-rank-solver.json','P4/limits-and-closure.json']
    write(run/'S4/coupled-checks.json',dict(status='passed_scoped',records=records,actual_new_process_restart=True,
        unchanged_physics_checks=[dict(path=str(APP/x),sha256=sha(APP/x)) for x in inherited]))
    write(run/'S4/coupling-decision.json',dict(status='passed_scoped',actual_explicit_grid=True,normal_storage_dt_s=.01,
        small_storage=.0002,small_storage_dt_s=spec(run,'coupled-small2')['times'][1],steps=4,
        pressure_scheme='x-oriented half-cell resistance, transverse flow closed',pressure_cells=2,
        small_storage_fixed_checks_cells=[2,4],general_coupled_monotonicity=False,pressure_spatial_accuracy=False,
        pure_solid_default=True,coupled_q5=False,source_sha256=source_files()))
    print('COUPLED_FINAL',records,flush=True)


def render(run,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);out=Path(output)
    if (run/'release.json').exists() and (out.resolve()==run.resolve() or run.resolve() in out.resolve().parents):raise ValueError('sealed output requires external destination')
    out.mkdir(parents=True,exist_ok=True);fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for name in ['coupled-closed2','coupled-drained2','coupled-small2']:
        folder=run/'cases'/name;h=GenerationStore(folder,read(folder/'identity.json')).history();rows=h[-1]['rows'];ts=[r['time'] for r in rows]
        ax=axes[0,1] if name=='coupled-small2' else axes[0,0]
        for k in range(2):ax.plot(ts,[r['pressure_Pa'][k] for r in rows],'.-',label=name+f' cell{k}')
        axes[1,0].plot(ts,[r['energy_balance_J'] for r in rows],'.-',label=name)
        with np.load(h[-1]['folder']/'frame.npz') as z:
            mid=z['X'].shape[2]//2;axes[1,1].scatter(z['X'][:,:,mid,0].ravel(),z['total_PK1'][:,:,mid,0,0].ravel(),s=7,label=name)
    for ax,title in zip(axes.ravel(),['normal storage pressure (Pa)','small storage pressure (Pa)','energy balance (J)','total PK1 P11 (Pa)']):
        ax.set(title=title,xlabel='reference x (m)' if ax is axes[1,1] else 'actual time (s)');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(out/'coupled-summary.png',dpi=140);plt.close(fig)
    (out/'index.html').write_text('<meta charset="utf-8"><h1>实际时间网格：四步压力耦合</h1><p>小储存采用更短物理时间；原始压力、总 PK1 和能量残差。</p><img width="100%" src="coupled-summary.png">')
    print(out/'index.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['diagnose','prepare','cycle','check','render']);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='render':render(a.run,a.output or a.run/'S4/visualization')
        else:{'diagnose':diagnose,'prepare':prepare,'cycle':cycle,'check':final_check}[a.phase](a.run)
