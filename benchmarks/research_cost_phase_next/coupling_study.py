"""Scoped two/four pressure-cell physics, mixed rank and real process restart."""
from pathlib import Path
import argparse,copy
import numpy as np
import scipy.linalg as la
from .provenance import read,write,sha,digest,register,source_files,snapshot,serial_lock,resources,verify
from . import config
from .spaces import load_selected
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric,regions
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_cost_phase_next.pressure import MultiCellAVF,CellGeometry
from engine.aniso_phase1.research_post_release.fields import CachedProbes


def model_config(run):
    import warp as wp
    run=Path(run);verify(run);wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    choice=read(run/'selected-space.json');cfg=config.make(read(run/'input-lock.json')['energy_scale_J'],dt=.01,end=.04,
        space=choice['package'],mass_order=choice['mass_order'],full_order=choice['full_order'])
    r,_=load_selected(choice['package'])
    return SegmentedModel(r,order=choice['full_order'],device='cuda:0',hold=0.),cfg


def source(c):return c.geometry.V0*.001*(np.arange(c.cells)<c.cells//2)


def analytic_flux(lengths):
    # Full six-face RT0 element: tensor cross terms are independently exercised.
    lengths=np.asarray(lengths);volume=np.prod(lengths);areas=volume/lengths
    A=np.array([[.1,.025,-.01],[.025,.08,.012],[-.01,.012,.06]]);inv=la.inv(A)
    x,w=np.polynomial.legendre.leggauss(3);u=(x+1)/2;w=w/2
    H=np.zeros((6,6));B=np.tile([-1.,1.],3)[None,:];centers=[]
    for i in range(3):
        for side in (0,1):
            center=lengths/2;center=center.copy();center[i]=side*lengths[i];centers.append(center)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                t=np.array([u[a],u[b],u[c]]);psi=np.zeros((6,3))
                for i in range(3):psi[2*i,i]=(1-t[i])/areas[i];psi[2*i+1,i]=t[i]/areas[i]
                H+=volume*w[a]*w[b]*w[c]*(psi@inv@psi.T)
    grad=np.array([.02,-.007,.009]);p=.01;flux=-A@grad;z=np.repeat(flux*areas,2)
    gb=B.ravel()*(p+(np.array(centers)-lengths/2)@grad)
    action=metric(H@z,B.ravel()*p-gb,1e-12,1e-10)
    system=np.block([[np.zeros((1,1)),B],[B.T,-H]])
    solution=la.solve(system,np.r_[0.,gb],assume_a='gen')
    expected=np.r_[p,z];error=float(np.max(abs(solution-expected)))
    if not action['passed'] or error>1e-10 or la.eigvalsh(H)[0]<=0:raise ValueError('non-diagonal RT0 manufactured flux failed')
    return dict(status='passed_scoped',A=A.tolist(),H=H.tolist(),B=B.tolist(),pressure_gradient=grad.tolist(),expected_flux=z.tolist(),
        solution_max_error=error,action=action,full_tensor_cross_terms=True,
        production_subspace='x-oriented global faces; transverse flux explicitly constrained to zero',
        legacy_difference='one-cell right-only RT0 H=L/(3 Aface mobility), old centroid resistance L/(2 Aface mobility); distinct spatial trial/boundary treatment, old fixture retained')


def checks(run):
    run=Path(run)
    register(run,'N4/physical-model.json',dict(solid=read(run/'selected-space.json'),cells=[2,4],alpha=.8,storage_Pa_inverse=.2,k_m2=1e-4,mu_Pa_s=1e-3,
        pressure0_Pa=.01,reservoir_Pa=.002,dt=.01,steps=4,source='0.001*cell reference volume per second in left half',
        mass_content_units='integrated volume m^3, not mass without density',flux_units='reference volume m^3/s',
        fixed_zero_grip=True,pressure='P0 cell means',flux='globally +x conservative half-cell resistance; transverse faces closed',
        geometry='Gauss on pressure/solid cells AND half-cell intersections',parameters='numerical fixture, not calibration',coupled_q5=False))
    m,cfg=model_config(run);c=MultiCellAVF(m,cfg);geo=c.geometry
    rng=np.random.default_rng(20261001);q=np.zeros_like(m.rest().q);q[m.free]=rng.normal(size=(len(m.free),3))*1e-5
    d=rng.normal(size=q.shape);d[m.fixed]=0.;d/=la.norm(d);eps=1e-6;g=geo.evaluate(q);p=np.array([.012,.008]);content=c.alpha*(g['volume']-geo.V0)+c.capacity*p
    def fluid(v):
        gg=geo.evaluate(v);pp=(content-c.alpha*(gg['volume']-geo.V0))/c.capacity
        return .5*np.sum(c.capacity*pp*pp),-c.alpha*np.einsum('k,kij->ij',pp,gg['gradient'])
    up,fp=fluid(q+eps*d);um,fm=fluid(q-eps*d);force=fluid(q)[1]
    derivative=metric((up-um)/(2*eps),np.sum(force*d),1e-10,2e-4)
    dg=geo.action(q,d);dp=-c.alpha*np.einsum('kij,ij->k',g['gradient'],d)/c.capacity
    tangent=metric((fp-fm)/(2*eps),-c.alpha*(np.einsum('k,kij->ij',dp,g['gradient'])+np.einsum('k,kij->ij',p,dg)),1e-8,2e-4)
    q1=q+1e-4*d;gbar=geo.discrete(q,q1);defect=np.einsum('kij,ij->k',gbar,q1-q)-(geo.evaluate(q1)['volume']-g['volume'])
    high=CellGeometry(m,2,order=geo.order+1).evaluate(q)
    adjacent={k:metric(g[k],high[k],1e-10,2e-5) for k in ('volume','gradient','H')}
    if not derivative['passed'] or not tangent['passed'] or np.max(abs(defect))>1e-10 or not all(v['passed'] for v in adjacent.values()):raise ValueError('cell geometry derivatives/integration failed')
    write(run/'P4/coupling-kernel-check.json',dict(status='passed_scoped',derivative=derivative,tangent=tangent,local_volume_defect_m3=defect.tolist(),adjacent=adjacent,
        reference_volumes=geo.V0.tolist(),geometry=geo.identity))
    legacy=analytic_flux([e[-1]-e[0] for e in m.parent.edges])
    write(run/'P4/inherited-tensor-law-check.json',dict(reference_only=True,**legacy))
    rest=geo.evaluate(m.rest().q);H=rest['H'];expected=np.diag([30.,60.,30.])
    assert np.max(abs(H-expected))<1e-8
    # Independent linear pressure / constant mobility face flux, both exterior faces included.
    gradient=.02;pbar=.01+gradient*.5*(geo.cuts[:-1]+geo.cuts[1:]);z=np.full(3,-geo.k_mu*gradient*geo.area)
    boundary=np.zeros(3);boundary[0]=-(.01+gradient*geo.cuts[0]);boundary[-1]=.01+gradient*geo.cuts[-1]
    error=float(np.max(abs(H@z-(geo.B.T@pbar-boundary))))
    assert error<1e-10
    write(run/'P4/flux-operator-check.json',dict(status='passed_scoped',constant_H=H.tolist(),expected_H=expected.tolist(),
        linear_pressure_flux_error_Pa=error,scope='x-only zero transverse flow, half-cell resistances',full_tensor_inverse_component=True))
    ranks=[]
    for cells in (2,4):
        for drained in (False,True):
            cc=MultiCellAVF(m,cfg,cells=cells,drained=drained);gg=cc.geometry.evaluate(m.rest().q);H=gg['H'][np.ix_(cc.active,cc.active)];G=gg['gradient'][:,m.free].reshape(cells,-1)
            if la.eigvalsh(H)[0]<=0:raise ValueError('active RT0 dissipation not SPD')
            L=cc.B@la.solve(H,cc.B.T,assume_a='pos');vals,modes=la.eigh(L)
            for S in (.2,.0002):
                A=cc.rest_matrix(.01,storage=S);a=A/np.maximum(np.max(abs(A),axis=1),1e-30)[:,None];a/=np.maximum(np.max(abs(a),axis=0),1e-30)[None,:]
                sv=la.svdvals(a);rank=int(np.sum(sv>64*len(A)*np.finfo(float).eps*sv[0]))
                if rank!=len(A):raise ValueError('multicell mixed rank failure')
                ranks.append(dict(cells=cells,drained=drained,storage=S,rank=rank,size=len(A),scaled_condition=float(sv[0]/sv[-1]),
                    pressure_diffusion_eigenvalues=vals.tolist(),pressure_modes=modes.T.tolist(),volume_gradient_singular_values=la.svdvals(G).tolist()))
    write(run/'P4/mixed-rank-solver.json',dict(status='passed_scoped',records=ranks,iteration_matrix='general chord approximation, rest solid K; exact volume/mass/flux off-diagonal blocks at evaluated state',
        no_CG=True,no_pressure_penalty=True,no_general_inf_sup_claim=True,zero_pressure_rest_reference=True))
    zero=MultiCellAVF(m,cfg,alpha=0.,pressure0=0.);solid=ValidatedAVF(m,cfg);ext=.001*geo.evaluate(m.rest().q)['gradient'].sum(axis=0)
    rows=[]
    for _ in range(4):rows.append(zero.step(.01,external_force=ext));solid.step(.01,external_force=ext)
    cache=CachedProbes(m);a=cache.frame(zero.state);b=cache.frame(solid.state)
    solid_errors={k:metric(a[k]-a['X'] if k=='x' else a[k],b[k]-b['X'] if k=='x' else b[k],at,.05,regions(a['X'])['global_domain']) for k,at in [('x',5e-5),('velocity',1e-4),('PK1',.02)]}
    fixed=[]
    for cells in (2,4):
        flow=MultiCellAVF(m,cfg,cells=cells,drained=True,fixed_solid=True,reservoir=.002)
        H=flow.geometry.evaluate(m.rest().q)['H'][np.ix_(flow.active,flow.active)];C=np.diag(flow.capacity);L=flow.B@la.solve(H,flow.B.T,assume_a='pos')
        rhs=source(flow)+flow.B@la.solve(H,flow.gb,assume_a='pos');p=np.full(cells,.01);errors=[];rr=[]
        for _ in range(4):
            p=la.solve(C+.005*L,(C-.005*L)@p+.01*rhs,assume_a='gen');rr.append(flow.step(.01,source_m3_s=source(flow)));errors.append(float(np.max(abs(p-rr[-1]['pressure_Pa']))))
        if max(errors)>1e-8:raise ValueError('fixed solid analytic midpoint flow mismatch')
        fixed.append(dict(cells=cells,max_pressure_error_Pa=max(errors),rows=rr))
    if not all(v['passed'] for v in solid_errors.values()):raise ValueError('pure solid alpha0 limit differs')
    write(run/'P4/limits-check.json',dict(status='passed_scoped',pure_solid=solid_errors,solid_rows=rows,fixed_flow=fixed))
    print('MULTICELL_KERNEL_LIMITS passed',flush=True)


def advance(c,store,rows,cache,steps):
    start=c.state.step
    while c.state.step<4 and c.state.step-start<steps:
        base=c.state
        try:
            row=c.step(.01,source_m3_s=source(c));store.save(c.state,[*rows,row],frame=c.frame(cache))
        except Exception:
            loaded=store.load()
            if loaded['state'].digest()==c.state.digest() and c.state.step==base.step+1:row=loaded['rows'][-1]
            else:
                from engine.aniso_phase1.research_d.common_state import StateTransaction
                c._transaction=StateTransaction(base,validator=c.validate);c.geometry.cache.clear();raise
        rows.append(row)
    return rows


def prepare(run):
    run=Path(run);register(run,'N4/transaction-protocol.json',dict(steps=4,restart_after=2,owned='q/v/predictor, cell p/content/source history, oriented face flux, boundary cumulative volume',
        fault='mutate local pressure/history after child solve then raise; pre/post generation publication'))
    m,cfg=model_config(run);cache=CachedProbes(m)
    for name,drained in [('coupled-closed2',False),('coupled-drained2',True)]:
        c=MultiCellAVF(m,cfg,drained=drained,reservoir=.002 if drained else 0.);folder=run/'cases'/name
        identity=dict(schema='cost-phase-half-cell-coupled-case-v1',input_lock_sha256=sha(run/'input-lock.json'),coupling=c.identity,numerical_source_sha256=source_files(),
            fixture_source_sha256=sha(Path(__file__)),protocol_sha256=digest(dict(dt=.01,steps=4,source=source(c).tolist())))
        write(folder/'identity.json',identity);write(folder/'execution-protocol.json',dict(config=cfg,coupling=c.identity,dt=.01,steps=4,source=source(c).tolist()))
        snapshot(folder/'source',source_files());store=GenerationStore(folder,identity);store.save(c.state,[],frame=c.frame(cache))
        rows=advance(c,store,[],cache,2 if drained else 4);write(folder/'ledger.json',rows)
    c=MultiCellAVF(m,cfg,drained=True,reservoir=.002);before=c.state.digest()
    def fault(where,state):
        state.child_states['fluid']['pressure_Pa'][0]=999.;state.child_states['fluid']['cumulative_source_m3'][1]=100.;raise RuntimeError('controlled local fluid failure')
    try:c.step(.01,source_m3_s=source(c),inject=fault)
    except RuntimeError:pass
    else:raise AssertionError('expected precommit fault')
    assert c.state.digest()==before
    write(run/'P4/rollback-check.json',dict(status='passed_scoped',mutated_local_arrays_and_history_rolled_back=True,source_sha256=source_files()))
    print('MULTICELL_PREPARED closed4 drained2',flush=True)


def cycle(run):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('sealed release')
    folder=run/'cases/coupled-drained2';identity=read(folder/'identity.json')
    if identity['numerical_source_sha256']!=source_files() or identity['fixture_source_sha256']!=sha(Path(__file__)):raise ValueError('coupling source changed')
    m,cfg=model_config(run);store=GenerationStore(folder,identity);data=store.load();c=MultiCellAVF(m,cfg,drained=True,reservoir=.002,state=data['state'])
    if c.state.step!=2:raise ValueError('expected actual step2 restart')
    rows=advance(c,store,data['rows'],CachedProbes(m),2);write(folder/'ledger.json',rows);print('MULTICELL_RESUMED 4',flush=True)


def final_check(run):
    run=Path(run);m,cfg=model_config(run);cache=CachedProbes(m);records=[];resolution=[]
    for name,drained in [('coupled-closed2',False),('coupled-drained2',True)]:
        folder=run/'cases'/name;history=GenerationStore(folder,read(folder/'identity.json')).history();saved=history[-1]['state'];rows=history[-1]['rows']
        direct=MultiCellAVF(m,cfg,drained=drained,reservoir=.002 if drained else 0.)
        for _ in range(4):direct.step(.01,source_m3_s=source(direct))
        errors={k:float(np.max(abs(getattr(saved,k)-getattr(direct.state,k)))) for k in ('q','velocity','predictor')}
        for k in ('pressure_Pa','flux_interval_m3_s','content_m3','cumulative_source_m3','cumulative_boundary_m3'):
            errors[k]=float(np.max(abs(np.array(saved.child_states['fluid'][k])-np.array(direct.state.child_states['fluid'][k]))))
        if max(errors.values())>1e-9 or len(history)!=5:raise ValueError('multicell new process differs')
        summary=dict(status='passed_scoped',case=name,steps=4,errors=errors,min_detF=min(r['min_detF'] for r in rows),
            max_local_mass_defect_m3=max(np.max(np.abs(r['mass_defect_m3'])) for r in rows),max_energy_balance_J=max(abs(r['energy_balance_J']) for r in rows),
            min_darcy_J=min(r['darcy_dissipation_J'] for r in rows),max_pressure_work_defect_J=max(np.max(np.abs(r['pressure_work_defect_J'])) for r in rows))
        f=saved.child_states['fluid'];initial=history[0]['state'].child_states['fluid']
        closure=float(np.sum(np.array(f['content_m3'])-initial['content_m3'])+f['cumulative_boundary_m3']-np.sum(f['cumulative_source_m3']))
        if abs(closure)>1e-10:raise ValueError('cumulative local histories do not close global content')
        summary['cumulative_content_closure_m3']=closure
        write(folder/'summary.json',summary);records.append(summary)
        fine=MultiCellAVF(m,cfg,cells=4,drained=drained,reservoir=.002 if drained else 0.);fine_rows=[]
        for _ in range(4):fine_rows.append(fine.step(.01,source_m3_s=source(fine)))
        p4=np.array(fine.state.child_states['fluid']['pressure_Pa']);p2=np.array(direct.state.child_states['fluid']['pressure_Pa']);pressure=metric(p2,p4.reshape(2,2).mean(axis=1),5e-4,.05,direct.geometry.V0)
        a,b=direct.frame(cache),fine.frame(cache)
        regional={region:{k:metric(a[k]-a['X'] if k=='x' else a[k],b[k]-b['X'] if k=='x' else b[k],at,.05,w) for k,at in [('x',5e-5),('total_PK1',.02)]} for region,w in regions(a['X']).items()}
        fields=regional['global_domain']
        resolution.append(dict(drained=drained,pressure_mean=pressure,fields=fields,regional_fields=regional,four_cell_rows=fine_rows,pressure_spatial_accuracy=False))
    write(run/'P4/limits-and-closure.json',dict(status='passed_scoped',records=records,resolution=resolution,
        comparison_scope='same reference cell means/physical positions; bounded sensitivity check, no continuum pressure certificate'))
    write(run/'P4/coupling-decision.json',dict(status='passed_scoped',physical_3D_coupling=True,pressure_dofs=2,drained_flux_dofs=2,
        four_cell_check=True,actual_new_process_restart=True,source_sha256=source_files(),pressure_spatial_accuracy=False,pressure_monotonicity_scope='fixed solid zero source drainage; P4/monotonicity-and-time.json',production_C_E_integration=False,
        pure_solid_default=True,material='sufficient rule only',records=records,
        four_cell_sensitivity_passed=all(x['pressure_mean']['passed'] and all(v['passed'] for region in x['regional_fields'].values() for v in region.values()) for x in resolution)))
    render(run,run/'P4/visualization');print('MULTICELL_FINAL',records,flush=True)


def render(run,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);out=Path(output)
    if (run/'release.json').exists() and (out.resolve()==run.resolve() or run.resolve() in out.resolve().parents):raise ValueError('sealed release: external output required')
    out.mkdir(parents=True,exist_ok=True);fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for name in ['coupled-closed2','coupled-drained2']:
        folder=run/'cases'/name;hist=GenerationStore(folder,read(folder/'identity.json')).history();rows=hist[-1]['rows'];t=[r['time'] for r in rows];p=np.array([r['pressure_Pa'] for r in rows])
        for k in range(2):axes[0,0].plot(t,p[:,k],'.-',label=name+f' cell{k}')
        axes[0,1].plot(t,[r['flux_interval_m3_s'][0] for r in rows],'.-',label=name+' internal face')
        axes[1,0].plot(t,[r['energy_balance_J'] for r in rows],'.-',label=name)
        with np.load(hist[-1]['folder']/'frame.npz') as z:
            X=z['X'];mid=X.shape[2]//2;ids=X[:,:,mid,0].ravel();stress=z['total_PK1'][:,:,mid,0,0].ravel()
            axes[1,1].scatter(ids,stress,s=8,label=name)
    for ax,y in zip(axes.ravel(),['cell mean pressure (Pa)','oriented face flow (m^3/s)','raw energy balance (J)','total PK1 P11 (Pa)']):ax.set(ylabel=y,xlabel='reference x (m)' if ax is axes[1,1] else 'time (s)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('3D selected solid + two pressure cells / half-cell face flow; four steps; raw fields')
    fig.savefig(out/'coupled-summary.png',dpi=140);plt.close(fig)
    (out/'index.html').write_text('<meta charset="utf-8"><h1>两单元压力与总应力</h1><p>原始单元均压、统一朝向通量和总 PK1；四步验证，无连续压力精度声明。</p><img style="max-width:100%" src="coupled-summary.png">')
    print(out/'index.html',flush=True)


def analytic_transient(run):
    run=Path(run);verify(run)
    register(run,'P4/transient-protocol.json',dict(cells=[2,4,8],schemes=['consistent_RT0','half_cell'],times_s=[0.,.01,.02,.03,.04],
        fixed_solid=True,source=0.,initial_pressure=.01,reservoir=.002,storage=.2,dt=.01,overshoot_atol_Pa=1e-5,
        scope='bounded x-only pressure fixture; no source and fixed solid; not general coupled monotonicity'))
    records=[]
    for scheme in ['consistent_RT0','half_cell']:
        for cells in (2,4,8):
            length=.75/cells;area=.0625;mobility=.1;H=np.zeros((cells+1,cells+1));B=np.zeros((cells,cells+1))
            for k in range(cells):
                local=length/(area*mobility)*(np.array([[1/3,1/6],[1/6,1/3]]) if scheme=='consistent_RT0' else .5*np.eye(2))
                H[k:k+2,k:k+2]+=local;B[k,k]=-1;B[k,k+1]=1
            H=H[1:,1:];B=B[:,1:];C=.2*length*area*np.eye(cells);gb=np.zeros(cells);gb[-1]=.002
            L=B@la.solve(H,B.T,assume_a='pos');b=B@la.solve(H,gb,assume_a='pos');peq=la.solve(L,b);p0=np.full(cells,.01)
            generator=-la.solve(C,L);off=generator.copy();np.fill_diagonal(off,0.)
            ts=np.linspace(0,.04,41);exact=np.array([peq+la.expm(generator*t)@(p0-peq) for t in ts])
            initial_flux=la.solve(H,B.T@p0-gb,assume_a='pos');p=p0.copy();rows=[]
            for _ in range(4):
                p1=la.solve(C+.005*L,(C-.005*L)@p+.01*b);z=la.solve(H,B.T@(.5*(p+p1))-gb,assume_a='pos')
                mass=C@(p1-p)+.01*B@z;D=.01*float(z@H@z)
                rows.append(dict(pressure=p1.tolist(),flux=z.tolist(),mass_defect=float(np.max(abs(mass))),dissipation_J=D));p=p1
            G=la.solve(C+.005*L,C-.005*L)
            rec=dict(scheme=scheme,cells=cells,H=H.tolist(),B=B.tolist(),C=C.tolist(),initial_flux=initial_flux.tolist(),
                initial_dp_dt=la.solve(C,b-L@p0).tolist(),exact_at_001=exact[10].tolist(),exact_max_overshoot_Pa=float(max(0.,exact.max()-.01)),
                exact_min_pressure=float(exact.min()),generator_min_offdiag=float(off.min()),midpoint_min_G=float(G.min()),
                conservative_midpoint_bound_s=float(2*np.min(np.diag(C)/np.diag(L))),rows=rows)
            records.append(rec)
    old=records[0]
    assert abs(old['initial_flux'][0]+.00011428571428571432)<1e-12
    assert abs(old['exact_at_001'][0]-.010186392017378799)<1e-10
    candidate=[x for x in records if x['scheme']=='half_cell' and x['cells'] in (2,4)]
    for x in candidate:
        assert x['exact_max_overshoot_Pa']<=1e-5 and x['generator_min_offdiag']>=-1e-12 and x['midpoint_min_G']>=-1e-12
        assert all(max(row['pressure'])<=.01001 and min(row['pressure'])>=.00199 and min(row['flux'])>=-1e-12 and row['mass_defect']<1e-12 and row['dissipation_J']>=0 for row in x['rows'])
    write(run/'P4/transient-baseline.json',dict(status='reproduced',record=old,source_sha256=sha(Path('docs/results/spatial-phase/20261001T043604Z-spatial-phase/N4/flux-transient-diagnostic.json'))))
    write(run/'P4/monotonicity-and-time.json',dict(status='passed_scoped',records=records,selected='half_cell',production_cells=[2,4],dt_s=.01,
        spatial_monotonicity=True,time_monotonicity='normal storage .2 and this grid/dt; not unconditional',small_storage='rank check only, positivity step limit must be re-evaluated',
        general_3D_pressure_accuracy=False,source_and_solid_compression_excluded_from_monotonicity_claim=True))
    (run/'P4/discretization-derivation.md').write_text('''# Half-cell conservative flow (limited x-only fixture)

The new pressure discretization differs from consistent RT0. With transverse flow constrained to zero,
A=(k/mu) J F^-1 F^-T and the effective scalar mobility is 1/(A^-1)xx, not Axx.
Integrate (A^-1)xx over each reference half-cell and divide by the squared cross-sectional area.
Each face resistance is the sum of its two adjacent half resistances; boundary faces use one half.
Reference cuts and cell midpoints are inserted into the actual quadrature partition, so masks never split an unsplit Gauss cell.

The face law is H z = B^T p - g_boundary. Opposite incidence signs preserve local conservation.
H is positive diagonal, hence z^T H z is nonnegative. Solid volume gradients and AVF pressure work remain unchanged.
At fixed solid with positive diagonal C, L=B H^-1 B^T has nonpositive off-diagonals and the homogeneous generator -C^-1 L is Metzler.
The midpoint method additionally requires a step restriction; h <= 2 min(Cii/Lii) is a conservative sufficient bound for the tested structure.
No pressure clipping, flux filtering, boundary smoothing, or ad hoc modification of the original RT0 certificate is performed.

This validates normal-storage two/four x cells and zero transverse flux. It is not a general 3D anisotropic-grid method or a continuum pressure accuracy certificate.
''')
    print('PRESSURE_ANALYTIC',[(x['scheme'],x['cells'],x['exact_max_overshoot_Pa']) for x in records],flush=True)


def publication_faults(run):
    run=Path(run);m,cfg=model_config(run);cache=CachedProbes(m);reports=[]
    for stage in ('before_pointer','after_pointer'):
        c=MultiCellAVF(m,cfg,drained=True,reservoir=.002);folder=run/'P4/publication-faults'/stage
        identity=dict(coupling=c.identity,source=source_files(),stage=stage);store=GenerationStore(folder,identity);store.save(c.state,[],frame=c.frame(cache));before=c.state.digest();pointer=read(store.pointer)
        save=store.save
        def fault(where):
            if where==stage:raise OSError('controlled fluid publication '+stage)
        def wrapped(*args,**kwargs):return save(*args,**kwargs,inject=fault)
        store.save=wrapped
        try:rows=advance(c,store,[],cache,1)
        except OSError:
            if stage!='before_pointer':raise
            assert c.state.digest()==before and read(store.pointer)==pointer and len(store.history())==1
        else:
            assert stage=='after_pointer' and c.state.step==1 and len(store.history())==2
            assert store.load()['state'].digest()==c.state.digest()
        reports.append(dict(stage=stage,passed=True,accepted_steps=c.state.step))
    write(run/'P4/publication-check.json',dict(status='passed_scoped',records=reports,actual_pressure_and_history=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['analytic','checks','prepare','cycle','final','publication','render']);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.phase=='render':render(a.run,a.output or a.run/'P4/visualization')
    else:
        with serial_lock(a.run):
            if (a.run/'release.json').exists():raise ValueError('sealed release')
            {'analytic':analytic_transient,'checks':checks,'prepare':prepare,'cycle':cycle,'final':final_check,'publication':publication_faults}[a.phase](a.run)
