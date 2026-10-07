"""Actual original144 / constant-pressure-cell 3D exchange and restart checks."""
from pathlib import Path
import argparse,copy
import numpy as np
import scipy.linalg as la
from .provenance import PARENT,read,write,sha,digest,register,source_files,snapshot,serial_lock,resources
from . import config
from benchmarks.research_sequential_next.model_package import load_reduction
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
from engine.aniso_phase1.research_reference_next.coupling import CoupledAVF,VolumeGeometry
from engine.aniso_phase1.research_post_release.fields import CachedProbes


def model_config(run):
    import warp as wp
    wp.config.kernel_cache_dir=str(Path(run).resolve()/'warp-cache')
    lock=read(Path(run)/'input-lock.json');cfg=config.make(lock['energy_scale_J'],dt=.01,end=.04)
    return SegmentedModel(load_reduction(PARENT),order=7,device='cuda:0',hold=0.),cfg


def checks(run):
    run=Path(run)
    register(run,'Q5/physical-model.json',dict(solid='actual original144, full M5, original Ks; fixed zero grip lifts',
        material_order=7,pressure='one constant cell on entire 3D reference domain',flux='one right-face volume rate; closed or drained zero-pressure reservoir',
        fluid_content='integral reference-volume fluid content increment, units m^3; not mass without density',
        alpha=.8,storage_Pa_inverse=.2,k_m2=1e-4,mu_Pa_s=1e-3,parameters='numerical validation fixture, not calibrated material',
        conductance='2*A/L times volume-averaged reference mobility J F^-1 (k/mu) F^-T xx',
        time='same solid AVF + midpoint pressure and Darcy flux; two-point exact volume discrete gradient',
        scope='minimal real 3D exchange; no resolved pressure field or production C-E integration claim',
        steps_per_case=4,dt=.01,energy_rtol=.01,energy_atol_J=1e-9,mass_atol_m3=1e-12))
    m,cfg=model_config(run);c=CoupledAVF(m,cfg);geo=c.geometry
    rng=np.random.default_rng(20261001);q=np.zeros_like(m.rest().q);q[m.free]=rng.normal(size=(len(m.free),3))*1e-5
    d=rng.normal(size=q.shape);d[m.fixed]=0;d/=np.linalg.norm(d);eps=1e-6
    g=geo.evaluate(q);p=.02;content=c.alpha*(g['volume']-geo.V0)+c.capacity*p
    def fluid(v):
        x=geo.evaluate(v);pv=(content-c.alpha*(x['volume']-geo.V0))/c.capacity
        return .5*c.capacity*pv*pv,-c.alpha*pv*x['gradient']
    up,fp=fluid(q+eps*d);um,fm=fluid(q-eps*d);force=fluid(q)[1]
    derivative=metric((up-um)/(2*eps),float(np.sum(force*d)),1e-10,2e-4)
    dg=geo.action(q,d);dp=-c.alpha*float(np.sum(g['gradient']*d))/c.capacity
    action=-c.alpha*(dp*g['gradient']+p*dg);tangent=metric((fp-fm)/(2*eps),action,1e-8,2e-4)
    q1=q+1e-4*d;gbar=geo.discrete(q,q1);volume_error=float(np.sum(gbar*(q1-q))-(geo.evaluate(q1)['volume']-g['volume']))
    higher=SegmentedModel(m.reduction,order=8,device='cuda:0',hold=0.);high=VolumeGeometry(higher).evaluate(q)
    adjacent=dict(volume=metric(g['volume'],high['volume'],1e-12,2e-5),gradient=metric(g['gradient'],high['gradient'],1e-10,2e-5),conductance=metric(g['conductance'],high['conductance'],1e-12,2e-5))
    if not derivative['passed'] or not tangent['passed'] or abs(volume_error)>1e-10 or not all(x['passed'] for x in adjacent.values()):raise ValueError('coupling derivative/integration check failed')
    write(run/'Q5/coupling-kernel-check.json',dict(status='passed_scoped',potential_direction=derivative,tangent_direction=tangent,volume_discrete_gradient_defect_m3=volume_error,adjacent_q7_q8=adjacent))
    ranks=[]
    for drained in (False,True):
        cc=CoupledAVF(m,cfg,drained=drained)
        for S in (.2,.0002):
            A=cc.rest_matrix(.01,storage=S);row=1/np.maximum(np.max(abs(A),axis=1),1e-30);B=row[:,None]*A;col=1/np.maximum(np.max(abs(B),axis=0),1e-30);sv=la.svdvals(B*col[None,:]);cut=64*len(A)*np.finfo(float).eps*sv[0];rank=int(np.sum(sv>cut))
            if rank!=len(A):raise ValueError('mixed rest system rank failure')
            ranks.append(dict(drained=drained,storage=S,size=len(A),rank=rank,scaled_min_singular=float(sv[-1]),scaled_condition=float(sv[0]/sv[-1])))
    write(run/'Q5/mixed-rank-solver.json',dict(status='passed_scoped',records=ranks,actual_rest_Jacobian=True,pressure_dofs=1,flux_dofs=1,CG=False,nonlinear_solve='general chord matrix, true residual and line search; no SPD assumption',no_pressure_penalty=True))
    # Nonzero-load pure solid limit against the existing AVF on the same space.
    zero=CoupledAVF(m,cfg,alpha=0.,pressure0=0.);solid=ValidatedAVF(m,cfg);ext=.001*geo.evaluate(m.rest().q)['gradient'];records=[]
    for _ in range(4):records.append(zero.step(.01,external_force=ext));solid.step(.01,external_force=ext)
    cache=CachedProbes(m);a=cache.frame(zero.state);b=cache.frame(solid.state)
    solid_errors=dict(displacement=metric(a['x'],b['x'],5e-5,.05),velocity=metric(a['velocity'],b['velocity'],1e-4,.05),stress=metric(a['PK1'],b['PK1'],.02,.05))
    if not all(x['passed'] for x in solid_errors.values()):raise ValueError('zero pressure solid limit differs')
    # Fixed solid midpoint drainage has an independent scalar closed form.
    fixed=CoupledAVF(m,cfg,drained=True,fixed_solid=True,pressure0=.01);p=.01;T=fixed.geometry.evaluate(m.rest().q)['conductance'];expected=[];rows=[]
    for _ in range(4):
        p=p*(fixed.capacity-.005*T)/(fixed.capacity+.005*T);expected.append(p);rows.append(fixed.step(.01))
    error=max(abs(a['pressure_Pa']-b) for a,b in zip(rows,expected))
    if error>1e-8:raise ValueError('fixed-solid Darcy limit differs')
    write(run/'Q5/limits-check.json',dict(status='passed_scoped',solid=solid_errors,solid_rows=records,fixed_flow_pressure_max_Pa=error,fixed_flow_rows=rows))
    print('COUPLING_KERNEL_LIMITS passed',derivative,tangent,volume_error,flush=True)


def prepare(run):
    run=Path(run);m,cfg=model_config(run)
    register(run,'Q5/transaction-protocol.json',dict(steps=4,dt=.01,source='0.001 times reference volume per second',fault='before outer commit, corrupt fluid then raise',restart='real CLI process at step2',case_scope='one pressure cell, actual 3D solid'))
    for name,drained in [('coupled-closed',False),('coupled-drained',True)]:
        c=CoupledAVF(m,cfg,drained=drained);folder=run/'cases'/name
        identity=dict(schema='reference-next-coupled-case-v1',input_lock_sha256=sha(run/'input-lock.json'),coupling=c.identity,protocol_sha256=digest(dict(dt=.01,steps=4,source_factor=.001)),numerical_source_sha256=source_files())
        write(folder/'identity.json',identity);snapshot(folder/'source',identity['numerical_source_sha256']);write(folder/'execution-protocol.json',dict(config=cfg,coupling=c.identity,dt=.01,steps=4,source_factor=.001))
        store=GenerationStore(folder,identity);store.save(c.state,[])
    c=CoupledAVF(m,cfg,drained=True);before=c.state.digest()
    def fault(where,state):state.child_states['fluid']['pressure_Pa']=999.;raise RuntimeError('controlled fluid precommit failure')
    try:c.step(.01,source_m3_s=c.geometry.V0*.001,inject=fault)
    except RuntimeError:pass
    else:raise AssertionError('expected failure')
    assert c.state.digest()==before
    write(run/'Q5/rollback-check.json',dict(status='passed_scoped',solid_pressure_flux_history_unchanged=True,controlled_fault=True))


def cycle(run,case,stop_after=None):
    run=Path(run)
    if (run/'release.json').exists():raise ValueError('sealed run')
    folder=run/'cases'/case;identity=read(folder/'identity.json')
    if identity['numerical_source_sha256']!=source_files():raise ValueError('coupling numerical source changed')
    m,cfg=model_config(run);store=GenerationStore(folder,identity);loaded=store.load();drained=identity['coupling']['drained']
    c=CoupledAVF(m,cfg,drained=drained,state=loaded['state']);rows=loaded['rows'];cache=CachedProbes(m);start=c.state.step
    while c.state.step<4 and (stop_after is None or c.state.step-start<stop_after):
        base=c.state;oldpointer=read(store.pointer)
        try:
            row=c.step(.01,source_m3_s=c.geometry.V0*.001);frame=cache.frame(c.state);frame['pressure_Pa']=np.array(c.state.child_states['fluid']['pressure_Pa'])
            store.save(c.state,[*rows,row],frame=frame)
        except Exception:
            committed=store.load()
            if committed['state'].digest()==c.state.digest() and c.state.step==base.step+1:row=committed['rows'][-1]
            else:c=CoupledAVF(m,cfg,drained=drained,state=base);raise
        rows.append(row)
    assert store.load()['state'].digest()==c.state.digest()
    summary=dict(status='passed_scoped' if c.state.step==4 else 'in_progress',steps=c.state.step,end_s=c.state.time,min_detF=min(x['min_detF'] for x in rows),
        max_mass_defect_m3=max(abs(x['mass_defect_m3']) for x in rows),max_energy_balance_J=max(abs(x['energy_balance_J']) for x in rows),
        max_pressure_work_defect_J=max(abs(x['pressure_work_defect_J']) for x in rows),min_darcy_dissipation_J=min(x['darcy_dissipation_J'] for x in rows),
        final_pressure_Pa=c.state.child_states['fluid']['pressure_Pa'],config=identity['coupling'],resources=resources())
    write(folder/'summary.json',summary);write(folder/'ledger.json',rows);print(case,summary,flush=True)


def final_check(run):
    run=Path(run);m,cfg=model_config(run);records=[]
    for name,drained in [('coupled-closed',False),('coupled-drained',True)]:
        folder=run/'cases'/name;identity=read(folder/'identity.json');history=GenerationStore(folder,identity).history();state=history[-1]['state'];c=CoupledAVF(m,cfg,drained=drained)
        for _ in range(4):c.step(.01,source_m3_s=c.geometry.V0*.001)
        errors={k:float(np.max(abs(getattr(state,k)-getattr(c.state,k)))) for k in ('q','velocity','predictor')}
        errors['pressure']=abs(state.child_states['fluid']['pressure_Pa']-c.state.child_states['fluid']['pressure_Pa'])
        if max(errors.values())>1e-9 or len(history)!=5:raise ValueError('coupled restart differs')
        records.append(dict(case=name,errors=errors,summary=read(folder/'summary.json')))
    write(run/'Q5/coupling-decision.json',dict(status='passed_scoped',actual_new_process_restart=True,records=records,physical_3D_coupling=True,
        uses_actual_original144_and_M5=True,pressure_dofs=1,flux_dofs=1,pressure_spatial_accuracy=False,production_C_E_integration=False,
        scope='independent minimal 3D single-cell fluid / actual solid coupling; keep pure solid daily entry'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for name in ['coupled-closed','coupled-drained']:
        rows=read(run/'cases'/name/'ledger.json');t=[x['time'] for x in rows]
        for ax,key,label in zip(axs,['pressure_Pa','content_m3','energy_balance_J'],['pressure (Pa)','fluid content (m^3)','raw energy balance (J)']):ax.plot(t,[x[key] for x in rows],'.-',label=name);ax.set(xlabel='time (s)',ylabel=label);ax.grid(alpha=.2)
    axs[0].legend();fig.suptitle('3D original144 + one constant pressure cell; scoped validation')
    fig.savefig(run/'Q5/coupled-summary.png',dpi=130);plt.close(fig)
    print('COUPLED_FINAL',[(x['case'],x['errors']) for x in records],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['checks','prepare','cycle','final']);p.add_argument('--run',type=Path,required=True);p.add_argument('--case');p.add_argument('--stop-after',type=int);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release; fork before running studies')
        if a.phase=='cycle':cycle(a.run,a.case,a.stop_after)
        else:{'checks':checks,'prepare':prepare,'final':final_check}[a.phase](a.run)
