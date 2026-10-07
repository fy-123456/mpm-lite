"""Two bounded reference levels with explicit coverage and uncertainty."""
from pathlib import Path
import argparse,time,gc,traceback,resource,threading,os
import numpy as np
from .provenance import APP,ROOT,PARENT,read,write,sha,digest,register,verify,serial_lock,resources
from benchmarks.research_post_release.reference_study import base_reference,reopen as reopen_old
from engine.aniso_phase1.research_post_release.ambient_reference import ambient_space,embedding_audit,operators
from engine.aniso_phase1.research_reference_next.reference import interior_h,nested_p_coefficients
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel
from engine.aniso_phase1.research_sequential_next.model import PracticalModel
from engine.aniso_phase1.research_sequential.condensation import Condensation
from benchmarks.research_sequential_next.spatial import static_solve,compare_fields
from benchmarks.research_sequential_next.compare import metric
from benchmarks.research_sequential_next.run import probe_frame
from engine.aniso_phase1.research_d.common_state import CommonState


def reopen(folder):
    folder=Path(folder);p=read(folder/'space-package.json');r,_,_,_=base_reference();s,_=ambient_space(r.parent,2,mode='p')
    if p['level']=='R3':s,_=interior_h(s)
    if sha(folder/'data.npz')!=p['data_sha256']:raise ValueError('reference data changed')
    with np.load(folder/'data.npz',allow_pickle=False) as z:r=Condensation(s,z['M'],z['K']);q=z['q'].copy();expected=z['PK1'].copy()
    if r.signature!=p['reduction_sha256']:raise ValueError('reconstructed reference identity changed')
    model=PracticalModel(r,order=p['material_order'],device='cpu',hold=.005)
    state=CommonState(q,np.zeros_like(q),time=.5,child_states={'identity':model.identity})
    err=float(np.max(abs(probe_frame(model,state,[33,7,7])['PK1']-expected)))
    if err>1e-7:raise ValueError('reference reloaded field differs')
    return r,model,state,err


def run_level(run,level):
    run=Path(run);verify(run)
    import warp as wp
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache')
    if not (run/'Q1/coverage-registration.json').exists():
        register(run,'Q1/coverage-registration.json',dict(main=dict(angle=45,hold_m=.005),heldout_hold_m=.00375,
            R0='mathematical parent level2 enriched Q4',R1=str(APP/'S2/p1/space-package.json'),
            R2='nested degree6 interior family',R3='R2 + half-cell interior x bubbles and cubic transverse y/z',
            h_centres=[.375,.625],max_new_levels=2,soft_seconds=600,hard_seconds=1200,
            contraction=.7,practical_rtol=.05,PK1_atol_Pa=.02,reference_is_empirical=True))
    register(run,f'Q1/{level}/protocol.json',dict(level=level,scope='same material, original Ks, actual internal coverage',max_rss_GiB=16))
    begin=time.perf_counter();folder=run/'Q1'/level
    def guard():
        while True:
            rows=Path('/proc/self/status').read_text().splitlines()
            high=int(next(x for x in rows if x.startswith('VmHWM:')).split()[1])
            if high>16*2**20:
                write(folder/'resource-stop.json',dict(reason='16 GiB process peak limit',peak_rss_GiB=high/2**20))
                os._exit(75)
            time.sleep(.2)
    threading.Thread(target=guard,daemon=True).start()
    try:
        if level=='R2':
            base,_,_,_=base_reference();previous,pm,ps,_=reopen_old(APP/'S2/p1')
            space,definition=ambient_space(base.parent,2,mode='p')
            initial=nested_p_coefficients(previous.expand(ps.q),read(APP/'S2/p1/definition.json'),definition,base.parent.ndof,space.ndof)
            # The actual represented R1 field, not raw coefficients, must embed.
            axes=[np.linspace(e[0]+1e-8,e[-1]-1e-8,n) for e,n in zip(space.edges,[29,5,5])]
            a,ga=previous.parent._sample(previous.parent.nodes(previous.expand(ps.q)),axes);b,gb=space._sample(space.nodes(initial),axes)
            nesting=dict(displacement_max=float(np.max(abs(a-b))),gradient_max=float(np.max(abs(ga-gb))))
            if max(nesting.values())>1e-8:raise ValueError('p5 represented field not nested in p6')
            parent=base
        else:
            parent,pm,ps,_=reopen(run/'Q1/R2');space,definition=interior_h(parent.parent)
            initial=np.vstack((parent.expand(ps.q),np.zeros((space.ndof-parent.parent.ndof,3))));nesting=embedding_audit(parent.parent,space)
        write(folder/'definition.json',dict(**definition,nesting=nesting))
        n=space.ndof;estimate=8*n*n*10+int(np.prod(space.shape))*3*8*16+np.prod([len(e)-1 for e in space.edges])*7**3*600
        write(folder/'resources.json',dict(predicted_core_bytes=int(estimate),**resources()))
        if estimate>12*2**30:raise MemoryError('reference estimated core exceeds 12 GiB reserved budget')
        embedding=embedding_audit(parent.parent,space)
        rr,op=operators(parent,space,max_seconds=max(1,600-(time.perf_counter()-begin)),progress=lambda *x:print(level,*x,flush=True))
        model=SegmentedModel(rr,order=7,device='cuda:0',hold=.005)
        state,solve=static_solve(model,initial,max_seconds=max(1,600-(time.perf_counter()-begin)))
        comparison,_,frame=compare_fields(pm,ps,model,state)
        a=model.evaluate(state.q);higher=SegmentedModel(rr,order=8,device='cuda:0',hold=.005);b=higher.evaluate(state.q)
        rng=np.random.default_rng(20261001);d=rng.normal(size=state.q.shape);d[rr.fixed]=0;d/=np.linalg.norm(d)
        checks=dict(U=metric(a['material_U'],b['material_U'],1e-10,.02),force=metric(a['material_force'],b['material_force'],1e-8,.02),
            tangent=metric(model.evaluate(state.q,d)['tangent_action'],higher.evaluate(state.q,d)['tangent_action'],1e-8,.03))
        if not all(x['passed'] for x in checks.values()):raise ValueError('reference sufficient material rule not qualified')
        oldforce=pm.evaluate(ps.q)['force'];reaction=metric(float(np.sum(a['force']*model.boundary.unit)),float(np.sum(oldforce*pm.boundary.unit)),1e-4,.05)
        np.savez_compressed(folder/'data.npz',M=rr.original_mass,K=rr.original_stiffness,q=state.q,full=rr.expand(state.q),X=frame['X'],x=frame['x'],PK1=frame['PK1'])
        write(folder/'space-package.json',dict(schema='reference-next-reference-v1',level=level,application_parent_sha256=read(run/'input-lock.json')['application_release_sha256'],
            data_sha256=sha(folder/'data.npz'),space_sha256=space.signature,reduction_sha256=rr.signature,material_order=7,mass_order=op['mass_order'],original_Ks=True))
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
        if peak>16:raise MemoryError('reference actual RSS exceeded 16 GiB')
        result=dict(status='passed_scoped',definition=definition,nesting=nesting,embedding=embedding,operators=op,solve=solve,
            adjacent=comparison,reaction=reaction,material=checks,seconds=time.perf_counter()-begin,peak_rss_GiB=peak)
        write(folder/'result.json',result);print(level,'DONE',result['seconds'],comparison['interior']['PK1'],flush=True)
    except (ValueError,MemoryError,TimeoutError) as exc:
        write(folder/'failure.json',dict(type=type(exc).__name__,reason=str(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-begin));raise


def decision(run):
    from .field_audit import invariants,compare_nodal
    run=Path(run);reports=[];nodal={};models={}
    register(run,'Q1/field-audit-protocol.json',dict(reason='uniform display probes may miss half-cell interior h supports',
        common_cells=True,quadrature_order=7,slab_cells=2,scope='same nonlinear PK1; use integration differences for reference decision'))
    for name in ('R0','R1','R2','R3'):
        if name=='R0':r,m,s,_=base_reference()
        elif name=='R1':r,m,s,_=reopen_old(APP/'S2/p1')
        else:
            if not (run/'Q1'/name/'space-package.json').exists():continue
            r,m,s,error=reopen(run/'Q1'/name)
            reports.append(dict(level=name,reloaded=True,field_max=error,reduction=r.signature,invariants=invariants(r.parent)))
        nodal[name]=(tuple(e.copy() for e in r.parent.edges),r.parent.p,r.parent.nodes(r.expand(s.q)))
        A=r.parent.A.copy();params=r.parent.params
        del r,m,s;gc.collect()
    write(run/'Q1/reload-check.json',dict(actual_new_process=True,records=reports))
    pairs={}
    for left,right in [('R0','R1'),('R1','R2'),('R2','R3')]:
        if left in nodal and right in nodal:
            pairs[left+'-'+right]=compare_nodal(nodal[left],nodal[right],A,params)
            print('COMMON_CELL',left,right,pairs[left+'-'+right]['interior'],flush=True)
    write(run/'Q1/common-cell-comparisons.json',dict(order=7,pairs=pairs,no_smoothing=True))
    available=all((run/'Q1'/n/'result.json').exists() for n in ('R2','R3'));regional={};reliable=available
    if available:
        first=pairs['R0-R1'];second=pairs['R1-R2'];third=pairs['R2-R3']
        for region in first:
            regional[region]={}
            for key in ('PK1','fiber_PK1'):
                d0=first[region][key]['absolute'];d1=second[region][key];d2=third[region][key]
                contraction=d1['absolute']<=max(.7*d0,1e-8)
                okay=contraction and d1['passed'] and d2['passed']
                regional[region][key]=dict(d10=d0,d21=d1['absolute'],d32=d2['absolute'],contracted=contraction,
                    empirical_uncertainty=max(d1['absolute'],d2['absolute'],1e-8),passed=okay)
                reliable&=okay
        reliable&=read(run/'Q1/R2/result.json')['reaction']['passed'] and read(run/'Q1/R3/result.json')['reaction']['passed']
    write(run/'Q1/reference-decision.json',dict(status='reference_usable_scoped' if reliable else 'reference_limited',
        reliable_for_candidates=bool(reliable),regional=regional,available=available,selected_reference='R3' if available else None,
        reloaded_in_new_process=True,scope='registered F45 static .005m and listed coverage; empirical, no continuum certificate'))
    print('REFERENCE_DECISION',reliable,regional,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['R2','R3','decision']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if (a.run/'release.json').exists():raise ValueError('sealed release; fork before running studies')
        decision(a.run) if a.phase=='decision' else run_level(a.run,a.phase)
