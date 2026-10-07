"""One registered F60 variation; authenticated external R5 and unchanged formal basis."""
from pathlib import Path
import argparse,copy,time,gc,resource,threading,os
import numpy as np
from dataclasses import replace
from .provenance import APP,LOCAL,ROOT,read,write,sha,digest,register,verify,serial_lock
from benchmarks.research_phase_reference_next.spatial_study import reload_level as external_reload, qcheck
from .spaces import load_selected
from benchmarks.research_sequential_next.spatial import static_solve
from benchmarks.research_reference_next.field_audit import compare_nodal
from benchmarks.research_sequential_next.compare import metric,regions
from benchmarks.research_sequential_next.run import probe_frame
from engine.aniso_phase1.research_sequential.condensation import Condensation
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel

def bounded_static(model,initial,diagnostics,max_seconds):
    import scipy.linalg as la
    from scipy.sparse.linalg import LinearOperator,gmres
    from engine.aniso_phase1.research_d.common_state import CommonState
    started=time.perf_counter();w=model.reduction.project(initial);w[model.fixed]=model.boundary.lift(.5)[model.fixed]
    factor=la.lu_factor(model.reduction.K[np.ix_(model.ids,model.ids)]);records=[];fallback=False
    def action(v):
        d=np.zeros_like(w);d[model.free]=v.reshape(-1,3)
        return model.evaluate(w,d)['tangent_action'][model.free].ravel()
    for it in range(12):
        out=model.evaluate(w);force=out['force'][model.free].ravel();norm=float(la.norm(force));record=dict(iteration=it,true_residual_N=norm,energy_J=out['U'],min_detF=out['min_detF']);records.append(record)
        write(diagnostics,dict(status='solving',records=records,exact_tangent_search=fallback))
        if norm<=1e-7:break
        if time.perf_counter()-started>max_seconds:raise TimeoutError('bounded direction solve timed out')
        for strategy in (['gmres'] if fallback else ['old_search','gmres']):
            if strategy=='old_search':update=la.lu_solve(factor,-force)
            else:
                fallback=True;operator=LinearOperator((len(force),len(force)),matvec=action,dtype=float)
                preconditioner=LinearOperator(operator.shape,matvec=lambda v:la.lu_solve(factor,v),dtype=float)
                update,info=gmres(operator,-force,M=preconditioner,rtol=1e-5,atol=1e-10,restart=30,maxiter=4)
                record['gmres_info']=int(info)
                if info!=0:raise ValueError('bounded exact-tangent search did not converge')
            record[strategy+'_residual_squared_directional_derivative']=float(2*force@action(update))
            accepted=False
            for ls in range(12):
                trial=w.copy();trial[model.free]+=2.**(-ls)*update.reshape(-1,3);new=model.evaluate(trial)
                if new['min_detF']>.1 and la.norm(new['force'][model.free])<norm:
                    w=trial;record['accepted_search']=strategy;record['line_search_halvings']=ls;accepted=True;break
            if accepted:break
        if not accepted:raise ValueError('both bounded search directions failed')
    else:raise ValueError('bounded F60 nonlinear solve did not converge')
    state=CommonState(w,np.zeros_like(w),time=.5,child_states={'identity':model.identity});model.validate(state,material=True)
    result=dict(accepted=True,iterations=records,reaction_N=float(np.sum(out['force']*model.boundary.unit)),energy_J=out['U'],nonlinear_residual_N=norm,
        seconds=time.perf_counter()-started,exact_tangent_search=fallback,physics_and_acceptance_unchanged=True)
    write(diagnostics,dict(status='passed_scoped',**result));return state,result


def prepare(run):
    run=Path(run);verify(run);package=read(APP/'S2/R5/space-package.json')
    if sha(APP/'S2/R5/data.npz')!=package['data_sha256'] or sha(APP/'S2/R5/definition.json')!=package['definition_sha256']:raise ValueError('reference package changed')
    write(run/'S2/reference-input-audit.json',dict(status='passed_scoped',external_R5=str(APP/'S2/R5'),package_sha256=sha(APP/'S2/R5/space-package.json'),
        inherited_main_comparison=read(APP/'S2/current-vs-R5.json'),inherited_adjacent=read(APP/'S2/reference-displacement-reaction.json'),new_main_solves=0))
    register(run,'S2/direction-protocol.json',dict(k_f=200.,mu=10.,lam=20.,fiber_angle_degrees=60.,fiber_direction=[.5,float(np.sqrt(3)/2),0.],peak_m=.005,
        cases=['formal','R4','R5'],case_count=3,formal_basis_fixed=True,retraining=False,original_mass=True,original_Ks=True,hidden=False,
        search_matrix='old material rest stiffness ONLY as quasi-Newton search metric; actual F60 residual required',soft_seconds=600,hard_seconds=1200,max_rss_GiB=16))


def solve(run,name):
    import warp as wp
    run=Path(run);verify(run);p=read(run/'S2/direction-protocol.json')
    if name not in p['cases']:raise ValueError('unregistered material case')
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');start=time.perf_counter();stop=threading.Event()
    def guard():
        while not stop.wait(.5):
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
            if rss>16 or time.perf_counter()-start>1200:
                write(run/f'S2/direction/{name}/failure.json',dict(reason='resource_limit',seconds=time.perf_counter()-start,peak_rss_GiB=rss));os._exit(75)
    threading.Thread(target=guard,daemon=True).start()
    if name=='formal':
        r,_=load_selected(read(run/'selected-space.json')['package'])
        with np.load(LOCAL/'S1/candidates/global-snapshot6/static.npz') as z:initial=z['full'].copy()
    else:r,initial,_,_=external_reload(APP,int(name[1:]))
    old=r;space=copy.copy(r.parent);space.params=replace(space.params,fiber_direction=np.array(p['fiber_direction']));space.A=space.params.A0
    space.signature=digest(dict(parent=old.parent.signature,material=dict(mu=space.params.mu,lam=space.params.lam,k_f=space.params.k_f,fiber=space.params.fiber_direction.tolist()),purpose='registered static generalization',Ks='original unchanged'))
    space.metadata=dict(space.metadata,material_variation_fiber_angle_degrees=60.,original_Ks=True)
    r=Condensation(space,old.original_mass,old.original_stiffness)
    if not np.array_equal(r.P,old.P) or not np.array_equal(r.M,old.M) or not np.array_equal(space.Ks,old.parent.Ks):raise ValueError('variation changed physical mass/condensation/stabilization')
    m=SegmentedModel(r,order=7,device='cuda:0',hold=.005);state,stats=bounded_static(m,initial,run/f'S2/search-diagnostics/{name}.json',max_seconds=max(1,600-(time.perf_counter()-start)));frame=probe_frame(m,state,[33,7,7]);del m,old;gc.collect()
    checks=qcheck(r,state.q);folder=run/'S2/direction'/name;folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'state-fields.npz',q=state.q,full=r.expand(state.q),nodes=space.nodes(r.expand(state.q)),edge0=space.edges[0],edge1=space.edges[1],edge2=space.edges[2],p=space.p,**frame)
    write(folder/'result.json',dict(status='passed_scoped',solve=stats,material=checks,model_space=space.signature,k_f=200.,fiber_angle_degrees=60.,original_mass_and_Ks=True,old_rest_matrix_search_only=True,peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,seconds=time.perf_counter()-start,data_sha256=sha(folder/'state-fields.npz')))
    stop.set()
    print('DIRECTION_VARIATION',name,stats['nonlinear_residual_N'],flush=True)

def analyze(run):
    from engine.aniso_phase1.types import AnisotropicMaterialParams
    run=Path(run);p=read(run/'S2/direction-protocol.json');names=p['cases'];data={name:np.load(run/'S2/direction'/name/'state-fields.npz') for name in names};a,b,c=names
    def nodal(name):
        z=data[name];return (tuple(z[f'edge{i}'] for i in range(3)),int(z['p']),z['nodes'])
    params=AnisotropicMaterialParams(10.,20.,200.,np.array(p['fiber_direction']));adj=compare_nodal(nodal(b),nodal(c),params.A0,params);formal=compare_nodal(nodal(a),nodal(c),params.A0,params)
    fields={k:metric(data[a]['x']-data[a]['X'],data[c]['x']-data[c]['X'],5e-5,.05,w) for k,w in regions(data[c]['X']).items()}
    reaction=metric(read(run/f'S2/direction/{a}/result.json')['solve']['reaction_N'],read(run/f'S2/direction/{c}/result.json')['solve']['reaction_N'],1e-4,.05)
    passed=all(x['passed'] for reg in formal.values() for x in reg.values()) and all(x['passed'] for x in fields.values()) and reaction['passed']
    reliable=all(x['passed'] and x['absolute']<=.25*x['budget'] for reg in adj.values() for x in reg.values())
    write(run/'S2/direction-check.json',dict(status='passed_scoped' if passed and reliable else 'material_or_reference_limited',formal_fields_passed=passed,reference_resolved_at_engineering_budget=reliable,
        errors=formal,reference_adjacent=adj,displacement=fields,reaction=reaction,hidden=False,scope='one F60 static .005 k_f=200 variation; fixed basis and original Ks; not uniform material/continuum certification'))
    write(run/'S2/space-scope-decision.json',dict(status='passed_scoped' if passed and reliable else 'reference_or_space_limited',formal_space_unchanged=True,spatial_accuracy=False,
        reference_level=5,direction_F60_passed=passed and reliable,R6_needed=not reliable,new_formal_candidates=0))
    print('GENERALIZATION',passed,reliable,formal['interior'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','solve','analyze']);p.add_argument('--name');p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='solve':solve(a.run,a.name)
        else:globals()[a.phase](a.run)
