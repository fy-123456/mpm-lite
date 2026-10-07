"""Bounded true-tangent correction for a slowly contracting old search metric."""
import time
import numpy as np
from .provenance import write

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
        if it>=2 and norm>.2*records[-2]['true_residual_N']:
            fallback=True;record['fallback_reason']='old search contracts too slowly; unchanged true tangent used'
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

