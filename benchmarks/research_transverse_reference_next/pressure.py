"""Bounded CPU reference study; no moving-solid accuracy claims."""
import argparse, time
import numpy as np
import scipy.linalg as la
from .provenance import *
from .runtime import update
from engine.aniso_phase1.research_transverse_reference_next.reference import algebra, exact, integrate, mode_increment_check
from engine.aniso_phase1.research_transverse_next.initial import pressure_profile
from benchmarks.research_transverse_next.observables import modes
from benchmarks.research_sequential_next.compare import metric


def mode_series(top,p):
    return {name:np.array([modes(top,row)[name]['value_Pa'] for row in p]) for name in ('Ay','Az')}


def compare(a, v, ex):
    checks=dict(pressure=metric(v['pressure'],ex['pressure'],.001,.05),
                content=metric(v['pressure']*a['C'],ex['pressure']*a['C'],1e-10,.05),
                face_flux=metric(v['flux'],ex['flux'],1e-10,.05),
                cumulative=metric(v['cumulative'],ex['cumulative'],1e-10,.05))
    top=a['top'];groups=np.array([top.boundary_sign*(top.axes==axis)*(top.boundary_sign==sgn) for axis in range(3) for sgn in (-1,1)])
    checks['six_sides']=metric(v['flux']@groups.T,ex['flux']@groups.T,1e-10,.05)
    checks['side_cumulative']=metric(v['cumulative']@groups.T,ex['cumulative']@groups.T,1e-10,.05)
    vm,em=mode_series(top,v['pressure']),mode_series(top,ex['pressure'])
    inc={k:mode_increment_check(vm[k],em[k]) for k in vm}
    return dict(checks=checks,mode_increments=inc,passed=all(x['passed'] for x in checks.values()) and all(x['passed'] for x in inc.values()))


def main(run):
    run=Path(run);mutable(run);started=time.perf_counter();p=read(run/'S0/physical-contract.json');v=p['parameters']
    a=algebra(p['cuts'],np.array(p['mobility']),v['storage'],v['reservoir_Pa'],source=p['source'])
    top=a['top'];obs=np.array(read(run/'S0/observation-contract.json')['observations_s'])
    checks=dict(cells=top.cells,nflux=top.nflux,symmetry=a['symmetry'],lambda_min=float(a['lam'][0]),lambda_max=float(a['lam'][-1]),volume_sum_m3=float(top.V0.sum()),source_zero=not np.any(a['rhs']+top.B@a['z0']))
    residual=np.linalg.norm(a['L']@a['pe']-a['rhs'])/(np.linalg.norm(a['L'])*np.linalg.norm(a['pe'])+np.linalg.norm(a['rhs']))
    checks['equilibrium_scaled_residual']=float(residual)
    write(run/'S1/operator-check.json',dict(status='passed_scoped',**checks))
    np.savez_compressed(run/'S1/operator.npz',H=a['H'],C=a['C'],B=top.B,L=a['L'],gb=a['gb'],lam=a['lam'],pe=a['pe'])
    records=[];selfchecks=[];steps=0;timescale=[]
    for case in ('YZ128','Y128'):
        p0,definition=pressure_profile(case,top);ex=exact(a,p0,obs);ms=mode_series(top,ex['pressure'])
        coeff=a['Q'].T@(a['D']*(p0-a['pe']))
        np.savez_compressed(run/'S1'/f'reference-{case}.npz',times=obs,pressure=ex['pressure'],cumulative=ex['cumulative'],flux=ex['flux'],Ay=ms['Ay'],Az=ms['Az'],p0=p0)
        mass=(ex['pressure']-p0)@a['C']+ex['cumulative']@top.boundary_sign
        initerr=float(np.max(abs(ex['pressure'][0]-p0)))
        row=dict(case=case,initial_error_Pa=initerr,max_content_balance_m3=float(np.max(abs(mass))),minimum_pressure_Pa=float(ex['pressure'].min()),equilibrium_scaled_residual=float(residual))
        if case=='YZ128':
            aug=np.zeros((len(p0)+1,len(p0)+1));aug[:-1,:-1]=-a['A'];aug[:-1,-1]=a['rhs']/a['D']
            errors=[]
            for t in (25e-6,175e-6):
                independent=(la.expm(t*aug)@np.r_[a['D']*p0,1.])[:-1]/a['D'];ref=exact(a,p0,[t])['pressure'][0]
                err=float(np.max(abs(independent-ref)));errors.append(dict(time_s=t,error_Pa=err,passed=err<=1e-8+1e-6*float(np.max(abs(ref)))))
            row['independent_matrix_exponential']=errors
        row['passed']=initerr<=1e-10 and max(abs(mass))<=1e-10 and residual<=1e-8 and ex['pressure'].min()>=0 and all(x['passed'] for x in row.get('independent_matrix_exponential',[]))
        selfchecks.append(row)
        for label in ('h','half'):
            t=np.array(p['times'][label]);actual=integrate(a,p0,t);ref=exact(a,p0,t);steps+=len(t)-1
            ids=np.array([np.argmin(abs(t-x)) for x in obs]);assert np.allclose(t[ids],obs,rtol=0,atol=1e-15)
            eng=dict(pressure=actual['pressure'][ids],cumulative=actual['cumulative'][ids],flux=np.diff(actual['cumulative'][ids],axis=0)/np.diff(obs)[:,None])
            cmp=compare(a,eng,ex);raw=compare(a,actual,ref)
            hard=all(abs(x['energy_balance_J'])<=1e-9 and x['mass_defect_m3']<=1e-10 and x['source_work_J']==0 and x['darcy_dissipation_J']>=0 for x in actual['ledger']) and actual['pressure'].min()>=0
            records.append(dict(case=case,schedule=label,engineering=cmp,raw=raw,hard_passed=bool(hard),matrix_steps=len(t)-1,maximum_energy_defect_J=max(abs(x['energy_balance_J']) for x in actual['ledger'])))
            am=mode_series(top,actual['pressure']);np.savez_compressed(run/'S1'/f'{case}-{label}.npz',times=t,pressure=actual['pressure'],cumulative=actual['cumulative'],flux=actual['flux'],Ay=am['Ay'],Az=am['Az'],theta=actual['theta'])
        centres=np.mean(np.asarray(top.cell_bounds),axis=2)
        mode_weights={}
        for axis,name in ((1,'Ay'),(2,'Az')):
            eta=2*(centres[:,axis]-.5)/.25;w=top.V0*eta/(top.V0@(eta*eta));amplitude=(w/a['D'])@a['Q']*coeff
            mode_weights[name]=dict(initial_amplitude_contributions_Pa=amplitude.tolist(),tau_s=(1/a['lam']).tolist())
        timescale.append(dict(case=case,modes={k:vv.tolist() for k,vv in ms.items()},times_s=obs.tolist(),eigenmode_weights=mode_weights))
    passed=all(x['passed'] for x in selfchecks)
    write(run/'S1/reference-self-check.json',dict(status='passed_scoped' if passed else 'limited',records=selfchecks,new_CPU_matrix_steps=steps,independent_expm_calls=2,seconds=time.perf_counter()-started))
    write(run/'S1/reference-input.json',dict(definitions={k:pressure_profile(k,top)[1] for k in ('YZ128','Y128')},parameters=v,source_zero=True,source_sha256=all_sources()))
    write(run/'S1/fixed-skeleton-time-review.json',dict(status='passed_scoped' if passed and all(x['engineering']['passed'] and x['hard_passed'] for x in records) else 'limited',records=records,scope='fixed skeleton same128grid only; raw intervals separate',coupled_temporal_accuracy=False))
    write(run/'S1/timescales.json',dict(records=timescale))
    mainrow=next(x for x in records if x['case']=='YZ128' and x['schedule']=='h')
    ref=timescale[0];changes={name:abs(ref['modes'][name][10]-ref['modes'][name][6]) for name in ('Ay','Az')}
    later={name:abs(ref['modes'][name][14]-ref['modes'][name][10]) for name in ('Ay','Az')}
    decision=dict(status='registered',reference_valid=passed,decision='extend4' if passed and max(changes.values())>5e-5 else 'limited_signal',additional_four_if_needed=passed and max(later.values())>5e-5,change_75_to125_Pa=changes,change_125_to175_Pa=later,half_coupled_trigger=passed and not mainrow['engineering']['passed'],raw_peak_accuracy=mainrow['raw']['passed'],extension_max_steps=8,reason='same-grid spectral signal and engineering time budgets; moving solid accuracy remains unqualified')
    register(run,'S1/window-and-time-decision.json',decision)
    if steps>200 or not passed:raise ValueError('reference self-check or budget failed; inspect scoped evidence')
    update(run,f'S1：128格固定骨架谱参考与独立矩阵指数核对通过；CPU时间步{steps}，扩窗决定{decision["decision"]}，耦合h/2触发={decision["half_coupled_trigger"]}。固定骨架参考不充当运动固体真值。')
    print('REFERENCE',decision,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
