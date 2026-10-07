"""Bounded, independent E validation with raw fields and per-step ledgers."""
from time import perf_counter
import numpy as np
from engine.aniso_phase1.types import AnisotropicMaterialParams
from engine.aniso_phase1.research_e.flow import Grid, Darcy, oriented_permeability
from engine.aniso_phase1.research_e.poro import Skeleton, Biot
from engine.aniso_phase1.research_e.stage2.evaluation import (StressEvaluator, stress_errors, stress_passed,
    physical_diagnostics, weighted_error)
from engine.aniso_phase1.research_e.stage2.residual import step_residual, step_ledger, valid_ledger
from benchmarks.research_e.experiments import bc_all, load_factor, relative
from benchmarks.research_e.reference import consolidation, consolidation_average, biot_manufactured, sine_fields


def make_model(n, *, dim=2, stiffness=200., angle=45., ratio=10., storage=.001, clamped=False):
    g=Grid((n,)*dim)
    material=AnisotropicMaterialParams(10,20,stiffness,[np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0])
    clamp=(lambda x,c:bool(np.any((x<1e-12)|(x>1-1e-12)))) if clamped else None
    s=Skeleton(g,material,clamp=clamp)
    K=oriented_permeability(.01,.01/ratio,np.deg2rad(angle)) if dim==2 else np.array([[.01]])
    return Biot(s,Darcy(g,K),storage)


def observed_step(b, old, result, dt, load, boundary, evaluator, source=0.):
    values=evaluator.evaluate(result.state.u,result.state.p)
    diag=physical_diagnostics(evaluator,values)
    residual=step_residual(b,old,result,dt,load,boundary,source)
    record=dict(result.metrics,time=result.state.time,**diag,
                residual_mechanics=residual['mechanics'],residual_mass=residual['mass'],residual_darcy=residual['darcy'])
    record['passed']=bool(result.converged and valid_ledger(result.metrics,residual,diag))
    return values,record,step_ledger(b,old,result,dt,source)


def save_ledger(raw,records,local):
    columns=list(records[0]);raw['ledger_columns']=np.array(columns)
    raw['ledger']=np.array([[r[k] for k in columns] for r in records],float)
    raw['local_mass_defect']=np.asarray(local)


def summary(records):
    return dict(passed=all(r['passed'] for r in records),steps=len(records),
        max_mass=max(r['mass_defect'] for r in records),max_local_mass=max(r['local_mass_defect'] for r in records),
        max_energy=max(abs(r['energy_residual']) for r in records),max_momentum=max(r['momentum_balance'] for r in records),
        max_residual=max(max(r['residual_mechanics'],r['residual_mass'],r['residual_darcy']) for r in records),
        physical_dissipation=sum(r['physical_dissipation'] for r in records),numerical_loss=sum(r['numerical_loss'] for r in records),
        external_work=sum(r['external_work'] for r in records),min_J=min(r['min_J'] for r in records),
        porosity_range=[min(r['porosity_min'] for r in records),max(r['porosity_max'] for r in records)])


def consolidation_audit(out,protocol):
    cfg=protocol['consolidation'];rows=[]
    cases=[('space',n,cfg['spatial_dt']) for n in cfg['grids']]+[('time',cfg['time_grid'],dt) for dt in cfg['time_dt']]
    for study,n,dt in cases:
        start=perf_counter();b=make_model(n,dim=1,stiffness=0);s=b.solid;g=s.grid;e=StressEvaluator(s)
        load=s.traction(0,1,[-.1]);old=b.initial(np.full(n,.1/1.04),load)
        boundary={(0,0):('flux',0.),(0,1):('pressure',0.)};records=[];local=[]
        for j in range(round(cfg['final_time']/dt)):
            r=b.step(old,dt,load,boundary)
            values,record,mass=observed_step(b,old,r,dt,load,boundary,e)
            records.append(record);local.append(mass);old=r.state
        pex=consolidation_average(n,cfg['final_time']);_,uex=consolidation(s.nodes[:,0],cfg['final_time'])
        ref=e.reference(lambda x:np.array([[(consolidation(x[0],cfg['final_time'])[0]-.1)/40]]),
                        lambda x:consolidation(x[0],cfg['final_time'])[0])
        errors=stress_errors(e,values,ref)
        row=dict(study=study,n=n,dt=dt,pressure_error=relative(old.p,pex),displacement_error=relative(old.u,uex),
                 stress=errors,ledger=summary(records),seconds=perf_counter()-start)
        row['passed']=bool(max(row['pressure_error'],row['displacement_error'])<.04 and stress_passed(errors) and row['ledger']['passed'])
        raw=dict(u=old.u,p=old.p,p_reference=pex,u_reference=uex,face_flux=r.flux,
                 reaction=(s.A@old.u-s.G.T@old.p-load)[s.fixed],points=e.points,weights=e.dV,**values)
        raw.update({'reference_'+k:v for k,v in ref.items()});save_ledger(raw,records,local)
        np.savez_compressed(out/f'consolidation_{study}_{n}_{dt}.npz',**raw);rows.append(row)
    trend={study:all(a['pressure_error']>b['pressure_error'] for a,b in zip([r for r in rows if r['study']==study], [r for r in rows if r['study']==study][1:])) for study in ('space','time')}
    return dict(passed=bool(rows[2]['passed'] and rows[-1]['passed'] and all(trend.values())),rows=rows,pressure_trend=trend,
                total_stress_reference='constant -0.1 Pa at all points; effective stress=p-0.1 Pa',units='1D unit area; stress in Pa')


def manufactured_case(out,n,dt,tag,protocol):
    start=perf_counter();cfg=protocol['manufactured'];b=make_model(n,clamped=True);s=b.solid;g=s.grid;e=StressEvaluator(s)
    ue,pe,force,source=biot_manufactured(s.material,b.flow.K[0],b.storage)
    load=s.body_force(force);source0=g.average(lambda x:source(x,0.));source1=g.average(lambda x:source(x,1.))-source0
    old=b.initial();boundary=bc_all(2);records=[];local=[]
    for j in range(1,round(cfg['final_time']/dt)+1):
        t=j*dt;src=source0+t*source1;r=b.step(old,dt,t*load,boundary,src)
        values,record,mass=observed_step(b,old,r,dt,t*load,boundary,e,src)
        records.append(record);local.append(mass);old=r.state
    T=cfg['final_time'];pex=T*g.average(pe,order=6);uex=T*np.array([ue(x) for x in s.nodes]).ravel()
    gradient=lambda x:T*np.outer(.0001*np.array([1.,.5]),sine_fields(x)[1])
    reference=e.reference(gradient,lambda x:T*pe(x))
    fine=StressEvaluator(s,6);r6=fine.reference(gradient,lambda x:T*pe(x))
    check=max(relative(reference[k],r6[k],1e-12) for k in ('effective_cell_mean','total_cell_mean'))
    norm_checks={}
    for kind in ('effective_full_field','total_full_field'):
        norm4=float(np.sum(e.dV*np.sum(reference[kind]**2,axis=(-1,-2))))
        norm6=float(np.sum(fine.dV*np.sum(r6[kind]**2,axis=(-1,-2))))
        norm_checks[kind]=abs(norm4-norm6)/max(norm6,1e-20)
    check=max(check,*norm_checks.values())
    errors=stress_errors(e,values,reference)
    old_metric=relative(s.cell_stress(old.u,old.p),reference['total_cell_mean'][...,:2,:2])
    pressure_error=relative(old.p,pex);displacement_error=relative(old.u,uex)
    row=dict(n=n,dt=dt,pressure_error=pressure_error,displacement_error=displacement_error,stress=errors,
        legacy_center_vs_mean_2x2=old_metric,corrected_mean_2x2=relative(values['total_cell_mean'][...,:2,:2],reference['total_cell_mean'][...,:2,:2]),
        reference_integration_difference=check,reference_full_L2_integration_checks=norm_checks,ledger=summary(records),seconds=perf_counter()-start,
        pressure_full_field=weighted_error(old.p[:,None]*np.ones(e.weights.shape),np.array([[T*pe(x) for x in cell] for cell in e.points]),e.dV))
    row['stress_passed']=stress_passed(errors,protocol['acceptance']['field_relative'])
    row['passed']=bool(max(pressure_error,displacement_error)<.04 and row['stress_passed'] and row['ledger']['passed'] and check<=1e-8)
    raw=dict(u=old.u,p=old.p,u_reference=uex,p_reference=pex,face_flux=r.flux,
             reaction=(s.A@old.u-s.G.T@old.p-T*load)[s.fixed],points=e.points,weights=e.dV,centers=g.centers,nodes=s.nodes,**values)
    raw.update({'reference_'+k:v for k,v in reference.items()});save_ledger(raw,records,local)
    np.savez_compressed(out/f'manufactured_{tag}_{n}_{dt}.npz',**raw)
    return row,raw


def manufactured_audit(out,protocol):
    cfg=protocol['manufactured'];rows=[]
    for n in cfg['grids']:
        row,_=manufactured_case(out,n,cfg['dt'],'space',protocol);rows.append(row)
        print('manufactured grid',n,'certified',row['passed'],flush=True)
        if n>=cfg['stop_after_first_certified_grid_at_least'] and row['passed']:break
    timerows=[];timeraw=[]
    for dt in cfg['independent_time_dt']:
        row,raw=manufactured_case(out,cfg['time_grid'],dt,'time',protocol);timerows.append(row);timeraw.append(raw)
    time_differences={k:[float(np.linalg.norm(timeraw[i][k]-timeraw[i+1][k])) for i in range(2)] for k in ('u','p','total_full_field','effective_full_field')}
    trend={}
    for region in ('global','boundary','interior'):
        for metric in ('total_cell_mean','total_full_field','effective_cell_mean','effective_full_field'):
            values=[r['stress'][region][metric]['absolute_rms'] for r in rows]
            trend[region+'_'+metric]=bool(all(b<=a+1e-9 for a,b in zip(values,values[1:])))
    pressure_trend=all(a['pressure_error']>b['pressure_error'] for a,b in zip(rows,rows[1:]))
    time_trend=all(v[1]<=v[0]+1e-9 for v in time_differences.values())
    return dict(passed=bool(rows[-1]['passed'] and all(trend.values()) and pressure_trend and time_trend and all(r['ledger']['passed'] for r in timerows)),
        rows=rows,time_rows=timerows,stress_absolute_trends=trend,pressure_trend=pressure_trend,time_absolute_differences=time_differences,
        time_trend=time_trend,time_note='analytic fields linear in time: BE exact derivative; remaining temporal differences are response to spatial projection error, not an independent nonzero analytic truncation error',
        old_comparison='legacy metric mixes 2x2 center stress and analytic mean; corrected means and 3x3 full fields separately saved')


def run_cycle(out,b,dt,tag,*,rate=1.,boundary_pressure=0.,sample=.04,keep_all=False):
    start=perf_counter();s=b.solid;g=s.grid;e=StressEvaluator(s);old=b.initial()
    boundary=bc_all(2,'flux');boundary[0,1]=('pressure',boundary_pressure);base=s.traction(0,1,[-.1,0])
    records=[];local=[];history={k:[] for k in ('time','pressure','displacement','stress_total','stress_effective','stress_center','stress_mean','reaction','face_flux')}
    steps=round(1.6/(dt*rate));stride=round(sample/(dt*rate))
    for j in range(1,steps+1):
        t=j*dt;load=base*load_factor(t*rate);r=b.step(old,dt,load,boundary)
        values,record,mass=observed_step(b,old,r,dt,load,boundary,e)
        records.append(record);local.append(mass);old=r.state
        if j%stride==0:
            fields=dict(time=t,pressure=old.p,displacement=old.u,stress_total=values['total_full_field'],
                        stress_effective=values['effective_full_field'],stress_center=values['total_center'],
                        stress_mean=values['total_cell_mean'],reaction=(s.A@old.u-s.G.T@old.p-load)[s.fixed],face_flux=r.flux)
            for k,v in fields.items():history[k].append(np.array(v,copy=True))
    arrays={k:np.array(v) for k,v in history.items()}
    raw=dict(arrays,centers=g.centers,nodes=s.nodes,points=e.points,weights=e.dV,shape=g.shape)
    save_ledger(raw,records,local);np.savez_compressed(out/(tag+'.npz'),**raw)
    row=dict(dt=dt,steps=steps,ledger=summary(records),seconds=perf_counter()-start,
             pressure_range=[float(arrays['pressure'].min()),float(arrays['pressure'].max())],
             stress_range=[float(arrays['stress_total'].min()),float(arrays['stress_total'].max())])
    row['passed']=bool(row['ledger']['passed'] and np.isfinite(list(row['pressure_range'])).all() and max(abs(v) for v in row['pressure_range'])<1)
    return row,arrays


def trajectory_error(a,b,scale):
    # RMS over common physical samples, and all declared tensor/DOF entries.
    rms=float(np.sqrt(np.mean((a-b)**2)));ref=float(np.sqrt(np.mean(b*b)))
    return dict(error=rms/(scale if ref<=1e-8*scale else ref),absolute_rms=rms,reference_rms=ref,
                normalization='fixed_absolute_scale' if ref<=1e-8*scale else 'relative_RMS')


def cycle_audit(out,protocol):
    cfg=protocol['cycle'];b=make_model(cfg['grid']);rows=[];solutions=[]
    for dt in cfg['dt']:
        row,arrays=run_cycle(out,b,dt,f'cycle_dt_{dt}');rows.append(row);solutions.append(arrays)
    fields=cfg['near_zero_scales'];reference=solutions[-1]
    for row,solution in zip(rows,solutions):
        row['whole']={k:trajectory_error(solution[k],reference[k],scale) for k,scale in fields.items()}
        row['stages']=[{k:trajectory_error(solution[k][i*10:(i+1)*10],reference[k][i*10:(i+1)*10],scale) for k,scale in fields.items()} for i in range(4)]
        row['ends']=[{k:trajectory_error(solution[k][i],reference[k][i],scale) for k,scale in fields.items()} for i in (9,19,29,39)]
    adjacent={k:[float(np.sqrt(np.mean((solutions[i][k]-solutions[i+1][k])**2))) for i in range(3)] for k in fields}
    trend={k:all(y<=x+1e-9 for x,y in zip(v,v[1:])) for k,v in adjacent.items()}
    target=rows[-2];checks=[target['whole']]+target['stages']+target['ends']
    accuracy=all(metric['error']<=.05 for check in checks for metric in check.values())
    return dict(passed=bool(all(r['passed'] for r in rows) and accuracy and all(trend.values())),rows=rows,
        adjacent_absolute_rms=adjacent,adjacent_shrinkage=trend,finest_comparison_passed=accuracy,
        reference='dt=0.005 is discrete reference only; certify dt=0.01 against same-space reference, including all stages and ends',
        sampling='0.04 s common physical times; every time step retains local mass and energy ledger',kinetic_energy=0.)


def pressure_audit(out,protocol):
    rows=[];raw={}
    for n in (4,8):
        for storage in (.01,1e-6,0.):
            for ratio in (1,1000):
                b=make_model(n,storage=storage,ratio=ratio,angle=np.rad2deg(.6));s=b.solid;g=s.grid
                schur=b.pressure_mechanical_schur();eig=np.linalg.eigvalsh(schur/g.volume);cb=(-1.)**g.indices.sum(axis=1)
                old=b.initial();bc=bc_all(2);load=s.traction(0,1,[-.1,0]);dt=.01
                direct=b.step(old,dt,load,bc);fixed=b.step(old,dt,load,bc,method='fixed_stress',tolerance=1e-11,maxiter=1000)
                dr=step_residual(b,old,direct,dt,load,bc);fr=step_residual(b,old,fixed,dt,load,bc)
                M,rhs,free,*_=b._system(old,dt,load,bc,0.)
                from scipy.sparse import diags
                signs=np.r_[np.ones(len(s.free)),-np.ones(g.nc),np.full(len(free),dt)]
                transformed=diags(signs)@M;difference=transformed-transformed.T
                sym=float(np.max(np.abs(difference.data),initial=0.))
                row=dict(n=n,storage=storage,ratio=ratio,null_modes=int(np.sum(eig<1e-9)),
                    checkerboard_energy=float(cb@schur@cb),direct_residual=dr,fixed_stress_residual=fr,
                    fixed_iterations=fixed.metrics['iterations'],solution_difference=relative(np.r_[fixed.state.u,fixed.state.p,fixed.flux],np.r_[direct.state.u,direct.state.p,direct.flux]),
                    raw_symmetric=False,transformed_symmetry_absolute=sym,spd=False)
                row['passed']=bool(row['null_modes']==0 and row['checkerboard_energy']>0 and direct.converged and fixed.converged and max(dr['maximum'],fr['maximum'])<=1e-8 and row['solution_difference']<1e-5 and sym<1e-10)
                rows.append(row);raw[f'spectrum_{n}_{storage}_{ratio}']=eig
    b=make_model(4,clamped=True,storage=0.,stiffness=0);eig=np.linalg.eigvalsh(b.pressure_mechanical_schur()/b.solid.grid.volume)
    mechanical_null=int(np.sum(abs(eig)<1e-9))
    M,rhs,*_=b._system(b.initial(),.01,np.zeros(b.solid.ndof),bc_all(2,'flux'),0.)
    sv=np.linalg.svd(M.toarray(),compute_uv=False);mixed_null=int(np.sum(sv<sv[0]*1e-12))
    # Gauge-constrained fully sealed system: explicit mean pressure, no added storage.
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    nu=len(b.solid.free);nc=b.solid.grid.nc;gauge=np.zeros(M.shape[0]);gauge[nu:nu+nc]=b.solid.grid.volume
    augmented=sp.bmat([[M,sp.csc_matrix(gauge[:,None])],[sp.csc_matrix(gauge[None,:]),None]],format='csc')
    gx=spsolve(augmented,np.r_[rhs,0.]);gauge_res=float(np.max(np.abs(M@gx[:-1]-rhs)))
    raw['clamped_spectrum']=eig;raw['sealed_singular_values']=sv;raw['sealed_gauge_solution']=gx
    np.savez_compressed(out/'pressure_modes.npz',**raw)
    return dict(passed=bool(all(r['passed'] for r in rows) and mechanical_null==1 and mixed_null==1 and gauge_res<1e-8),rows=rows,
        fully_clamped_mechanical_null_modes=mechanical_null,fully_sealed_zero_storage_null_modes=mixed_null,
        gauge_residual=gauge_res,gauge='volume-weighted mean pressure zero via Lagrange multiplier; no physical penalty',
        solvers='direct and converged fixed-stress certified here; GMRES allowed by algebra, not benchmarked; raw PCG/MINRES rejected; row scaling diag(1,-1,dt) symmetric indefinite, MINRES not certified')


def robustness_audit(out,protocol):
    cfg=protocol['robustness'];rows=[];sensitivity=[];idx=0
    for ratio in cfg['ratios']:
        for angle in cfg['angles_deg']:
            for stiffness in cfg['fiber_stiffness']:
                n=cfg['grid_cycle'][idx%3];rate=cfg['rate_by_angle'][(idx//3)%3];pb=cfg['pressure_cycle'][idx%2];storage=cfg['storage_cycle'][idx%3]
                b=make_model(n,stiffness=stiffness,angle=angle,ratio=ratio,storage=storage)
                row,arrays=run_cycle(out,b,.04/rate,f'robustness_{idx:02d}',rate=rate,boundary_pressure=pb,sample=.4)
                row.update(case=idx,n=n,ratio=ratio,angle_deg=angle,stiffness=stiffness,rate=rate,storage=storage,boundary_pressure=pb,accuracy_certified=False)
                rows.append(row)
                if idx in cfg['sensitivity_cases']:
                    bf=make_model(2*n,stiffness=stiffness,angle=angle,ratio=ratio,storage=storage)
                    rf,af=run_cycle(out,bf,.04/rate,f'robustness_{idx:02d}_refined',rate=rate,boundary_pressure=pb,sample=.4)
                    # Integral means over each original cell: nested grid aggregation preserves locations and volumes.
                    def cell_means(a):
                        return a.reshape((4,n,2,n,2)+a.shape[2:]).mean(axis=(2,4)).reshape((4,n*n)+a.shape[2:])
                    coarse_stress=arrays['stress_mean'];fine_stress=cell_means(af['stress_mean'])
                    fine_p=cell_means(af['pressure'])
                    # Q2 coarse nodes are an exact subset of refined nodes.
                    fine_u=af['displacement'].reshape(4,4*n+1,4*n+1,2)[:,::2,::2].reshape(4,-1)
                    sensitivity.append(dict(case=idx,coarse_n=n,fine_n=2*n,refined_ledger_passed=rf['passed'],
                        pressure=trajectory_error(arrays['pressure'],fine_p,.1),
                        stress_cell_mean=trajectory_error(coarse_stress,fine_stress,.1),
                        displacement=trajectory_error(arrays['displacement'],fine_u,.0025),
                        accuracy_certified=False,reason='one adjacent grid difference is sensitivity evidence, not certified continuum reference'))
                idx+=1
    return dict(passed=bool(all(r['passed'] for r in rows) and all(r['refined_ledger_passed'] for r in sensitivity)),cases=rows,
                failure_map=[r['case'] for r in rows if not r['passed']],sensitivity=sensitivity,
                coverage='36 ratio x fiber-angle x stiffness cases; remaining parameters deterministically cycled, not full Cartesian product',
                accuracy='stability only; including ratio 1000, no new analytic accuracy claim')
