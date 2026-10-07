"""Bounded task E experiments; all timings are shared-host CPU observations."""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from engine.aniso_phase1.types import AnisotropicMaterialParams
from engine.aniso_phase1.constitutive import energy, pk1
from engine.aniso_phase1.direction_moments import audit_direction_mixture
from engine.aniso_phase1.research_e.material import FullMaterial, tangent, linear_stress
from engine.aniso_phase1.research_e.flow import Grid, Darcy, oriented_permeability, quadrature
from engine.aniso_phase1.research_e.poro import Skeleton, Biot, PoroState, CouplingTransaction
from engine.aniso_phase1.research_e.drag import implicit_drag, drag_tensor, exact_relative
from engine.aniso_phase1.research_e.state import SaturatedState
from .reference import consolidation, consolidation_average, sine_fields, biot_manufactured


def relative(a,b,scale=1e-12): return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b),scale))
def bc_all(dim, kind='pressure', value=0.): return {(d,s):(kind,value) for d in range(dim) for s in (0,1)}

def material_audit(out, protocol):
    F=np.array([[1.04,.03,0],[.01,.99,.02],[0,0,1.01]])
    D=np.array([[.2,-.1,.05],[.01,.03,0],[0,.02,-.01]])
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    rows=[]
    for label,angle,kf in [('ISO',0,0),('F0',0,200),('F45',45,200),('F90',90,200)]:
        a=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0])
        p=AnisotropicMaterialParams(10,20,kf,a);A=p.A0
        exactE=float(np.sum(pk1(F,A,p)*D));exactT=tangent(F,A,D,p)
        curves=[]
        for eps in [1e-3,1e-4,1e-5,1e-6]:
            de=(energy(F+eps*D,A,p)-energy(F-eps*D,A,p))/(2*eps)
            dt=(pk1(F+eps*D,A,p)-pk1(F-eps*D,A,p))/(2*eps)
            curves.append(dict(eps=eps,energy_relative=abs(de-exactE)/max(abs(exactE),1.),tangent_relative=relative(dt,exactT,1.)))
        H=np.column_stack([tangent(F,A,e.reshape(3,3),p).ravel() for e in np.eye(9)])
        rows.append(dict(material=label,curves=curves,energy=energy(F,A,p),stress=pk1(F,A,p).tolist(),
            objectivity=relative(pk1(R@F,A,p),R@pk1(F,A,p),1.),symmetry=relative(H,H.T,1.)))
    dirs=[[1,0,0],[1,1,0],[0,1,0]]
    moments=audit_direction_mixture(F,dirs)
    m=FullMaterial(dirs,[.2,.3,.5],AnisotropicMaterialParams(10,20,200))
    Fs=np.array([F,np.diag([1.02,.99,1]),np.diag([.98,1.03,1.01])])
    actual=m.evaluate(Fs)[0];averaged=m.evaluate(np.broadcast_to(np.einsum('n,nij->ij',m.volumes,Fs),Fs.shape))[0]
    passed=all(max(r['curves'][2]['energy_relative'],r['curves'][2]['tangent_relative'],r['objectivity'],r['symmetry'])<protocol['acceptance']['material_directional_relative'] for r in rows)
    return dict(passed=passed,rows=rows,common_F_fourth_moment=moments,nonuniform_full_energy=actual,
                invalid_averaged_F_energy=averaged,nonuniform_F_compression_used=False)


def flow_audit(out,protocol):
    rows=[]; worst_scene=None
    for angle,ratio in [(0,1),(0,10),(45,10),(45,100)]:
        K=oriented_permeability(.01,.01/ratio,np.deg2rad(angle))
        group=[]
        for n in protocol['darcy_grids']:
            start=perf_counter();g=Grid((n,n));flow=Darcy(g,K)
            pressure=lambda x:sine_fields(x)[0]
            source=lambda x:-float(np.sum(K*sine_fields(x)[2]))
            r=flow.solve(bc_all(2),source)
            pavg=g.average(pressure);qavg=g.average(lambda x:-K@sine_fields(x)[1])
            ep=eq=np_=nq=0.
            for s,w in quadrature(2,4):
                points=(g.indices+s)*g.h
                pex=np.array([pressure(x) for x in points]);qex=np.array([-K@sine_fields(x)[1] for x in points])
                qnum=r.face_flux[g.cell_faces]@g.flux_basis(s)
                ep+=w*np.sum((r.pressure-pex)**2);eq+=w*np.sum((qnum-qex)**2)
                np_+=w*np.sum(pex**2);nq+=w*np.sum(qex**2)
            row=dict(n=n,angle_deg=angle,permeability_ratio=ratio,pressure_cell_mean_error=relative(r.pressure,pavg),
                flux_cell_mean_error=relative(r.cell_flux,qavg),pressure_full_L2=float(np.sqrt(ep/np_)),flux_full_L2=float(np.sqrt(eq/nq)),
                mass_defect=float(np.max(abs(r.mass_residual))),true_residual=r.true_residual,
                dissipation=r.dissipation_rate,pressure_dof=g.nc,flux_dof=g.nf,seconds=perf_counter()-start)
            rows.append(row);group.append(row)
            if angle==45 and ratio==10 and n==max(protocol['darcy_grids']):worst_scene=(g,r,K)
        assert len(group)==3
    g=Grid((8,8));f=Darcy(g,oriented_permeability(.01,.001,.5));bc=bc_all(2,'flux')
    linear=np.array([1.,-.4]);q=-f.K[0]@linear
    for d in range(2):
        for side in (0,1):bc[d,side]=('flux',(2*side-1)*q[d])
    neumann=f.solve(bc,mean_pressure=2.)
    compatible_error=relative(neumann.pressure,2+(g.centers-.5)@linear)
    try:f.solve(bc,source=1.);rejected=False
    except ValueError:rejected=True
    threshold=protocol['acceptance'];passed=True
    for start in range(0,len(rows),3):
        a,b,c=rows[start:start+3]
        passed &= c['pressure_cell_mean_error']<threshold['darcy_finest_cell_mean_relative'] and c['flux_cell_mean_error']<threshold['darcy_finest_cell_mean_relative']
        passed &= max(c['pressure_full_L2'],c['flux_full_L2'])<threshold['darcy_finest_full_field_relative']
        passed &= c['pressure_full_L2']<b['pressure_full_L2']<a['pressure_full_L2'] and c['flux_full_L2']<b['flux_full_L2']<a['flux_full_L2']
    passed &= max(r['mass_defect'] for r in rows)<1e-6 and rejected and compatible_error<1e-6
    g,r,K=worst_scene
    np.savez_compressed(out/'darcy_fields.npz',centers=g.centers,pressure=r.pressure,flux=r.cell_flux,face_flux=r.face_flux,K=K,shape=g.shape)
    return dict(passed=bool(passed),rows=rows,neumann_mean_pressure=2.,neumann_error=compatible_error,incompatible_neumann_rejected=rejected)


def drag_audit(out,protocol):
    rows=[]
    for ratio in [1,10,1000]:
        D=drag_tensor(oriented_permeability(1.,1/ratio,.6),1.,.35,1.)
        for density_ratio in [.1,1.,10.]:
            ms,mf=2.,2.*density_ratio;T=.5
            ref=exact_relative([1.,-.4],ms,mf,D,T)
            for steps in [1,16,32,64]:
                vs=np.zeros(2);vf=np.array([1.,-.4]);mom=balance=0.;physical=numeric=0.
                for i in range(steps):
                    r=implicit_drag(vs,vf,ms,mf,D,T/steps)
                    vs,vf=r.solid_velocity,r.fluid_velocity;mom=max(mom,float(np.max(abs(r.momentum_defect))))
                    balance=max(balance,abs(r.energy_residual));physical+=r.physical_dissipation;numeric+=r.numerical_loss
                rows.append(dict(ratio=ratio,density_ratio=density_ratio,steps=steps,dt=T/steps,
                    relative_velocity_error=relative(vf-vs,ref,1.),momentum_defect=mom,energy_residual=balance,
                    physical_dissipation=physical,numerical_loss=numeric,finite=bool(np.isfinite(np.r_[vs,vf]).all())))
    passed=all(r['finite'] and r['momentum_defect']<1e-6 and r['energy_residual']<1e-6 and r['physical_dissipation']>=0 for r in rows)
    for start in range(0,len(rows),4):
        _,a,b,c=rows[start:start+4];passed &= c['relative_velocity_error']<b['relative_velocity_error']<a['relative_velocity_error']
    return dict(passed=bool(passed),rows=rows,large_dt_stability_is_not_accuracy=True)


def consolidation_audit(out,protocol):
    rows=[];raw={}
    for study in ['space','time']:
        cases=[(n,.0005) for n in protocol['consolidation_grids']] if study=='space' else [(32,dt) for dt in [.02,.01,.005]]
        for n,dt in cases:
            start=perf_counter();g=Grid((n,));solid=Skeleton(g,AnisotropicMaterialParams(10,20,0));b=Biot(solid,Darcy(g,[[.01]]),.001)
            load=solid.traction(0,1,[-.1]);old=b.initial(np.full(n,.1/1.04),load)
            boundary={(0,0):('flux',0.),(0,1):('pressure',0.)};trace=[]
            outflow=0.;maxmass=maxenergy=maxmomentum=0.;physical=numeric=0.
            for step in range(round(.2/dt)):
                r=b.step(old,dt,load,boundary);old=r.state;m=r.metrics
                if not r.converged:raise RuntimeError('consolidation step failed')
                outflow+=m['boundary_outflow_volume'];physical+=m['physical_dissipation'];numeric+=m['numerical_loss']
                maxmass=max(maxmass,m['local_mass_defect']);maxenergy=max(maxenergy,abs(m['energy_residual']));maxmomentum=max(maxmomentum,m['momentum_balance'])
                trace.append([old.time,float(old.p.mean()),float(old.u[-1]),outflow,m['elastic_energy'],m['storage_energy'],physical,numeric])
            rp=consolidation_average(n,.2);_,ru=consolidation(solid.nodes[:,0],.2)
            row=dict(study=study,n=n,dt=dt,pressure_error=relative(old.p,rp),displacement_error=relative(old.u,ru),
                pressure_absolute_rms=float(np.sqrt(np.mean((old.p-rp)**2))),mass_defect=maxmass,energy_residual=maxenergy,
                momentum_balance=maxmomentum,outflow_volume=outflow,solid_dof=solid.ndof,pressure_dof=g.nc,flux_dof=g.nf,
                material_points=solid.integration_points,seconds=perf_counter()-start)
            rows.append(row);raw[f'{study}_{n}_{dt}_trace']=np.array(trace);raw[f'{study}_{n}_{dt}_pressure']=old.p
    finest=rows[2];tfine=rows[-1]
    passed=max(finest['pressure_error'],finest['displacement_error'],tfine['pressure_error'],tfine['displacement_error'])<protocol['acceptance']['consolidation_finest_relative']
    passed &= rows[2]['pressure_error']<rows[1]['pressure_error']<rows[0]['pressure_error']
    passed &= rows[-1]['pressure_error']<rows[-2]['pressure_error']<rows[-3]['pressure_error']
    passed &= max(r['mass_defect'] for r in rows)<1e-6
    np.savez_compressed(out/'consolidation_raw.npz',**raw)
    return dict(passed=bool(passed),rows=rows,load_step_at_t0=True,early_absolute_scale_Pa=.1,
                trace_columns=['time','mean_pressure','tip_displacement','outflow_volume','elastic_energy','storage_energy','cumulative_Darcy_loss','cumulative_numerical_loss'])


def oblique_audit(out,protocol):
    rows=[];raw={};params=AnisotropicMaterialParams(10,20,200,[1,1,0]);K=oriented_permeability(.01,.001,np.pi/4)
    ue,pe,force,source=biot_manufactured(params,K,.001)
    for n in [4,8,16]:
        start=perf_counter();g=Grid((n,n));clamp=lambda x,c:bool(np.any((x<1e-12)|(x>1-1e-12)))
        s=Skeleton(g,params,clamp=clamp);b=Biot(s,Darcy(g,K),.001)
        load=s.body_force(force);old=b.initial();boundary=bc_all(2);maxmass=maxenergy=0.
        for j in range(1,11):
            time=j*.01;r=b.step(old,.01,time*load,boundary,source=lambda x:source(x,time));old=r.state
            if not r.converged:raise RuntimeError('oblique manufactured step failed')
            maxmass=max(maxmass,r.metrics['local_mass_defect']);maxenergy=max(maxenergy,abs(r.metrics['energy_residual']))
        pexact=.1*g.average(pe);uexact=.1*np.array([ue(x) for x in s.nodes]).ravel()
        stress=s.cell_stress(old.u,old.p)
        exact_grad=.1*np.einsum('i,cj->cij',.0001*np.array([1.,.5]),g.average(lambda x:sine_fields(x)[1]))
        exact_stress=linear_stress(exact_grad,params)-pexact[:,None,None]*np.eye(2)
        partitions={name:relative(stress[mask],exact_stress[mask]) for name,mask in [('boundary',np.any((g.centers<.26)|(g.centers>.74),axis=1)),('interior',np.all((g.centers>=.26)&(g.centers<=.74),axis=1))]}
        rows.append(dict(n=n,stress_cell_mean_error=relative(stress,exact_stress),stress_partition_errors=partitions,pressure_error=relative(old.p,pexact),displacement_error=relative(old.u,uexact),
            pressure_absolute_rms=float(np.sqrt(np.mean((old.p-pexact)**2))),mass_defect=maxmass,energy_residual=maxenergy,
            stress_min=float(stress.min()),stress_max=float(stress.max()),solid_dof=s.ndof,pressure_dof=g.nc,flux_dof=g.nf,
            material_points=s.integration_points,seconds=perf_counter()-start))
        raw[f'pressure_{n}']=old.p;raw[f'pressure_exact_{n}']=pexact;raw[f'displacement_{n}']=old.u;raw[f'displacement_exact_{n}']=uexact
    np.savez_compressed(out/'oblique_manufactured_raw.npz',**raw)
    passed=max(rows[-1]['pressure_error'],rows[-1]['displacement_error'])<.04
    passed &= rows[-1]['pressure_error']<rows[-2]['pressure_error']<rows[-3]['pressure_error']
    return dict(passed=bool(passed),rows=rows,exact_time_dependence='linear; backward Euler derivative exact',spatial_reference='analytic manufactured displacement and pressure')


def pressure_audit(out,protocol):
    rows=[]
    for n in [4,8]:
        for storage in [.01,1e-6,0.]:
            for contrast in [1,1000]:
                g=Grid((n,n));p=AnisotropicMaterialParams(10,20,200,[1,1,0]);s=Skeleton(g,p)
                b=Biot(s,Darcy(g,oriented_permeability(.01,.01/contrast,.6)),storage)
                S=b.pressure_mechanical_schur();eig=np.linalg.eigvalsh(S/g.volume)
                cb=(-1.)**g.indices.sum(axis=1)
                r=b.step(b.initial(),.01,s.traction(0,1,[-.1,0]),bc_all(2))
                rows.append(dict(n=n,storage=storage,contrast=contrast,mechanical_pressure_eigen_min=float(eig[0]),
                    mechanical_pressure_eigen_max=float(eig[-1]),extra_null_modes=int(np.sum(eig<1e-9)),
                    checkerboard_energy=float(cb@S@cb),pressure_min=float(r.state.p.min()),pressure_max=float(r.state.p.max()),
                    true_residual=r.metrics['true_residual'],converged=r.converged))
    # Fully clamped mechanics has exactly the expected constant pressure gauge.
    g=Grid((4,4));clamp=lambda x,c:bool(np.any((x<1e-12)|(x>1-1e-12)))
    s=Skeleton(g,AnisotropicMaterialParams(10,20,0),clamp=clamp)
    b=Biot(s,Darcy(g,np.eye(2)),0);eig=np.linalg.eigvalsh(b.pressure_mechanical_schur()/g.volume)
    gauge_modes=int(np.sum(abs(eig)<1e-9))
    return dict(passed=bool(all(r['extra_null_modes']==0 and r['converged'] and r['checkerboard_energy']>0 for r in rows) and gauge_modes==1),
        rows=rows,fully_clamped_constant_gauge_modes=gauge_modes,physical_storage_separate_from_stabilization=True,numerical_pressure_stabilization=0.)


def load_factor(time):
    if time <= .4:return time/.4
    if time <= .8:return 1.
    if time <= 1.2:return (1.2-time)/.4
    return 0.


def cycle_audit(out,protocol):
    rows=[];solutions=[];allraw={}
    g=Grid((8,8));s=Skeleton(g,AnisotropicMaterialParams(10,20,200,[1,1,0]))
    b=Biot(s,Darcy(g,oriented_permeability(.01,.001,np.pi/4)),.001)
    boundary=bc_all(2,'flux');boundary[0,1]=('pressure',0.);base=s.traction(0,1,[-.1,0])
    for dt in protocol['cycle_dt']:
        start=perf_counter();tx=CouplingTransaction(b,b.initial());trace=[];ps=[];us=[];stresses=[];fluxes=[]
        maxmass=maxenergy=maxmomentum=0.;D=N=W=outflow=0.;nrange=[1.,0.];Jmin=1.
        for j in range(1,round(1.6/dt)+1):
            t=j*dt;r=tx.begin_trial(dt=dt,load=base*load_factor(t),boundary=boundary);tx.commit();old=tx.committed;m=r.metrics
            maxmass=max(maxmass,m['local_mass_defect']);maxenergy=max(maxenergy,abs(m['energy_residual']));maxmomentum=max(maxmomentum,m['momentum_balance'])
            D+=m['physical_dissipation'];N+=m['numerical_loss'];W+=m['external_work'];outflow+=m['boundary_outflow_volume']
            J=np.linalg.det(np.eye(2)+s.cell_gradients(old.u));n=1-(1-.35)/J
            nrange=[min(nrange[0],float(n.min())),max(nrange[1],float(n.max()))];Jmin=min(Jmin,float(J.min()))
            if j % round(.04/dt)==0:
                stress=s.cell_stress(old.u,old.p);reaction=(s.A@old.u-s.G.T@old.p)[s.fixed].reshape(-1,2).sum(axis=0)
                trace.append([t,old.p.mean(),old.p.min(),old.p.max(),old.u.reshape(-1,2)[:,0].min(),reaction[0],m['elastic_energy'],m['storage_energy'],D,N,W,outflow])
                ps.append(old.p.copy());us.append(old.u.copy());stresses.append(stress);fluxes.append(g.cell_flux(r.flux))
        arrays=dict(trace=np.array(trace),pressure=np.array(ps),displacement=np.array(us),stress=np.array(stresses),flux=np.array(fluxes))
        solutions.append(arrays)
        for k,v in arrays.items():allraw[f'dt_{dt}_{k}']=v
        rows.append(dict(dt=dt,steps=round(1.6/dt),max_local_mass_defect=maxmass,max_energy_residual=maxenergy,
            max_momentum_balance=maxmomentum,physical_dissipation=D,numerical_loss=N,external_work=W,
            pressure_min=float(arrays['pressure'].min()),pressure_max=float(arrays['pressure'].max()),
            porosity_range_from_kinematics=nrange,min_J=Jmin,seconds=perf_counter()-start))
    reference=solutions[-1];errors=[]
    for row,solution in zip(rows,solutions):
        row['errors_vs_finest']={k:relative(solution[k],reference[k],.1 if k=='pressure' else 1e-6) for k in ['pressure','displacement','stress']}
        row['stage_errors']=[{k:relative(solution[k][i*10:(i+1)*10],reference[k][i*10:(i+1)*10],.1 if k=='pressure' else 1e-6) for k in ['pressure','displacement','stress']} for i in range(4)]
        row['stage_end_errors']=[{k:relative(solution[k][i],reference[k][i],.1 if k=='pressure' else 1e-6) for k in ['pressure','displacement','stress']} for i in [9,19,29,39]]
        errors.append(row['errors_vs_finest']['pressure'])
    old=b.initial();load=.1*base;direct=b.step(old,.02,load,boundary)
    partition=[]
    for method,tol in [('once',1e-8),('fixed_stress',1e-7),('fixed_stress',1e-9)]:
        r=b.step(old,.02,load,boundary,method=method,tolerance=tol)
        partition.append(dict(method=method,tolerance=tol,converged=r.converged,iterations=r.metrics['iterations'],true_residual=r.metrics['true_residual'],pressure_difference=relative(r.state.p,direct.state.p,.1)))
    allraw.update(centers=g.centers,nodes=s.nodes,shape=np.array(g.shape))
    np.savez_compressed(out/'cycle_raw.npz',**allraw)
    passed=all(r['max_local_mass_defect']<1e-6 and r['max_energy_residual']<1e-6 and r['max_momentum_balance']<1e-6 and r['min_J']>0 and 0<r['porosity_range_from_kinematics'][0]<=r['porosity_range_from_kinematics'][1]<1 for r in rows)
    passed &= errors[2]<errors[1]<errors[0] and max(rows[2]['errors_vs_finest'].values())<protocol['acceptance']['cycle_finest_relative']
    passed &= all(max(stage.values())<protocol['acceptance']['cycle_finest_relative'] for stage in rows[2]['stage_errors'])
    passed &= all(p['converged'] for p in partition[1:]) and not partition[0]['converged']
    return dict(passed=bool(passed),rows=rows,partition=partition,solid_dof=s.ndof,pressure_dof=g.nc,flux_dof=g.nf,
        material_points=s.integration_points,kinetic_points=0,fluid_quadrature_points=g.nc*4,
        reaction_note='traction controlled: total reaction balance alone is not accuracy evidence',
        porosity_note='kinematic diagnostic; small-strain mass balance uses alpha div u + S p, not finite-J transport',
        trace_columns=['time','mean_p','min_p','max_p','min_ux','left_reaction_x','elastic','storage','Darcy_dissipation','numerical_loss','external_work','outflow_volume'])


def state_audit(out,protocol):
    N=4;V=np.full(N,.25);K0=np.diag([.01,.001,.001]);directions=np.tile([1.,0,0],(N,1))
    state=SaturatedState(V,.35,K0,directions);ref=None;maxmass=maxflux=0.;nmin=1.;nmax=0.;eigmin=1.;material_energy=[]
    m=FullMaterial(directions,V,AnisotropicMaterialParams(10,20,200));trial_history=[]
    for step in range(1,41):
        angle=.5*np.sin(step*np.pi/20);stretch=1+.02*np.sin(step*np.pi/20)
        R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
        F=np.broadcast_to(R@np.diag([stretch,1,1]),(N,3,3)).copy();J=np.linalg.det(F)
        old=state.committed;inflow=(J-np.linalg.det(old['F']))*V
        t=state.begin_trial(F,inflow,pressure=np.full(N,.1))
        maxmass=max(maxmass,float(np.max(abs(t['fluid_mass']-old['fluid_mass']-1000*inflow))))
        nmin=min(nmin,float(t['n'].min()));nmax=max(nmax,float(t['n'].max()));eigmin=min(eigmin,float(np.linalg.eigvalsh(t['K']).min()))
        # Affine Piola pullback: actual spatial permeability on a deformed domain.
        F2=F[0,:2,:2];J2=np.linalg.det(F2);inv=np.linalg.inv(F2);Kref=J2*inv@t['K'][0,:2,:2]@inv.T
        g=Grid((2,2));flow=Darcy(g,Kref);r=flow.solve(bc_all(2,value=lambda X:.1*X[0]))
        qsp=r.cell_flux@F2.T/J2;expected=-t['K'][0,:2,:2]@(inv.T@np.array([.1,0]))
        maxflux=max(maxflux,float(np.max(abs(qsp-expected))))
        state.commit();material_energy.append(m.evaluate(F)[0]);trial_history.append([step,stretch,angle,nmin,nmax])
        if step==20:
            state.checkpoint(out/'state_restart.json');ref=SaturatedState.restart(out/'state_restart.json')
        elif step>20:
            rold=ref.committed;rinflow=(J-np.linalg.det(rold['F']))*V
            ref.begin_trial(F,rinflow,pressure=np.full(N,.1));ref.commit()
    restart=max(float(np.max(abs(state.committed[k]-ref.committed[k]))) for k in state.committed)
    initial_F=state.committed['F'].copy();before=state.committed;failures=[]
    for mode in ['negative_J','invalid_porosity','missing_fluid_mass','nonfinite_pressure']:
        F=initial_F.copy();inc=np.zeros(N);pressure=np.zeros(N)
        if mode=='negative_J':F[:,0]*=-1
        elif mode=='invalid_porosity':F*=.5
        elif mode=='missing_fluid_mass':F*=1.1
        else:pressure[:]=np.nan
        try:state.begin_trial(F,inc,pressure=pressure);rejected=False
        except ValueError:rejected=True
        try:state.commit();blocked=False
        except RuntimeError:blocked=True
        same=all(np.array_equal(before[k],state.committed[k]) for k in before)
        failures.append(dict(mode=mode,rejected=rejected,commit_blocked=blocked,committed_unchanged=same))
    np.savez_compressed(out/'state_raw.npz',history=trial_history,nonlinear_material_energy=material_energy,**state.committed)
    return dict(passed=bool(maxmass<1e-6 and maxflux<1e-8 and restart<1e-10 and nmin>0 and nmax<1 and eigmin>0 and all(x['rejected'] and x['commit_blocked'] and x['committed_unchanged'] for x in failures)),
        porosity_range=[nmin,nmax],min_K_eigenvalue=eigmin,fluid_mass_defect=maxmass,piola_flux_error=maxflux,restart_difference=restart,
        solid_mass=float(state.committed['solid_mass'].sum()),fluid_mass=float(state.committed['fluid_mass'].sum()),
        failure_injection=failures,finite_kinematics='prescribed affine cycle with distributed reservoir source d(J V)/dt; polar rotation of K; exact Piola flow pullback',
        nonlinear_force_equilibrium_solved=False)


def robustness_audit(out,protocol):
    rows=[];idx=0
    for ratio in [1,10,100,1000]:
        for angle in [0,45,90]:
            for stiffness in [0,20,200]:
                n=[4,6,8][idx%3];rate=[.5,1.,2.][(idx//3)%3];pb=[0,.05][idx%2];density_ratio=[.1,1.,10.][idx%3]
                start=perf_counter();g=Grid((n,n));s=Skeleton(g,AnisotropicMaterialParams(10,20,stiffness,[np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0]))
                K=oriented_permeability(.01,.01/ratio,np.deg2rad(angle));b=Biot(s,Darcy(g,K),[0,.001,.01][idx%3]);tx=CouplingTransaction(b,b.initial())
                boundary=bc_all(2,'flux');boundary[0,1]=('pressure',pb)
                base=s.traction(0,1,[-.1,0]);mass=energy_res=momentum=0.;pmin=np.inf;pmax=-np.inf;physical=numeric=0.;all_converged=True
                pressure_regions={};stress_regions={};Jmin=1.;nmin=1.;nmax=0.;history=[]
                try:
                    for j in range(1,41):
                        r=tx.begin_trial(dt=.04/rate,load=base*load_factor(j*.04),boundary=boundary);tx.commit();m=r.metrics;old=tx.committed
                        mass=max(mass,m['local_mass_defect']);energy_res=max(energy_res,abs(m['energy_residual']));momentum=max(momentum,m['momentum_balance'])
                        pmin=min(pmin,float(old.p.min()));pmax=max(pmax,float(old.p.max()));physical+=m['physical_dissipation'];numeric+=m['numerical_loss']
                        J=np.linalg.det(np.eye(2)+s.cell_gradients(old.u));porosity=1-(1-.35)/J;Jmin=min(Jmin,float(J.min()));nmin=min(nmin,float(porosity.min()));nmax=max(nmax,float(porosity.max()))
                        history.append([j*.04/rate,float(old.p.min()),float(old.p.max()),m['mass_defect'],m['energy_residual'],m['physical_dissipation'],m['numerical_loss']])
                        if j in [10,20,30,40]:
                            stress=s.cell_stress(old.u,old.p)
                            for name,mask in [('clamp',g.centers[:,0]<.26),('interior',(g.centers[:,0]>.26)&(g.centers[:,0]<.74)),('drain',g.centers[:,0]>.74)]:
                                pressure_regions[f'{j}_{name}']=[float(old.p[mask].min()),float(old.p[mask].max())]
                                stress_regions[f'{j}_{name}']=[float(stress[mask].min()),float(stress[mask].max())]
                    np.savez_compressed(out/f'matrix_case_{idx:02d}.npz',pressure=old.p,displacement=old.u,stress=s.cell_stress(old.u,old.p),face_flux=r.flux,centers=g.centers,history=history)
                    reason=None
                except (ValueError,RuntimeError) as ex:
                    all_converged=False;reason=str(ex)
                D=drag_tensor(K,1.,.35,1.);drag=implicit_drag([0,0],[1,-.2],2,2*density_ratio,D,.04/rate)
                passed=bool(all_converged and mass<1e-6 and energy_res<1e-6 and momentum<1e-6 and physical>=-1e-10 and Jmin>0 and 0<nmin<=nmax<1 and np.isfinite([pmin,pmax]).all() and max(abs(pmin),abs(pmax))<1.)
                rows.append(dict(case=idx,n=n,permeability_ratio=ratio,fiber_angle_deg=angle,fiber_stiffness=stiffness,
                    loading_rate=rate,boundary_pressure=pb,density_ratio=density_ratio,storage=b.storage,passed=passed,reason=reason,
                    mass_defect=mass,energy_residual=energy_res,momentum_balance=momentum,physical_dissipation=physical,numerical_loss=numeric,
                    min_J=Jmin,porosity_range=[nmin,nmax],pressure_range=[pmin,pmax],pressure_regions=pressure_regions,stress_regions=stress_regions,
                    drag_momentum_defect=float(np.max(abs(drag.momentum_defect))),solid_mass=1300.,initial_fluid_mass=350.,
                    density_scope='density ratio only affects separate inertial drag; quasistatic Biot has no phase inertia',
                    diffusion_number=float(.01*(.04/rate)/((b.storage+1/40)*g.h[0]**2)),
                    permeability_eigenvalues=np.linalg.eigvalsh(K).tolist(),solid_dof=s.ndof,pressure_dof=g.nc,flux_dof=g.nf,
                    material_points=s.integration_points,seconds=perf_counter()-start))
                idx+=1
    return dict(passed=all(r['passed'] for r in rows),cases=rows,failure_map=[r['case'] for r in rows if not r['passed']],
        coverage='36 deterministic cases: full ratio x angle x fiber-stiffness product; remaining axes cycled, not full Cartesian coverage',
        unsupported='finite-deformation force equilibrium, moving particle transport, nonlinear permeability feedback on Biot')
