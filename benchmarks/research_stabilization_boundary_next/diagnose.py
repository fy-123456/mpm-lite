"""Five-node mass-projection counterfactual, without a new trajectory."""
from pathlib import Path
import argparse,time,resource
import numpy as np
import scipy.linalg as la
from .provenance import *
from .run import load_model
from .spaces import load_selected
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.compare import regions
from engine.aniso_phase1.research_post_release.fields import CachedProbes

INDICES=[0,64,128,192,256]

def parts(m,cache,q,t):
    response=m.evaluate(q);fm=response['material_force'];fs=response['force']-fm;out={}
    for name,f in [('material',fm),('stabilization',fs)]:
        a=np.zeros_like(q);a[m.free]=la.cho_solve(m.mass_factor,-f[m.free]);out[name]=cache.maps[0]@a
    ab=m.boundary.unit*(-.005*2*np.pi**2*np.cos(2*np.pi*(t-.6)));a=ab.copy();a[m.free]=la.cho_solve(m.mass_factor,-m.M[np.ix_(m.free,m.fixed)]@ab[m.fixed]);out['boundary_cross_mass']=cache.maps[0]@a;out['total']=sum(out.values())
    r=m.reduction;s=m.parent;y=s.reference[:s.n]+r.expand(q)[:s.n];exact=r.P[:s.n].T@(s.Ks@y);U=.5*float(np.sum(y*(s.Ks@y)))
    err=float(np.max(abs(exact-fs)));ue=abs(U-response['stabilization_U'])
    if err>1e-8 or ue>1e-10:raise ValueError('stabilization coordinate chain differs')
    return out,dict(force_error_N=err,energy_error_J=ue,material_J=response['material_U'],stabilization_J=U,free_force=fs[m.free].ravel())

def run_diagnosis(run):
    run=Path(run);verify(run);begun=time.perf_counter()
    register(run,'S1/diagnostic-protocol.json',dict(status='registered',indices=INDICES,new_dynamic_steps=0,max_seconds=1200,max_modes=8,projection='existing constrained complete ambient R3 mass',no_changed_stabilization=True))
    hc=history(APP/'cases/candidate-quarter');hf=history(OBS/'cases/phase-quarter');cfg=read(APP/'cases/candidate-quarter/execution-protocol.json');fcfg=read(OBS/'cases/phase-quarter/execution-protocol.json')
    rformal,_=load_selected(fcfg['physical_space'])
    with np.load(Path(fcfg['physical_space']['path']).parent/'space.npz') as z:Tf=z['T'].copy()
    with np.load(Path(cfg['physical_space']['path']).parent/'space.npz') as z:T=z['T'].copy()
    with np.load(REFERENCE/'Q1/R3/data.npz') as z:Ma=z['M'].copy()
    formal_ambient=[(Tf@rformal.expand(hf[i]['state'].q),Tf@rformal.velocity(hf[i]['state'].velocity)) for i in INDICES]
    m,_=load_model(run,cfg);r=m.reduction;cache=CachedProbes(m);A=T@r.P[:,m.free];G=A.T@Ma@A;factor=la.cho_factor(G)
    with np.load(APP/'S1/formal-acceleration.npz') as z:af={k:z[k].copy() for k in z.files}
    with np.load(APP/'S1/candidate-acceleration.npz') as z:acold={k:z[k].copy() for k in z.files}
    rows=[];arrays={};forces=[];state_map=[];Ws=regions(cache.X)
    for j,i in enumerate(INDICES):
        c,f=hc[i]['state'],hf[i]['state'];m.validate(c)
        if abs(c.time-f.time)>1e-13:raise ValueError('time mismatch')
        q=m.boundary.lift(c.time);v=m.boundary.speed(c.time);tq,tv=formal_ambient[j]
        q[m.free]=la.cho_solve(factor,A.T@Ma@(tq-T@(r.offset+r.P@q)));v[m.free]=la.cho_solve(factor,A.T@Ma@(tv-T@(r.P@v)))
        normal={name:float(la.norm(A.T@Ma@(T@(r.expand(x) if name=='q' else r.velocity(x))-target))) for name,x,target in [('q',q,tq),('v',v,tv)]}
        if max(normal.values())>1e-9:raise ValueError('projection residual')
        actual,da=parts(m,cache,c.q,c.time);projected,dp=parts(m,cache,q,c.time);forces.append(da.pop('free_force'));dp.pop('free_force');components={};closure=0.;reproduction=0.
        for key in actual:
            hist=actual[key]-projected[key];space=projected[key]-af[key][j];total=actual[key]-af[key][j]
            closure=max(closure,float(np.max(abs(total-hist-space))));reproduction=max(reproduction,float(np.max(abs(actual[key]-acold[key][j]))))
            def rms(a,w):return float(np.sqrt(np.sum(w.reshape(-1,1)*a*a)/np.sum(w)))
            components[key]={reg:dict(history_rms=rms(hist,w),projection_rms=rms(space,w),total_rms=rms(total,w)) for reg,w in Ws.items()}
            for label,value in [('actual',actual[key]),('projected',projected[key]),('history',hist),('space',space)]:arrays.setdefault(key+'_'+label,[]).append(value)
        if closure>1e-8 or reproduction>1e-8:raise ValueError('attribution reproduction failed')
        rows.append(dict(index=i,time_s=c.time,projection_normal=normal,components=components,vector_closure_error=closure,inherited_reproduction_error=reproduction,actual_chain=da,projected_chain=dp,initial_projection_q_error=float(np.max(abs(q-c.q))) if i==0 else None))
        state_map.append(dict(index=i,time_s=c.time,candidate_state_sha256=sha(hc[i]['folder']/'state.json'),formal_state_sha256=sha(hf[i]['folder']/'state.json')))
        arrays.setdefault('projected_q',[]).append(q);arrays.setdefault('projected_velocity',[]).append(v)
        print('PROJECTION_NODE',i,components['stabilization']['global_domain'],flush=True)
        if time.perf_counter()-begun>1200:raise TimeoutError('diagnosis budget')
    lam,V=la.eigh(m.rest_K[np.ix_(m.ids,m.ids)],m.M3ff);force_coeff=np.array(forces)@V;score=np.max(abs(force_coeff),axis=0);chosen=np.argsort(score)[-8:][::-1];Ks=r.P[:m.parent.n].T@m.parent.Ks@r.P[:m.parent.n];Ks3=np.kron(Ks,np.eye(3));modes=[]
    for k in chosen:
        d=np.zeros_like(m.rest().q);d[m.free]=V[:,k].reshape(-1,3);physical=cache.maps[0]@d;stiff=float(d.ravel()@Ks3@d.ravel());reaction=float(np.sum((Ks@d)*m.boundary.unit))
        modes.append(dict(mode=int(k),period_s=float(2*np.pi/np.sqrt(lam[k])),total_rest_curvature=float(lam[k]),stabilization_curvature=stiff,material_rest_curvature=float(lam[k]-stiff),max_mass_normalized_restoring_force=float(score[k]),physical_unit_velocity_rms=float(la.norm(physical)/np.sqrt(len(physical))),stabilization_reaction_direction_N=reaction))
    np.savez_compressed(run/'S1/projection-diagnostic.npz',X=cache.X,**{k:np.array(v) for k,v in arrays.items()})
    write(run/'S1/state-map.json',dict(status='passed_scoped',records=state_map,candidate=cfg['physical_space'],formal=fcfg['physical_space'],indices=INDICES))
    write(run/'S1/projection-vs-history.json',dict(status='passed_scoped',records=rows,norms_not_additive=True,diagnostic_not_exact_causal_separation=True,new_dynamic_steps=0,seconds=time.perf_counter()-begun,host_peak_rss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20))
    write(run/'S1/stabilization-chain.json',dict(status='passed_scoped',unchanged=True,formula='y=reference+offset_carrier+P_carrier q; U=.5 tr(y.T Ks y); f=P_carrier.T Ks y',max_force_error=max(x[k]['force_error_N'] for x in rows for k in ('actual_chain','projected_chain')),max_energy_error=max(x[k]['energy_error_J'] for x in rows for k in ('actual_chain','projected_chain')),existing_derivative_check=dict(path=str(OBS/'S2/directional-energy-check.json'),sha256=sha(OBS/'S2/directional-energy-check.json'))))
    write(run/'S1/output-sensitive-directions.json',dict(status='diagnostic',records=modes,selected_from_measured_stabilization_force=True,mass_orthogonality=float(la.norm(V.T@m.M3ff@V-np.eye(len(lam)))),scope='candidate rest-linear directions only, no same-index cross-space match; reaction component is not full interval reaction'))
    write(run/'S1/diagnosis-decision.json',dict(status='passed_scoped',unexplained_numeric_anomaly=False,implementation_bug_found=False,numerical_equations_changed=False,new_dynamic_steps=0,reason='same-time constrained projection separates representation restoring acceleration from evolved-history response; coordinate chain and original derivatives agree',space_retraining_requires_short_reference=True,late_nodes_not_certified_training_targets=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):run_diagnosis(a.run)
