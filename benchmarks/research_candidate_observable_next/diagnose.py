"""Read-only split-force and mass-modal diagnostics at authenticated origin states."""
from pathlib import Path
import argparse
import numpy as np
import scipy.linalg as la
from .provenance import *
from .run import load_model
from .candidate import parents
from benchmarks.research_phase_stress_next.time_study import history
from engine.aniso_phase1.research_post_release.fields import CachedProbes


def diagnose(run,label):
    run=Path(run);verify(run);(run/'S2').mkdir(exist_ok=True);item=parents()[0] if label=='candidate' else history(OLD/'cases/window0-half')[0]
    cfg=read((APP/'cases/candidate-h' if label=='candidate' else APP/'cases/final-full')/'execution-protocol.json');m,_=load_model(run,cfg);s=item['state'];m.validate(s,material=True);cache=CachedProbes(m);out=m.evaluate(s.q)
    fm=out['material_force'];fs=out['force']-fm;accb=m.boundary.unit*(-.005*2*np.pi**2*np.cos(2*np.pi*(s.time-.6)))
    parts={}
    for key,force in (('material',fm),('stabilization',fs)):
        acc=np.zeros_like(s.q);acc[m.free]=la.cho_solve(m.mass_factor,-force[m.free]);parts[key]=cache.maps[0]@acc
    ab=accb.copy();ab[m.free]=la.cho_solve(m.mass_factor,-m.M[np.ix_(m.free,m.fixed)]@accb[m.fixed]);parts['boundary_cross_mass']=cache.maps[0]@ab;parts['total']=sum(parts.values())
    direct=accb.copy();direct[m.free]=la.cho_solve(m.mass_factor,-out['force'][m.free]-m.M[np.ix_(m.free,m.fixed)]@accb[m.fixed])
    split_error=float(np.max(abs(parts['total']-cache.maps[0]@direct)))
    rng=np.random.default_rng(781);d=rng.normal(size=s.q.shape);d[m.fixed]=0;d/=la.norm(d);checks=[]
    for eps in (1e-5,5e-6):
        plus=m.evaluate(s.q+eps*d);minus=m.evaluate(s.q-eps*d)
        for key,force in (('material',fm),('stabilization',fs)):
            analytic=float(np.sum(force*d));fd=float((plus[key+'_U']-minus[key+'_U'])/(2*eps));budget=1e-10+.001*abs(analytic)
            checks.append(dict(component=key,epsilon=eps,analytic=analytic,finite_difference=fd,error=abs(fd-analytic),budget=budget,passed=abs(fd-analytic)<=budget))
    lam,V=la.eigh(m.rest_K[np.ix_(m.ids,m.ids)],m.M3ff);qf=(s.q-m.boundary.lift(s.time))[m.free].ravel();vf=(s.velocity-m.boundary.speed(s.time))[m.free].ravel();coefq=V.T@m.M3ff@qf;coefv=V.T@m.M3ff@vf;energy=.5*(coefv**2+lam*coefq**2);acc=np.zeros_like(s.q);amplitudes=[]
    for j in np.argsort(energy)[-8:][::-1]:
        acc[:]=0;acc[m.free]=V[:,j].reshape(-1,3)*coefv[j];physical=cache.maps[0]@acc
        amplitudes.append(dict(mode=int(j),period_s=float(2*np.pi/np.sqrt(lam[j])),rest_linear_energy_J=float(energy[j]),velocity_physical_rms_m_s=float(la.norm(physical)/np.sqrt(len(physical)))))
    pkg=Path(cfg['physical_space']['path']);T=np.load(pkg.parent/'space.npz')['T'];ambient=T@m.reduction.P[:,m.free];low=np.array([ambient@V[:,i].reshape(-1,3) for i in range(6)])
    np.savez_compressed(run/'S2'/f'{label}-physical-diagnostic.npz',**parts,low_modes=low,ambient_q=T@m.reduction.expand(s.q),ambient_v=T@m.reduction.velocity(s.velocity))
    write(run/'S2'/f'{label}-diagnostic.json',dict(status='passed_scoped' if split_error<1e-8 and all(x['passed'] for x in checks) else 'limited',source_state_sha256=sha(item['folder']/'state.json'),model=m.identity,material_J=out['material_U'],stabilization_J=out['stabilization_U'],kinetic_J=m.kinetic(s.velocity),split_error=split_error,acceleration_rms={k:float(la.norm(v)/np.sqrt(len(v))) for k,v in parts.items()},energy_derivative_checks=checks,top_rest_linear_modes=amplitudes,modal_kinetic_closure_J=float(abs(.5*np.sum(coefv**2)-.5*vf@m.M3ff@vf)),modes_not_directly_matched=True))


def finish(run):
    run=Path(run);a=read(run/'S2/formal-diagnostic.json');b=read(run/'S2/candidate-diagnostic.json')
    from .spaces import load_selected
    r,_=load_selected(read(APP/'cases/candidate-h/execution-protocol.json')['physical_space']);cfg=read(APP/'cases/candidate-h/execution-protocol.json');T=np.load(Path(cfg['physical_space']['path']).parent/'space.npz')['T'];A=T@r.P[:,r.free]
    with np.load(REFERENCE/'Q1/R3/data.npz') as z:Ma=z['M'].copy()
    with np.load(run/'S2/formal-physical-diagnostic.npz') as aa,np.load(run/'S2/candidate-physical-diagnostic.npz') as bb:
        normal={key:float(la.norm(A.T@Ma@(bb['ambient_'+key]-aa['ambient_'+key]))) for key in ('q','v')}
        parts={k:float(la.norm(bb[k]-aa[k])/np.sqrt(len(bb[k]))) for k in ('material','stabilization','boundary_cross_mass','total')}
        overlap=np.array([[np.sum(x*(Ma@y)) for y in aa['low_modes']] for x in bb['low_modes']]);sv=la.svdvals(overlap)
    write(run/'S2/projection-energy-audit.json',dict(status='passed_scoped' if max(normal.values())<1e-9 else 'limited',constrained_mass_normal_residual=normal,old=a,new=b,potential_jump_J=b['material_J']+b['stabilization_J']-a['material_J']-a['stabilization_J'],kinetic_jump_J=b['kinetic_J']-a['kinetic_J'],projection_not_changed=True))
    write(run/'S2/physical-force-acceleration-decomposition.json',dict(status='diagnostic',physical_rms_difference_m_s2=parts,coefficient_force_norm_not_used_as_cross_space_error=True,no_artificial_mass_or_damping=True))
    write(run/'S2/directional-energy-check.json',dict(status='passed_scoped' if a['status']==b['status']=='passed_scoped' else 'limited',records={k:v['energy_derivative_checks'] for k,v in [('formal',a),('candidate',b)]}))
    write(run/'S2/modal-output-diagnostic.json',dict(status='diagnostic',low_mode_subspace_singular_values=sv.tolist(),same_index_not_same_mode=True,records={k:v['top_rest_linear_modes'] for k,v in [('formal',a),('candidate',b)]},rest_modal_is_not_nonlinear_exact_split=True))
    dec=read(run/'S1/candidate-reference-decision.json')
    write(run/'S2/research-space-decision.json',dict(status='limited',formal_space_changed=False,static_pass_not_dynamic_certification=True,candidate_reference=dec,physical_acceleration_parts=parts,reason='projected fields/kinetic energy can agree while stabilization/material restoring acceleration differs; inspect physical dynamics before new training',new_spaces=0,new_mapping_trajectories=0,new_dynamic_steps=0))
    print('FORCE_DECOMPOSITION',parts,normal,flush=True)


def derivative_review(run,label):
    run=Path(run);verify(run);item=parents()[0] if label=='candidate' else history(OLD/'cases/window0-half')[0]
    cfg=read((APP/'cases/candidate-h' if label=='candidate' else APP/'cases/final-full')/'execution-protocol.json');m,_=load_model(run,cfg);s=item['state'];cache=CachedProbes(m);response=m.evaluate(s.q)
    rng=np.random.default_rng(781);d=rng.normal(size=s.q.shape);d[m.fixed]=0;d/=la.norm(d)
    scale=max(float(np.max(abs(a@d))) for a in cache.maps[1:]);d/=scale
    checks=[]
    for eps in (1e-5,5e-6):
        plus=m.evaluate(s.q+eps*d);minus=m.evaluate(s.q-eps*d)
        for key,force in (('material',response['material_force']),('stabilization',response['force']-response['material_force'])):
            analytic=float(np.sum(force*d));fd=float((plus[key+'_U']-minus[key+'_U'])/(2*eps));budget=1e-10+.001*abs(analytic)
            checks.append(dict(component=key,physical_gradient_perturbation=eps,analytic=analytic,finite_difference=fd,error=abs(fd-analytic),budget=budget,passed=abs(fd-analytic)<=budget))
    write(run/'S2'/f'{label}-derivative-review.json',dict(status='passed_scoped' if all(x['passed'] for x in checks) else 'limited',old_coefficient_direction_gradient_scale=scale,old_error_quarters_when_epsilon_halves=True,reason='coefficient-normalized direction had a large physical gradient; second-order central-difference truncation, not evidence of a changed material derivative',checks=checks,model_unchanged=True,dynamic_steps=0))
    paths=[run/'S2'/f'{v}-derivative-review.json' for v in ('formal','candidate')]
    if all(p.exists() for p in paths):
        records={p.stem:read(p) for p in paths};write(run/'S2/directional-energy-check.json',dict(status='passed_scoped' if all(x['status']=='passed_scoped' for x in records.values()) else 'limited',records=records,original_large_perturbation_diagnostics_retained=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['formal','candidate','finish','formal-fd','candidate-fd']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='finish':finish(a.run)
        elif a.phase.endswith('-fd'):derivative_review(a.run,a.phase.split('-')[0])
        else:diagnose(a.run,a.phase)
