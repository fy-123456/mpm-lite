"""Dense tensor errors, amplitude/phase attribution, independent terminal audits."""
import json
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from benchmarks.aniso_v17_time import OUT,BASE,load,write,sources
from benchmarks.aniso_v17_modes import snapshot_model,weighted_rms
from benchmarks.aniso_compatible_controls import stiffness
from benchmarks.aniso_carrier_joint import load_case,spectrum
from engine.aniso_phase1.carrier_joint import gradient
from engine.aniso_phase1.unresolved_velocity import pack
from engine.aniso_phase1.carrier_avf import SITES


def arrays(p):
    with np.load(p) as f:return {k:f[k].copy() for k in f.files}
def power(P,V):return np.einsum('...pab,...pab,p->...',P,P,V/V.sum(),optimize=True)
def time_rms(P,V,t):return float(np.sqrt(np.trapezoid(power(P,V),t)/(t[-1]-t[0])))
def comparison(P,Q,V,t):
    den=time_rms(Q,V,t)
    return dict(relative=time_rms(P-Q,V,t)/den,absolute_Pa=time_rms(P-Q,V,t),
        terminal_relative=weighted_rms(P[-1]-Q[-1],V)/weighted_rms(Q[-1],V),
        terminal_absolute_Pa=weighted_rms(P[-1]-Q[-1],V),max_instantaneous_absolute_Pa=float(np.sqrt(power(P-Q,V)).max()))


def audit(a,name,spec,series,last):
    z=arrays(OUT/'cases'/name/'terminal.npz');e=a.e;g=a.g;dt=spec['dt'];F=gradient(e.B,z['Y']);Fn=gradient(e.B,z['Y_before'])
    out=e.evaluate(z['Y']);old=e.evaluate(z['Y_before']);v=pack(z['v'],z['C']);v0=pack(z['v_before'],z['C_before']);q=g.metric
    K=.5*np.sum(q[:,None]*v*v);K0=.5*np.sum(q[:,None]*v0*v0);dy=z['Y']-z['Y_before']
    average=sum(.5*e.evaluate(z['Y_before']+alpha*dy)['force'] for alpha in SITES)
    work=np.sum(average*dy);force_defect=float(K-K0+work);quad_error=float(out['U']-old['U']-work)
    Km,asym=stiffness(F,e.A,e.V,[sp.csr_matrix(b@g.Q) for b in e.B],200.)
    ks=g.Q.T@e.Ks@g.Q;independent=Km.toarray()+la.block_diag(ks,ks,ks);exact=e.tangent(z['Y'],g.Q)
    gate=spectrum(independent);fd=float(la.norm(exact-independent)/la.norm(exact))
    errors=dict(F=float(np.max(abs(F-z['F']))),P=float(np.max(abs(out['P']-series['P'][-1]))),
        history=float(np.max(abs(F-Fn-gradient(e.B,dy)))),frozen_x=float(np.max(abs(z['x']-a.s.x))),
        total_J=abs(float(K+out['U'])-last['total_J']),work_defect_J=abs(force_defect-last['kinetic_force_work_defect_J']),
        quadrature_error_J=abs(quad_error-last['potential_quadrature_error_J']))
    assert max(errors.values())<1e-10 and gate['passed'] and fd<1e-6,(name,errors,fd,gate)
    return dict(case=name,errors=errors,massless=gate,fd_tangent_relative=fd)


def phase_amplitude(a,series,dt):
    t=series['time'];rows=[]
    for i in a.order[:20]:
        ph=(series['modal_q'][:,i]-a.eq[i])-1j*series['modal_v'][:,i]/a.omega[i]
        init=-a.eq[i]-1j*a.v0[i]/a.omega[i]
        ratio=ph/init;amplitude=np.abs(ratio)
        resolved=bool(a.omega[i]*dt<np.pi)
        phase=np.unwrap(np.angle(ratio))-a.omega[i]*t if resolved else np.angle(ratio*np.exp(-1j*a.omega[i]*t))
        rows.append(dict(mode=int(i),omega_rad_s=float(a.omega[i]),stress_self_fraction=float(a.score[i]/a.score.sum()),phase_resolved=resolved,
            amplitude_mean_gain=float(np.mean(amplitude)),amplitude_max_relative_error=float(np.max(abs(amplitude-1))),
            terminal_phase_error_rad=float(phase[-1]),max_phase_error_rad=float(np.max(abs(phase))),
            midpoint_predicted_terminal_phase_error_rad=float((2*np.arctan(a.omega[i]*dt/2)/dt-a.omega[i])*t[-1])))
    return rows


def linear_error_attribution(a,dt):
    # 1 microsecond quadrature separates output aliasing from time stepping.
    t=np.linspace(0,.05,50001);dq=a.coordinates(t,dt)-a.coordinates(t)
    weights=np.ones(len(t));weights[[0,-1]]=.5;weights/=weights.sum()
    self_power=np.diag(a.gram)*np.sum(dq*dq*weights,axis=1);order=np.argsort(-self_power)
    full=np.einsum('it,ij,jt->t',dq,a.gram,dq,optimize=True);total=float(full@weights)
    # Compare with a second, half-density quadrature for the stress-error norm.
    coarse=float(np.trapezoid(full[::2],t[::2])/.05)
    groups=[]
    for count in (1,3,10):
        ids=order[:count];part=dq[ids];G=a.gram[np.ix_(ids,ids)]
        val=float(np.einsum('it,ij,jt->t',part,G,part,optimize=True)@weights)
        rest=dq.copy();rest[ids]=0
        remain=float(np.einsum('it,ij,jt->t',rest,a.gram,rest,optimize=True)@weights)
        groups.append(dict(top=count,group_error_RMS_Pa=float(np.sqrt(max(val,0))),remaining_error_RMS_Pa=float(np.sqrt(max(remain,0)))))
    return dict(dt=dt,error_rms_Pa=float(np.sqrt(max(total,0))),quadrature_norm_relative_difference=abs(np.sqrt(total)-np.sqrt(coarse))/max(np.sqrt(total),1e-30),
        ranking=[dict(mode=int(i),omega_rad_s=float(a.omega[i]),self_error_fraction=float(self_power[i]/self_power.sum())) for i in order[:20]],groups=groups,
        convention='Linear midpoint interpolation with its exact modified frequency; endpoint states match discrete midpoint. Actual nonlinear comparisons never use this interpolation or phase alignment.')


def main():
    assert not (OUT/'time-summary.json').exists();p=load(OUT/'protocol.json');assert sources()==p['source_sha256'];assert load(OUT/'batch.json')['completed']
    cases={};pairs={};audits=[];attribution=[];count=0
    for label,start in [('early',1.1),('late',1.6)]:
        a=snapshot_model(start);frames={}
        for name,spec in p['cases'].items():
            if spec['label']!=label:continue
            f=arrays(OUT/'cases'/name/'series.npz');frames[name]=f
            rows=[json.loads(v) for v in (OUT/'cases'/name/'steps.jsonl').read_text().splitlines()]
            assert len(rows)==round(.05/spec['dt']) and len(f['time'])==len(rows)+1
            np.testing.assert_allclose(f['P'][0],a.P0,rtol=0,atol=0)
            exact=a.stress(f['time']);linear=a.stress(f['time'],spec['dt'])
            item=dict(dt=spec['dt'],steps=len(rows),stress_outputs=len(f['time']),
                nonlinear_vs_exact_linear=comparison(f['P'],exact,a.e.V,f['time']),
                nonlinear_vs_discrete_linear=comparison(f['P'],linear,a.e.V,f['time']),
                linear_midpoint_vs_exact=comparison(linear,exact,a.e.V,f['time']),
                top_mode_phase_amplitude=phase_amplitude(a,f,spec['dt']),
                terminal_energy_change_J=load(OUT/'cases'/name/'status.json')['terminal_total_change_J'],
                max_work_defect_J=max(abs(r['kinetic_force_work_defect_J']) for r in rows),
                max_quadrature_error_J=max(abs(r['potential_quadrature_error_J']) for r in rows),
                max_history_error=max(r['history_commit_max'] for r in rows))
            probe=np.einsum('tp,p->t',f['P'][:,:,0,0],a.e.V/a.e.V.sum())
            pref=np.einsum('tp,p->t',exact[:,:,0,0],a.e.V/a.e.V.sum())
            item['mean_Pxx_range']=dict(nonlinear_Pa=float(np.ptp(probe)),exact_at_step_times_Pa=float(np.ptp(pref)),
                nonlinear_max_Pa=float(probe.max()),nonlinear_min_Pa=float(probe.min()))
            # Direct bridge to the v16 result at the original fourth dt.
            if spec['dt']==.000125:
                old=arrays(BASE/'v16/avf/cases'/f'{"early_hold" if label=="early" else "late_hold"}-frozen-avf-fourth/frames.npz')
                from benchmarks.aniso_v16_analysis import stress
                check=float(np.max(abs(f['P'][::40]-stress(old['F'],a.e.A))))
                assert check<1e-12,check;item['v16_sparse_reproduction_max_Pa']=check
            cases[name]=item;count+=len(rows);audits.append(audit(a,name,spec,f,rows[-1]));print('audit',name,flush=True)
        names=sorted(frames,key=lambda n:p['cases'][n]['dt'],reverse=True);pp=[]
        for ca,fi in zip(names[:-1],names[1:]):
            c=frames[ca];f=frames[fi];ratio=round(p['cases'][ca]['dt']/p['cases'][fi]['dt'])
            np.testing.assert_allclose(c['time'],f['time'][::ratio],rtol=0,atol=1e-14)
            r=comparison(c['P'],f['P'][::ratio],a.e.V,c['time']);r.update(coarse=ca,fine=fi,dt_coarse=p['cases'][ca]['dt'],dt_fine=p['cases'][fi]['dt'])
            sparse=round(.005/p['cases'][ca]['dt']);r['eleven_sample_RMS_relative']=weighted_rms(c['P'][::sparse]-f['P'][::ratio*sparse],a.e.V)/weighted_rms(f['P'][::ratio*sparse],a.e.V)
            r['both_interval_and_terminal_below_2percent']=max(r['relative'],r['terminal_relative'])<.02;pp.append(r)
        pairs[label]=pp
        attribution.append(dict(label=label,records=[linear_error_attribution(a,dt) for dt in (.000125,.00003125,.0000078125)]))
        print('pairs',label,pp,flush=True)
    result=dict(completed=True,formal_trajectories=len(cases),formal_steps=count,stress_outputs=count+len(cases),cases=cases,refinement=pairs,independent_audits=audits,
        time_error_mode_attribution=attribution,fixed_geometry_last_pair_accepted=all(v[-1]['both_interval_and_terminal_below_2percent'] for v in pairs.values()),
        nonlinear_truth_available=False,full_load_cycle_completed=False,spatial_accuracy_accepted=False,production_default_changed=False,
        error_norm='volume-weighted full Piola tensor, time-trapezoidal RMS at every shared actual step, no phase alignment; terminal separately normalized; 11-output legacy metric reported separately',
        reference_scope='exact continuous time for local linear same-discrete-system K/M; nonlinear linearization error reported against same-dt midpoint control')
    write(OUT/'time-summary.json',result);print('DONE',len(cases),count,flush=True)
if __name__=='__main__':main()
