"""Independent v16 snapshot audit, time comparison, and exact linear modal control."""
import json
from pathlib import Path
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from benchmarks.aniso_v16_experiments import OUT,BASE,ROOT,load,write,sources
from benchmarks.aniso_carrier_joint import load_case,spectrum
from benchmarks.aniso_compatible_diagnosis import maps,gradient
from benchmarks.aniso_compatible_controls import stiffness
from benchmarks.aniso_dynamic_check import pk1
from engine.aniso_phase1.material_patch import carrier_map
from engine.aniso_phase1.selective_patch import polynomial
from engine.aniso_phase1.diagnostics import energy_density
from engine.aniso_phase1.types import AnisotropicMaterialParams
from engine.aniso_phase1.carrier_joint import Geometry,State
from engine.aniso_phase1.unresolved_velocity import pack
from engine.aniso_phase1.history_increment import material_tangent
PARAMS=AnisotropicMaterialParams(10.,20.,200.)
LEVELS=('coarse','fine','finest','fourth')


def arrays(path):
    with np.load(path) as f:return {k:np.array(f[k],copy=True) for k in f.files}
def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]
def rms(a):return float(np.sqrt(np.mean(np.sum(a*a,axis=(-2,-1)))))
def stress(F,A):return pk1(F.reshape(-1,3,3),np.tile(A,(len(F),1,1)),200.).reshape(F.shape)
def metric(x,m,h):
    f=x/h-.5;f-=np.floor(f);D=h*h*(f*(1-f)+.25)
    return np.concatenate([m]+[m*D[:,k] for k in range(3)])


def audit(name,spec,frames,logged):
    z=arrays(OUT/'cases'/name/'audit-terminal.npz');s,e,m,h,_=load_case(spec['start'])
    B=[sp.csr_matrix(b) for b in e.B];F=gradient(B,z['Y']);F0=gradient(B,z['Y_before'])
    res=np.einsum('cij,cja->cia',z['P'],z['Y'][z['ids']]);Us=.5*float(np.sum(z['weights'][:,None,None]*res*res))
    Um=float(z['V']@energy_density(F,z['A'],PARAMS));Pm=pk1(F,z['A'],200.)
    cf=np.zeros_like(z['Y']);np.add.at(cf,z['ids'].ravel(),(z['weights'][:,None,None]*np.einsum('cji,cja->cia',z['P'],res)).reshape(-1,3))
    force=cf+sum(b.T@(z['V'][:,None]*Pm[:,:,k]) for k,b in enumerate(B))
    xg=z['x_before'] if spec['moving'] else s.x;Yg=z['Y_before'] if spec['moving'] else s.Y
    Fg=F0 if spec['moving'] else gradient(B,s.Y)
    T,G=maps(xg,z['nodes'],h);_,_,N=carrier_map(Yg,z['nodes'],h)
    L=[sum(b.multiply(np.linalg.inv(Fg)[:,k,j,None]) for k,b in enumerate(B)) for j in range(3)]
    J=np.vstack([(T@z['E'])]+[l.toarray() for l in L]);Jold=np.vstack([(T@z['E'])]+[(g@z['E']) for g in G])
    q=metric(xg,m,h);q1=metric(z['x'],m,h);zz=pack(z['v'],z['C']);zz0=pack(z['v_before'],z['C_before']);dz=zz-zz0
    K=.5*float(np.sum(q1[:,None]*zz*zz));Kf=.5*float(np.sum(q[:,None]*zz*zz));K0=.5*float(np.sum(q[:,None]*zz0*zz0))
    dy=z['Y']-z['Y_before'];work=float(np.sum(force*dy));defect=Kf-K0+.5*float(np.sum(q[:,None]*dz*dz))+work
    er=dict(F=float(np.max(abs(F-z['F']))),material_J=abs(Um-logged['material_J']),stabilization_J=abs(Us-logged['stabilization_J']),
        kinetic_J=abs(K-logged['kinetic_J']),work_defect_J=abs(defect-logged['kinetic_force_work_defect_J']),
        N=float(np.max(abs(N.toarray()-z['N']))),J=float(np.max(abs(J-z['J']))),Jold=float(np.max(abs(Jold-z['Jold']))),q=float(np.max(abs(q-z['q']))),
        right_inverse=float(np.max(abs(z['N']@z['E']-np.eye(len(z['Y']))))))
    for k,v in er.items():assert v<(1e-12 if k.endswith('_J') else 1e-9),(name,k,v)
    ferr={k:float(np.max(abs(z[k]-frames[k][-1]))) for k in ('x','Y','v','C','F')};assert max(ferr.values())==0
    # A newly rebuilt map at the terminal moving state, not merely the last
    # accepted map; no inertia in this independent FD stiffness construction.
    state=State(z['x'],z['Y'],z['v'],z['C']);current=Geometry(state,e,m,h)
    Q=current.Q if spec['moving'] else z['Q'];Br=[b@sp.csr_matrix(Q) for b in B]
    Km,asym=stiffness(F,z['A'],z['V'],Br,200.);Ks=Q.T@e.Ks@Q
    Kstatic=Km.toarray()+la.block_diag(Ks,Ks,Ks);gate=spectrum(Kstatic)
    exact=e.tangent(z['Y'],Q);fd_error=float(la.norm(Kstatic-exact)/la.norm(exact))
    assert gate['passed'] and fd_error<1e-6,(name,gate,fd_error)
    return dict(case=name,errors=er,saved_frame_errors=ferr,massless=gate,fd_exact_relative=fd_error,material_fd_asymmetry=asym)


def pair(a,b,A):
    assert np.max(abs(a['time']-b['time']))<1e-10
    pa,pb=stress(a['F'],A),stress(b['F'],A);absolute=rms(pa-pb)
    return dict(P_relative=absolute/rms(pb),P_absolute_Pa=absolute,
        P_terminal_relative=rms(pa[-1]-pb[-1])/rms(pb[-1]),P_terminal_absolute_Pa=rms(pa[-1]-pb[-1]))


def modal_control(t,label,frames):
    s,e,m,h,_=load_case(t);g=Geometry(s,e,m,h);K=e.tangent(s.Y,g.Q)
    f=(g.Q.T@e.evaluate(s.Y)['force']).T.ravel();z=pack(s.v,s.C);records=[]
    Mr=g.Q.T@g.M@g.Q;b=(g.Q.T@g.J.T@(g.metric[:,None]*z)).T.ravel()
    mass=la.block_diag(Mr,Mr,Mr);lam,V=la.eigh(K,mass);assert lam.min()>0
    omega=np.sqrt(lam);fm=V.T@f;eq=-fm/lam;v0=V.T@b;c0=-eq-1j*v0/omega
    # Linearized Piola response to each material coordinate, independently of
    # the nonlinear trajectory. U and kinetics otherwise use the same maps.
    Bt=[bb@g.Q for bb in e.B];P0=pk1(gradient(e.B,s.Y),e.A,200.);F0=gradient(e.B,s.Y)
    def modal_stress(tau,dt=None):
        result=[]
        for time in tau:
            factor=np.exp(1j*omega*time) if dt is None else (1-1j*omega*dt)**(-round(time/dt))
            qmode=eq+np.real(c0*factor);du=(V@qmode).reshape(3,g.Q.shape[1]).T;dF=gradient(Bt,du)
            result.append(P0+material_tangent(F0,e.A,dF,PARAMS))
        return np.array(result)
    tau=np.arange(11)*.005;exactP=modal_stress(tau)
    for level,dt in zip(LEVELS,(.001,.0005,.00025,.000125)):
        linearP=modal_stress(tau,dt);nonlinear=stress(frames[f'{label}-frozen-joint-{level}']['F'],e.A)
        records.append(dict(level=level,dt=dt,linear_BE_vs_exact_relative=rms(linearP-exactP)/rms(exactP),
            nonlinear_vs_linear_BE_relative=rms(nonlinear-linearP)/rms(linearP),
            nonlinear_vs_linear_exact_relative=rms(nonlinear-exactP)/rms(exactP)))
    spectra=[]
    for kind,M in [('grid_lumped',g.Mgrid),('particle_joint',g.M)]:
        mm=g.Q.T@M@g.Q;M3=la.block_diag(mm,mm,mm);lam,U=la.eigh(K,M3);w=np.sqrt(lam);release=.5*(U.T@f)**2/lam
        r=dict(metric=kind,min_frequency_rad_s=float(w.min()),max_frequency_rad_s=float(w.max()),
            fastest_period_s=float(2*np.pi/w.max()),min_free_mass_eigenvalue=float(la.eigvalsh(mm)[0]),
            local_force_release_J=float(release.sum()),release_fraction_omega_dt_gt1=float(release[w*.000125>1].sum()/release.sum()),
            release_fraction_BE_exponent_gt1=float(release[w*w*.000125*.05/2>1].sum()/release.sum()))
        spectra.append(r)
    np.savez_compressed(OUT/f'modal-{label}.npz',relative_time=tau,P_exact_linear=exactP,omega=omega,
        generalized_eigen_residual=np.array([la.norm(K@V-(mass@V)*np.square(omega))/la.norm(K@V)]))
    return dict(time=t,spectra=spectra,time_comparison=records,
        scope='Local linearization, same M/K/F/Y and input velocities, exact modal time evolution. It is not a nonlinear or spatial reference solution; fastest mode alone is not an attribution of all stress error.')


def main():
    assert not (OUT/'summary.json').exists();p=load(OUT/'protocol.json');assert sources()==p['source_sha256'];assert load(OUT/'batch.json')['completed']
    frames={};cases={};audits=[];initial_checks=[]
    for name,spec in p['cases'].items():
        f=arrays(OUT/'cases'/name/'frames.npz');rr=rows(OUT/'cases'/name/'steps.jsonl');frames[name]=f
        assert len(rr)==round(spec['duration']/spec['dt']);s,e,m,h,_=load_case(spec['start'])
        errors={k:float(np.max(abs(f[k][0]-getattr(s,k)))) for k in ('x','Y','v','C')};assert max(errors.values())==0
        initial_checks.append(dict(case=name,errors=errors))
        audit_record=audit(name,spec,f,rr[-1]);audits.append(audit_record)
        cases[name]=dict(steps=len(rr),terminal=rr[-1],max_work_defect_J=max(abs(r['kinetic_force_work_defect_J']) for r in rr),
            sum_work_defect_J=sum(r['kinetic_force_work_defect_J'] for r in rr),
            sum_kinetic_increment_J=sum(r['kinetic_increment_norm_J'] for r in rr),sum_potential_BE_loss_J=sum(r['potential_backward_euler_loss_J'] for r in rr),
            sum_metric_change_J=sum(r['metric_change_J'] for r in rr),max_positive_total_change_J=max(0.,max(r['delta_total_J'] for r in rr)),
            max_history_commit=max(r['history_commit_max'] for r in rr),min_det_F=min(r['min_det_F'] for r in rr),
            actual_steps_with_support_expansion=sum(r['grid_nodes']>r['carriers'] for r in rr))
    refinement={}
    for label,t in [('early_hold',1.1),('late_hold',1.6)]:
        _,e,_,_,_=load_case(t)
        for geom in ('frozen','moving'):
            for mode in ('legacy_split','common_split','joint'):
                key=f'{label}-{geom}-{mode}';refinement[key]=[]
                for a,b in zip(LEVELS[:-1],LEVELS[1:]):
                    r=pair(frames[key+'-'+a],frames[key+'-'+b],e.A);r['pair']=[a,b];refinement[key].append(r)
    modal=[modal_control(t,label,frames) for label,t in [('early_hold',1.1),('late_hold',1.6)]]
    result=dict(completed=True,formal_trajectories=len(cases),formal_steps=sum(c['steps'] for c in cases.values()),
        cases=cases,initial_checks=initial_checks,independent_snapshots=audits,refinement=refinement,modal_controls=modal,
        default_changed=False,overall_accuracy_accepted=False,full_load_cycle_completed=False,spatial_accuracy_revalidated=False,
        scope=p['scope'],velocity_controls=p['velocity_controls'])
    write(OUT/'summary.json',result)
    print('AUDITED',result['formal_trajectories'],result['formal_steps'],len(audits),flush=True)
    for key,value in refinement.items():print(key,value[-1],flush=True)
    for r in modal:print('MODAL',r,flush=True)

if __name__=='__main__':main()
