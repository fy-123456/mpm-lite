"""Independent time-centered update audit and comparison to phase one."""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from benchmarks.aniso_v16_avf import AVF,avf_sources
from benchmarks.aniso_v16_experiments import OUT,load,write
from benchmarks.aniso_v16_analysis import arrays,rows,metric,rms,stress,pair,LEVELS,PARAMS
from benchmarks.aniso_carrier_joint import load_case,spectrum
from benchmarks.aniso_compatible_diagnosis import gradient,maps
from benchmarks.aniso_compatible_controls import stiffness
from benchmarks.aniso_dynamic_check import pk1
from engine.aniso_phase1.diagnostics import energy_density
from engine.aniso_phase1.carrier_joint import Geometry,State
from engine.aniso_phase1.material_patch import carrier_map
from engine.aniso_phase1.unresolved_velocity import pack
from engine.aniso_phase1.history_increment import material_tangent


def energy_force(Y,B,z):
    F=gradient(B,Y);P=pk1(F,z['A'],200.)
    r=np.einsum('cij,cja->cia',z['P'],Y[z['ids']]);Us=.5*float(np.sum(z['weights'][:,None,None]*r*r))
    Um=float(z['V']@energy_density(F,z['A'],PARAMS));f=np.zeros_like(Y)
    np.add.at(f,z['ids'].ravel(),(z['weights'][:,None,None]*np.einsum('cji,cja->cia',z['P'],r)).reshape(-1,3))
    f+=sum(b.T@(z['V'][:,None]*P[:,:,k]) for k,b in enumerate(B))
    return F,Um,Us,f


def audit(name,spec,f,last):
    z=arrays(AVF/'cases'/name/'audit-terminal.npz');s,e,m,h,_=load_case(spec['start']);B=[sp.csr_matrix(b) for b in e.B]
    F,Um,Us,force=energy_force(z['Y'],B,z);F0,Um0,Us0,_=energy_force(z['Y_before'],B,z);dy=z['Y']-z['Y_before']
    fbar=sum(.5*energy_force(z['Y_before']+a*dy,B,z)[-1] for a in (.5-1/np.sqrt(12),.5+1/np.sqrt(12)))
    xx=z['x_before'] if spec['moving'] else s.x;YY=z['Y_before'] if spec['moving'] else s.Y;FF=F0 if spec['moving'] else gradient(B,s.Y)
    T,_=maps(xx,z['nodes'],h);_,_,N=carrier_map(YY,z['nodes'],h);invF=np.linalg.inv(FF)
    L=[sum(b.multiply(invF[:,k,j,None]) for k,b in enumerate(B)) for j in range(3)]
    J=np.vstack([T@z['E']]+[l.toarray() for l in L]);q=metric(xx,m,h);q1=metric(z['x'],m,h)
    zz=pack(z['v'],z['C']);zz0=pack(z['v_before'],z['C_before']);K=.5*float(np.sum(q1[:,None]*zz*zz))
    Kf=.5*float(np.sum(q[:,None]*zz*zz));K0=.5*float(np.sum(q[:,None]*zz0*zz0));work=float(np.sum(fbar*dy))
    defect=Kf-K0+work;quad=Um+Us-Um0-Us0-work
    r=(J@z['Q']).T@(q[:,None]*(zz-zz0))+spec['dt']*z['Q'].T@fbar
    er=dict(F=float(np.max(abs(F-z['F']))),material_J=abs(Um-last['material_J']),stabilization_J=abs(Us-last['stabilization_J']),kinetic_J=abs(K-last['kinetic_J']),
        work_defect_J=abs(defect-last['kinetic_force_work_defect_J']),quadrature_J=abs(quad-last['potential_quadrature_error_J']),
        J=float(np.max(abs(J-z['J']))),N=float(np.max(abs(N.toarray()-z['N']))),q=float(np.max(abs(q-z['q']))))
    assert all(v<(1e-12 if k.endswith('_J') else 1e-9) for k,v in er.items()),(name,er)
    assert la.norm(r)/spec['dt']<1e-7,(name,la.norm(r)/spec['dt'])
    ferr={k:float(np.max(abs(f[k][-1]-z[k]))) for k in ('x','Y','v','C','F')};assert max(ferr.values())==0
    current=Geometry(State(z['x'],z['Y'],z['v'],z['C']),e,m,h);Q=current.Q if spec['moving'] else z['Q']
    Km,_=stiffness(F,z['A'],z['V'],[b@sp.csr_matrix(Q) for b in B],200.);Ks=Q.T@e.Ks@Q;Kstatic=Km.toarray()+la.block_diag(Ks,Ks,Ks)
    gate=spectrum(Kstatic);exact=e.tangent(z['Y'],Q);rel=float(la.norm(Kstatic-exact)/la.norm(exact));assert gate['passed'] and rel<1e-6
    return dict(case=name,errors=er,saved_frame_errors=ferr,massless=gate,fd_exact_relative=rel,independent_force_residual_N=float(la.norm(r)/spec['dt']))


def linear_comparison(t,label,frames):
    s,e,m,h,_=load_case(t);g=Geometry(s,e,m,h);K=e.tangent(s.Y,g.Q);Mr=g.Q.T@g.M@g.Q;M=la.block_diag(Mr,Mr,Mr)
    lam,V=la.eigh(K,M);w=np.sqrt(lam);f=(g.Q.T@e.evaluate(s.Y)['force']).T.ravel();b=(g.Q.T@g.J.T@(g.metric[:,None]*pack(s.v,s.C))).T.ravel()
    eq=-(V.T@f)/lam;c=-eq-1j*(V.T@b)/w;B=[bb@g.Q for bb in e.B];F0=gradient(e.B,s.Y);P0=pk1(F0,e.A,200.)
    exact=arrays(OUT/f'modal-{label}.npz')['P_exact_linear'];records=[]
    for level,dt in zip(LEVELS,(.001,.0005,.00025,.000125)):
        values=[]
        for tau in np.arange(11)*.005:
            fac=((1+1j*w*dt/2)/(1-1j*w*dt/2))**round(tau/dt);u=(V@(eq+np.real(c*fac))).reshape(3,g.Q.shape[1]).T
            values.append(P0+material_tangent(F0,e.A,gradient(B,u),PARAMS))
        linear=np.array(values);nonlinear=stress(frames[f'{label}-frozen-avf-{level}']['F'],e.A)
        records.append(dict(level=level,dt=dt,linear_centered_vs_exact_relative=rms(linear-exact)/rms(exact),
            nonlinear_vs_linear_centered_relative=rms(nonlinear-linear)/rms(linear),nonlinear_vs_linear_exact_relative=rms(nonlinear-exact)/rms(exact)))
    return dict(time=t,records=records,scope='Fixed geometry only; exact-time solution of the same locally linearized M/K system, not a continuum or nonlinear reference.')


def main():
    assert not (AVF/'summary.json').exists();p=load(AVF/'protocol.json');assert avf_sources()==p['source_sha256'];assert load(AVF/'batch.json')['completed']
    cases={};frames={};audits=[]
    for name,spec in p['cases'].items():
        f=arrays(AVF/'cases'/name/'frames.npz');rr=rows(AVF/'cases'/name/'steps.jsonl');frames[name]=f;s,e,m,h,_=load_case(spec['start'])
        assert all(np.array_equal(f[k][0],getattr(s,k)) for k in ('x','Y','v','C'))
        audits.append(audit(name,spec,f,rr[-1]));cases[name]=dict(steps=len(rr),terminal=rr[-1],
            max_work_defect_J=max(abs(r['kinetic_force_work_defect_J']) for r in rr),sum_work_defect_J=sum(r['kinetic_force_work_defect_J'] for r in rr),
            max_quadrature_error_J=max(abs(r['potential_quadrature_error_J']) for r in rr),sum_quadrature_error_J=sum(r['potential_quadrature_error_J'] for r in rr),
            sum_metric_change_J=sum(r['metric_change_J'] for r in rr),sum_total_change_J=sum(r['delta_total_J'] for r in rr),
            max_history_commit=max(r['history_commit_max'] for r in rr),min_det_F=min(r['min_det_F'] for r in rr),
            max_positive_total_change_J=max(0.,max(r['delta_total_J'] for r in rr)))
    refinement={};linear=[]
    for label,t in [('early_hold',1.1),('late_hold',1.6)]:
        _,e,_,_,_=load_case(t)
        for geom in ('frozen','moving'):
            key=f'{label}-{geom}-avf';refinement[key]=[]
            for a,b in zip(LEVELS[:-1],LEVELS[1:]):
                r=pair(frames[key+'-'+a],frames[key+'-'+b],e.A);r['pair']=[a,b];refinement[key].append(r)
        linear.append(linear_comparison(t,label,frames))
    result=dict(completed=True,formal_trajectories=len(cases),formal_steps=sum(r['steps'] for r in cases.values()),cases=cases,independent_snapshots=audits,refinement=refinement,
        exact_linear_controls=linear,default_changed=False,overall_accuracy_accepted=False,full_load_cycle_completed=False,spatial_accuracy_revalidated=False)
    write(AVF/'summary.json',result)
    print('AVF AUDITED',len(cases),result['formal_steps'],flush=True)
    for name,values in refinement.items():print(name,values[-1],flush=True)
    for r in linear:print('LINEAR',r,flush=True)

if __name__=='__main__':main()
