"""Independent physical audit and unsmoothed time-convergence measurements."""
import json
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from benchmarks.aniso_v19_runs import OUT,BASE,ROOT,load,write,sha
from benchmarks.aniso_v17_modes import controlled_case
from benchmarks.aniso_v17_analysis import arrays,comparison
from benchmarks.aniso_dynamic_check import pk1
from engine.aniso_phase1.diagnostics import energy_density
from engine.aniso_phase1.carrier_joint import Geometry,State,gradient
from engine.aniso_phase1.unresolved_velocity import pack
from benchmarks.aniso_compatible_controls import stiffness
from benchmarks.aniso_carrier_joint import spectrum

STAGES=dict(ramp=(0.,.5),hold=(.5,.6),unload=(.6,1.1),final_hold=(1.1,1.6))
def readrows(path):return [json.loads(v) for v in path.read_text().splitlines()]
def rms(v):return float(np.sqrt(np.mean(np.asarray(v)**2)))
def frequency(r):
    y=np.array([v['reaction_N'] for v in r if v['time']>1.1+1e-10]);y-=y.mean();f=np.fft.rfft(y);power=abs(f)**2;power[1:]*=2
    if len(y)%2==0:power[-1]/=2
    freq=np.fft.rfftfreq(len(y));return dict(above_80pct_Nyquist_fraction=float(power[freq>.4].sum()/power.sum()),adjacent_correlation=float(np.corrcoef(y[1:],y[:-1])[0,1]),oscillatory_rms_N=rms(y))

def audit(path,row,stress):
    z=arrays(path);s,e,m,h,_=controlled_case();old=State(z['x_before'],z['Y_before'],z['v_before'],z['C_before']);new=State(z['x'],z['Y'],z['v'],z['C']);g=Geometry(old,e,m,h);end=Geometry(new,e,m,h);dt=row['dt'];dy=new.Y-old.Y
    def elastic(Y):
        F=gradient(e.B,Y);P=pk1(F,e.A,200.);Um=float(e.V@energy_density(F,e.A,e.params));r=e.P@Y[e.ids];Us=.5*float(np.sum(e.weights[:,None,None]*r*r));f=np.zeros_like(Y)
        np.add.at(f,e.ids.ravel(),(e.weights[:,None,None]*(e.P.swapaxes(1,2)@r)).reshape(-1,3));f+=sum(b.T@(e.V[:,None]*P[:,:,j]) for j,b in enumerate(e.B));return F,P,Um,Us,f
    F,P,Um,Us,_=elastic(new.Y);Fn,Pn,Umn,Usn,_=elastic(old.Y);fbar=sum(.5*elastic(old.Y+a*dy)[-1] for a in (.5-np.sqrt(3)/6,.5+np.sqrt(3)/6));z0=pack(old.v,old.C);z1=pack(new.v,new.C);q=g.metric;q1=end.metric;W=z['W']
    def project(J,metric,zv):
        U,sv,_=la.svd(np.sqrt(metric)[:,None]*J,full_matrices=False);U=U[:,sv>1e-12*sv[0]];return U@(U.T@(np.sqrt(metric)[:,None]*zv))/np.sqrt(metric)[:,None]
    def lift(geo):
        fixed=(geo.nodes[:,0]*h<=.25)|(geo.nodes[:,0]*h>=.75);grid=np.zeros((len(geo.nodes),3));grid[geo.nodes[:,0]*h>=.75,0]=1.;return la.lstsq(geo.E[fixed],grid[fixed],cond=1e-11)[0]
    L=lift(g);Le=lift(end);trial=z0+2*(g.J@W-project(g.J,q,z0));target=end.J@Le*row['endpoint_speed'];expected=trial-project(end.J,q1,trial)+project(end.J@end.Q,q1,trial-target)+target;delta=z1-trial
    i0=2*g.J.T@(q[:,None]*(g.J@W-z0))+dt*fbar;i1=end.J.T@(q1[:,None]*delta);R=float((np.sum(i0*L)+np.sum(i1*Le))/dt);work=float(np.sum(i0*L)*row['loading_speed']+np.sum(i1*Le)*row['endpoint_speed']);loss=.5*float(np.sum(q1[:,None]*delta*delta))
    K=lambda q,z:.5*float(np.sum(q[:,None]*z*z));K0=K(q,z0);K1=K(q1,z1);metric=K(q1,trial)-K(q,trial);forcework=float(np.sum(fbar*dy));quad=Um+Us-Umn-Usn-forcework;solve=K(q,trial)-K0+forcework-np.sum(i0*L)*row['loading_speed']
    errors=dict(F=float(np.max(abs(F-z['F']))),P=float(np.max(abs(P-stress))),history=float(np.max(abs(F-Fn-gradient(e.B,dy)))),position=float(np.max(abs(new.x-old.x-dt*g.T@W))),velocity=float(np.max(abs(expected-z1))),
        material_J=abs(Um-row['material_J']),stabilization_J=abs(Us-row['stabilization_J']),kinetic_J=abs(K1-row['kinetic_J']),reaction_N=abs(R-row['reaction_N']),boundary_work_J=abs(work-row['boundary_work_J']),constraint_loss_J=abs(loss-row['constraint_kinetic_loss_J']),metric_J=abs(metric-row['metric_change_J']),
        solve_J=abs(solve-row['kinetic_force_work_defect_J']),quad_J=abs(quad-row['potential_quadrature_error_J']),budget_J=abs(K1-K0+Um+Us-Umn-Usn-work-metric-solve-quad+loss),
        free_impulse=float(la.norm(end.Q.T@i1)),linear_impulse=float(la.norm(i1.sum(0)-m@delta[:len(m)])))
    angular=np.sum(np.cross(new.x,m[:,None]*delta[:len(m)]),axis=0)
    for j in range(3):angular+=np.sum(np.cross(np.eye(3)[j],q1[(j+1)*len(m):(j+2)*len(m),None]*delta[(j+1)*len(m):(j+2)*len(m)]),axis=0)
    errors['angular_impulse']=float(la.norm(np.sum(np.cross(new.Y,i1),axis=0)-angular))
    Km,_=stiffness(F,e.A,e.V,[sp.csr_matrix(b@end.Q) for b in e.B],200.);Ks=end.Q.T@e.Ks@end.Q;Kstatic=Km.toarray()+la.block_diag(Ks,Ks,Ks);gate=spectrum(Kstatic);rel=float(la.norm(Kstatic-e.tangent(new.Y,end.Q))/la.norm(Kstatic))
    assert max(errors.values())<1e-7 and max(v for k,v in errors.items() if k.endswith('_J'))<1e-12,(path,errors)
    assert gate['passed'] and rel<1e-6
    return dict(snapshot=str(path.relative_to(OUT)),errors=errors,static_gate=gate,independent_tangent_relative=rel)


def audit_case(folder,rows,frames):
    paths=sorted(folder.glob('audit-*.npz'));digest={v.name:sha(v) for v in paths+[folder/'steps.jsonl',folder/'stress.npz']};cache=OUT/f'audits-{folder.name}.json'
    if cache.exists():
        data=load(cache);assert data['input_sha256']==digest;return data['records']
    records=[]
    for path in paths:
        k=int(path.stem.split('-')[-1]);records.append(audit(path,rows[k-1],frames['P'][k]))
    write(cache,dict(input_sha256=digest,records=records));return records

def cycle():
    assert load(OUT/'cycle-batch.json')['completed'];p=load(OUT/'cycle-protocol.json');_,e,_,_,_=controlled_case();frames={};rows={};cases={};audits=[]
    for level,dt in enumerate(p['dt']):
        name=f'cycle-L{level}';folder=OUT/'cases'/name;r=readrows(folder/'steps.jsonl');f=arrays(folder/'stress.npz');rows[name]=r;frames[name]=f
        assert len(r)==round(1.6/dt)
        sums={k:sum(v[k] for v in r) for k in ('boundary_work_J','constraint_kinetic_loss_J','metric_change_J','kinetic_force_work_defect_J','potential_quadrature_error_J')}
        balance=r[-1]['total_J']-sums['boundary_work_J']+sums['constraint_kinetic_loss_J']-sums['metric_change_J']-sums['kinetic_force_work_defect_J']-sums['potential_quadrature_error_J']
        assert abs(balance)<1e-12
        entry=dict(steps=len(r),dt=dt,terminal=r[-1],energy_sums_J=sums,budget_closure_J=balance,loss_over_peak_energy=sums['constraint_kinetic_loss_J']/max(v['total_J'] for v in r),frequency=frequency(r),
            maxima={k:max(abs(v[k]) for v in r) for k in ('budget_defect_J','history_commit_max','endpoint_velocity_constraint','grip_velocity_error','endpoint_free_impulse_error','kinetic_force_work_defect_J','potential_quadrature_error_J')},stages={})
        for stage,(lo,hi) in STAGES.items():
            a,b=round(lo/dt),round(hi/dt);part=r[a:b];entry['stages'][stage]=dict(energy_change_J=part[-1]['total_J']-(r[a-1]['total_J'] if a else 0.),**{k:sum(v[k] for v in part) for k in sums})
        audits.extend(audit_case(folder,r,f))
        cases[name]=entry;print('audited',name,flush=True)
    pairs=[]
    for level in range(3):
        a=f'cycle-L{level}';b=f'cycle-L{level+1}';fa,fb=frames[a],frames[b];r=comparison(fa['P'],fb['P'][::2],e.V,fa['time']);r.update(coarse=a,fine=b)
        r['stages']={stage:comparison(fa['P'][(fa['time']>=lo-1e-12)&(fa['time']<=hi+1e-12)],fb['P'][::2][(fa['time']>=lo-1e-12)&(fa['time']<=hi+1e-12)],e.V,fa['time'][(fa['time']>=lo-1e-12)&(fa['time']<=hi+1e-12)]) for stage,(lo,hi) in STAGES.items()}
        ra=np.array([x['reaction_N'] for x in rows[a]]);rb=np.array([x['reaction_N'] for x in rows[b]]);r['raw_reaction_relative']=rms(ra-rb[1::2])/rms(rb[1::2]);r['same_interval_impulse_relative']=rms(ra-rb.reshape(-1,2).mean(1))/rms(rb.reshape(-1,2).mean(1))
        r['reaction_stages']={}
        for stage,(lo,hi) in STAGES.items():
            aa,bb=round(lo/p['dt'][level]),round(hi/p['dt'][level]);rc=ra[aa:bb];rf=rb[2*aa:2*bb]
            r['reaction_stages'][stage]=dict(raw_absolute_N=rms(rc-rf[1::2]),reference_rms_N=rms(rf[1::2]),raw_relative=rms(rc-rf[1::2])/rms(rf[1::2]),same_interval_relative=rms(rc-rf.reshape(-1,2).mean(1))/rms(rf.reshape(-1,2).mean(1)))
        pairs.append(r)
    bridges={}
    for level,oldlevel in ((1,0),(2,1),(3,2)):
        name=f'cycle-L{level}';oldpath=BASE/'v18/cases'/f'cycle-L{oldlevel}';old=arrays(oldpath/'stress.npz');bridges[name]=dict(stress=comparison(frames[name]['P'],old['P'],e.V,old['time']),old_frequency=frequency(readrows(oldpath/'steps.jsonl')))
    last=pairs[-1];stress_pass=all(max(v['relative'],v['terminal_relative'])<.02 for v in last['stages'].values()) and max(last['relative'],last['terminal_relative'])<.02;reaction_pass=max(last['raw_reaction_relative'],last['same_interval_impulse_relative'])<.02
    loss=[v['energy_sums_J']['constraint_kinetic_loss_J'] for v in cases.values()];loss_decreases=all(a>b for a,b in zip(loss[:-1],loss[1:]))
    write(OUT/'cycle-acceptance.json',dict(completed=True,cases=cases,pairs=pairs,audits=audits,same_dt_old_bridges=bridges,stress_time_passed=stress_pass,raw_reaction_time_passed=reaction_pass,constraint_loss_decreases=loss_decreases,passed=stress_pass and reaction_pass and loss_decreases))


def short():
    batch=load(OUT/'short-moving-summary.json');assert batch['completed'];out={}
    for name in ('sampled','gauss3','compatible16'):
        cases=[]
        for level in range(3):
            folder=OUT/'short-moving'/f'{name}-L{level}';f=arrays(folder/'stress.npz');rr=readrows(folder/'steps.jsonl');cases.append((f,rr))
        pairs=[comparison(cases[i][0]['P'],cases[i+1][0]['P'][::2],cases[i][0]['V'],cases[i][0]['time']) for i in range(2)]
        out[name]=dict(pairs=pairs,passed=max(pairs[-1]['relative'],pairs[-1]['terminal_relative'])<.02,
            histories=[max(r['history_commit_max'] for r in rs) for f,rs in cases],energy_losses=[sum(r['constraint_kinetic_loss_J'] for r in rs) for f,rs in cases])
    write(OUT/'short-moving-acceptance.json',dict(completed=True,cases=out,scope='Native weighted stress norms; 6.25 ms common small perturbation release, not a combined full loading cycle.'))

if __name__=='__main__':
    import sys
    if sys.argv[1]=='cycle':cycle()
    else:short()
