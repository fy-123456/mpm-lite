"""Independent moving-map, energy/work, stiffness and temporal acceptance."""
import argparse,json
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from benchmarks.aniso_v18_runs import OUT,BASE,load,write,initial
from benchmarks.aniso_v17_analysis import arrays,comparison,time_rms
from benchmarks.aniso_v17_modes import weighted_rms
from benchmarks.aniso_compatible_controls import stiffness
from benchmarks.aniso_dynamic_check import pk1
from engine.aniso_phase1.diagnostics import energy_density
from engine.aniso_phase1.carrier_joint import Geometry,State,gradient
from engine.aniso_phase1.unresolved_velocity import pack
from benchmarks.aniso_carrier_joint import spectrum


def audit(name,spec,path,stress,logged):
    z=arrays(path);s,e,m,h,_=initial(spec);old=State(z['x_before'],z['Y_before'],z['v_before'],z['C_before']);new=State(z['x'],z['Y'],z['v'],z['C']);g=Geometry(old,e,m,h)
    def elastic(Y):
        F=gradient(e.B,Y);Pm=pk1(F,e.A,200.);Um=float(e.V@energy_density(F,e.A,e.params));r=e.P@Y[e.ids]
        Us=.5*float(np.sum(e.weights[:,None,None]*r*r));force=np.zeros_like(Y)
        np.add.at(force,e.ids.ravel(),(e.weights[:,None,None]*(e.P.swapaxes(1,2)@r)).reshape(-1,3))
        force+=sum(b.T@(e.V[:,None]*Pm[:,:,j]) for j,b in enumerate(e.B))
        return F,Pm,Um,Us,force
    def metric(x):
        q=x/h-.5;q-=np.floor(q);D=h*h*(q*(1-q)+.25);return np.concatenate([m]+[m*D[:,j] for j in range(3)])
    F,P,Um,Us,_=elastic(new.Y);_,_,Um0,Us0,_=elastic(old.Y);dt=spec['dt'];dy=new.Y-old.Y
    fbar=sum(.5*elastic(old.Y+a*dy)[-1] for a in (.5-np.sqrt(3)/6,.5+np.sqrt(3)/6));zz=pack(new.v,new.C);zz0=pack(old.v,old.C);q=metric(old.x);q1=metric(new.x)
    kinetic=lambda a,b:.5*float(np.sum(a[:,None]*b*b));K=kinetic(q1,zz);Kf=kinetic(q,zz);K0=kinetic(q,zz0);work=float(np.sum(fbar*dy))
    W=z['W'];impulse=2*g.J.T@(q[:,None]*(g.J@W-zz0))+dt*fbar
    fixed=(g.nodes[:,0]*h<=.25)|(g.nodes[:,0]*h>=.75);grid=np.zeros((len(g.nodes),3));grid[g.nodes[:,0]*h>=.75,0]=1.
    lift=la.lstsq(g.E[fixed],grid[fixed],cond=1e-11)[0]
    bw=float(np.sum(impulse*lift)*logged['loading_speed']) if spec['mode']=='driven' else 0.
    Jproj=g.J@g.Q if spec['mode']=='hold' else g.J
    U,sv,_=la.svd(np.sqrt(q)[:,None]*Jproj,full_matrices=False);U=U[:,sv>1e-12*sv[0]];proj=U@(U.T@(np.sqrt(q)[:,None]*zz0))/np.sqrt(q)[:,None]
    expected=zz0+2*(g.J@W-proj)
    errors=dict(F=float(np.max(abs(F-z['F']))),stress_Pa=float(np.max(abs(P-stress))),x=float(np.max(abs(new.x-old.x-dt*g.T@W))),
        Y=float(np.max(abs(dy-dt*W))),velocity=float(np.max(abs(expected-zz))),
        grip_velocity=float(np.max(abs((g.E@W)[fixed]-grid[fixed]*logged['loading_speed']))),
        N=float(np.max(abs(g.N-z['N']))),E=float(np.max(abs(g.E-z['E']))),J=float(np.max(abs(g.J-z['J']))),
        material_J=abs(Um-logged['material_J']),stabilization_J=abs(Us-logged['stabilization_J']),kinetic_J=abs(K-logged['kinetic_J']),
        boundary_work_J=abs(bw-logged['boundary_work_J']),work_defect_J=abs(Kf-K0+work-bw-logged['kinetic_force_work_defect_J']),
        path_error_J=abs(Um+Us-Um0-Us0-work-logged['potential_quadrature_error_J']))
    current=Geometry(new,e,m,h);Q=current.Q;Km,_=stiffness(F,e.A,e.V,[sp.csr_matrix(b@Q) for b in e.B],200.);Ks=Q.T@e.Ks@Q
    Kstatic=Km.toarray()+la.block_diag(Ks,Ks,Ks);gate=spectrum(Kstatic);exact=e.tangent(new.Y,Q);fd=float(la.norm(Kstatic-exact)/la.norm(exact))
    assert max(errors.values())<1e-7 and max(v for k,v in errors.items() if k.endswith('_J'))<1e-12,(name,path,errors)
    assert gate['passed'] and fd<1e-6,(name,path,gate,fd)
    return dict(case=name,snapshot=path.name,errors=errors,current_massless=gate,fd_tangent_relative=fd)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('group',choices=['moving','cycle']);group=parser.parse_args().group
    assert not (OUT/f'{group}-acceptance.json').exists();p=load(OUT/'protocol.json');batch=load(OUT/f'batch-{group}.json');specs={n:s for n,s in p['cases'].items() if s['kind']==('snapshot' if group=='moving' else 'cycle')}
    cases={};frames={};rows={};audits=[];failures=[]
    for name,spec in specs.items():
        status=load(OUT/'cases'/name/'status.json')
        if not status['completed']:failures.append(dict(case=name,status=status));continue
        f=arrays(OUT/'cases'/name/'stress.npz');rr=[json.loads(v) for v in (OUT/'cases'/name/'steps.jsonl').read_text().splitlines()];frames[name]=f;rows[name]=rr
        s,e,m,h,_=initial(spec);g=Geometry(s,e,m,h);E0=e.evaluate(s.Y)['U']+.5*float(np.sum(g.metric[:,None]*pack(s.v,s.C)**2))
        sums={key:sum(r[key] for r in rr) for key in ('boundary_work_J','metric_change_J','kinetic_force_work_defect_J','potential_quadrature_error_J')}
        total=rr[-1]['total_J']-E0;balance=total-sum(sums.values());assert abs(balance)<1e-11
        case=dict(steps=len(rr),terminal=rr[-1],energy_change_J=total,energy_components_J=sums,budget_closure_J=balance,
            max_work_defect_J=max(abs(r['kinetic_force_work_defect_J']) for r in rr),max_history=max(r['history_commit_max'] for r in rr),
            max_path_error_J=max(abs(r['potential_quadrature_error_J']) for r in rr),max_grip_velocity_error=max(r['grip_velocity_error'] for r in rr),
            max_newton_residual=max(r['newton_residual'] for r in rr),
            rank_range=[min(r['kinetic_projection_rank'] for r in rr),max(r['kinetic_projection_rank'] for r in rr)])
        for snap in sorted((OUT/'cases'/name).glob('audit-*.npz')):
            k=int(snap.stem.split('-')[-1]);audits.append(audit(name,spec,snap,f['P'][k],rr[k-1]))
        if group=='moving' and spec['dt']==.000125:
            folder=BASE/'v16/avf/cases'/f'{"early_hold" if spec["label"]=="early" else "late_hold"}-moving-avf-fourth'
            z=arrays(folder/'frames.npz');old=pk1(z['F'].reshape(-1,3,3),np.tile(e.A,(len(z['F']),1,1)),200.).reshape(z['F'].shape)
            bridge=weighted_rms(f['P'][::40]-old,e.V)/weighted_rms(old,e.V);assert bridge<1e-5,bridge;case['v16_equivalent_path_bridge_relative']=bridge
        if group=='cycle':
            case['stages']={}
            for stage,bounds in p['cycle'].items():
                if stage=='peak_displacement':continue
                lo,hi=bounds;start=round(lo/spec['dt']);stop=round(hi/spec['dt']);part=rr[start:stop]
                Ebegin=rr[start-1]['total_J'] if start else E0;end=part[-1]
                case['stages'][stage]=dict(start_energy_J=Ebegin,end_energy_J=end['total_J'],energy_change_J=end['total_J']-Ebegin,
                    boundary_work_J=sum(r['boundary_work_J'] for r in part),metric_change_J=sum(r['metric_change_J'] for r in part),
                    solve_work_defect_J=sum(r['kinetic_force_work_defect_J'] for r in part),path_error_J=sum(r['potential_quadrature_error_J'] for r in part),
                    endpoint_stress_rms_Pa=end['stress_rms_Pa'],endpoint_reaction_N=end['reaction_N'],
                    max_kinetic_J=max(r['kinetic_J'] for r in part),min_kinetic_J=min(r['kinetic_J'] for r in part))
        cases[name]=case;print('audit',name,len(rr),flush=True)
    pairs={}
    labels=['early','late'] if group=='moving' else ['cycle']
    for label in labels:
        names=sorted([n for n in frames if specs[n]['label']==label],key=lambda n:specs[n]['dt'],reverse=True);records=[]
        for ca,fi in zip(names[:-1],names[1:]):
            a,b=frames[ca],frames[fi];ratio=round(specs[ca]['dt']/specs[fi]['dt']);np.testing.assert_allclose(a['time'],b['time'][::ratio],atol=1e-12)
            _,e,_,_,_=initial(specs[ca]);r=comparison(a['P'],b['P'][::ratio],e.V,a['time']);r.update(coarse=ca,fine=fi,dt_coarse=specs[ca]['dt'],dt_fine=specs[fi]['dt'])
            if group=='cycle':
                stages={}
                for stage,bounds in p['cycle'].items():
                    if stage=='peak_displacement':continue
                    lo,hi=bounds
                    mask=(a['time']>=lo-1e-12)&(a['time']<=hi+1e-12)
                    stages[stage]=comparison(a['P'][mask],b['P'][::ratio][mask],e.V,a['time'][mask])
                r['stages']=stages
                ra=np.array([row['reaction_N'] for row in rows[ca]]);rb=np.array([row['reaction_N'] for row in rows[fi]])[ratio-1::ratio]
                r['reaction_RMS_relative']=float(la.norm(ra-rb)/la.norm(rb))
                r['accepted']=max([r['relative'],r['terminal_relative']]+[x[k] for x in stages.values() for k in ('relative','terminal_relative')]+[r['reaction_RMS_relative']])<.02
            else:r['accepted']=max(r['relative'],r['terminal_relative'])<.02
            records.append(r)
        pairs[label]=records
    passed=not failures and bool(cases) and all(v and v[-1]['accepted'] for v in pairs.values())
    result=dict(completed=not failures,passed=passed,group=group,cases=cases,failures=failures,refinement=pairs,independent_audits=audits,
        steps=sum(r['steps'] for r in cases.values()),full_cycle_completed=group=='cycle' and not failures,production_default_changed=False)
    write(OUT/f'{group}-acceptance.json',result);print('ACCEPTANCE',group,passed,pairs,flush=True)
if __name__=='__main__':main()
