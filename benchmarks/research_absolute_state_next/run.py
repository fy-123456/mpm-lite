"""Absolute closure, fixed-inertia and boundary work probes with explicit scope."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.research_absolute_state_next.state import MaterialStateModel,State

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2))

def closure(run,m):
    rng=np.random.default_rng(7108);s=m.space;r=m.r;state=m.rest();shape=state.q.shape
    X=rng.uniform([.13,.38,.38],[.87,.62,.62],(96,3));X[:16,0]=rng.uniform(.13,.24,16);X[16:32,0]=rng.uniform(.76,.87,16)
    d=np.zeros(shape);d[r.free]=rng.normal(size=(len(r.free),3))*1e-5
    B,D=m.basis(X);transfer=m.corrected_transfer(state,X,d)
    q=d.copy();Y=m.absolute_Y(q);U,f,Hd=m.stabilization(q,d);rows=[]
    for eps in (1e-3,1e-4,1e-5):
        Ep,fp,_=m.stabilization(q+eps*d);Em,fm,_=m.stabilization(q-eps*d)
        rows.append(dict(eps=eps,force_relative=float(abs((Ep-Em)/(2*eps)-np.sum(f*d))/max(abs(np.sum(f*d)),1e-12)),
            tangent_relative=float(la.norm((fp-fm)/(2*eps)-Hd)/max(la.norm(Hd),1e-12))))
    zeroU,zerof,_=m.stabilization(state.q)
    # Real compatible state progression, including one small prescribed translation.
    histories=[]
    for factor in (1.,.5,-.3,-1.2):
        inc=factor*d;packet=m.corrected_transfer(state,X,inc);x,F,_=m.fields(state,X)
        nxt=m.trial(state,inc,np.zeros(shape),.001)
        xx,FF,_=m.fields(nxt,X)
        histories.append(dict(position_gap=float(np.max(abs(xx-x-packet['displacement']))),F_gap=float(np.max(abs(FF-F-packet['material_gradient']))),minJ=float(np.linalg.det(FF).min())))
        state=nxt
    # Objectivity of the original absolute coefficient stabilization.
    angle=.1;R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    full=r.expand(q);rot=full@R.T;rot[:s.n]+=(s.reference[:s.n]@R.T-s.reference[:s.n]);qr=r.project(rot)
    Ur,fr,_=m.stabilization(qr)
    translation=full.copy();translation[:s.n]+=np.array([.003,-.002,.001]);Ut,_,_=m.stabilization(r.project(translation))
    report=dict(identity=m.identity,original_mass_audit=r.audit,derivatives=rows,unloaded_stabilization_U=zeroU,unloaded_stabilization_free_force=float(la.norm(zerof[r.free])),
        raw_SH_displacement_gap=float(np.max(abs(transfer['raw']-transfer['exact']))),corrected_displacement_gap=float(np.max(abs(transfer['displacement']-transfer['exact']))),
        corrected_gradient_gap=float(np.max(abs(transfer['material_gradient']-transfer['exact_material_gradient']))),
        grip_free_displacement_max=float(abs(transfer['displacement'][:32]).max()),histories=histories,
        final_absolute_cycle_gap=float(np.max(abs(m.fields(state,X)[0]-X))),rotation_energy_absolute=abs(Ur-U),translation_energy_absolute=abs(Ut-U),
        algebraic_stationarity=float(la.norm(r.algebraic_residual(r.expand(q)))),state_bytes=state.q.nbytes+state.velocity.nbytes,
        gate_S_material_route=True,updated_C1_space_adopted=False,production_SH_integrated=False,
        limitation='Exact correction explicitly restores ORIGINAL material basis at point queries; it does not qualify naive or masked updated SH as the physical basis.')
    assert max(t['position_gap'] for t in histories)<1e-9 and max(t['F_gap'] for t in histories)<1e-8
    assert report['corrected_displacement_gap']<1e-10 and report['corrected_gradient_gap']<1e-9
    assert min(a['force_relative'] for a in rows)<.001 and min(a['tangent_relative'] for a in rows)<.001
    assert report['grip_free_displacement_max']<1e-10
    m.save(state,run/'S1/state.npz');restored=m.load(run/'S1/state.npz');assert restored.digest()==state.digest()
    write(run/'S1/closure.json',report)

def dry_probe(run,m):
    """Four-step LINEARIZED formal solid. Never claims nonlinear qualification."""
    r=m.r;s=m.space;M=np.kron(r.M,np.eye(3));K=r.K;ids=r.ids
    fixed=(3*r.fixed[:,None]+np.arange(3)).ravel();n=len(M);q=np.zeros(n);v=q.copy();h=.001
    # Right-grip affine-in-time displacement: t^2 * 0.1 m/s^2, left held.
    unit=np.zeros((r.P.shape[1],3));orig=s.carrier_X[s.fixed_scalar_ids,0];unit[r.fixed[orig>=.75],0]=1.;unit=unit.ravel()
    A=2*M/h**2+.5*K;fac=la.cho_factor(A[np.ix_(ids,ids)])
    X=np.stack(np.meshgrid(np.linspace(.125,.875,33),np.linspace(.375,.625,5),np.array([.5]),indexing='ij'),axis=-1).reshape(-1,3)
    frames=[m.fields(State(q.reshape(-1,3),v.reshape(-1,3)),X)[0]];vel=[np.zeros_like(X)];metrics=[]
    # Exact rest tangent model is only an inertia/work test; pressure disabled.
    for i in range(4):
        t1=(i+1)*h;delta=np.zeros(n);delta[fixed]=unit[fixed]*.1*(t1*t1-(t1-h)**2)
        rhs=2*M@v/h-K@q
        delta[ids]=la.cho_solve(fac,rhs[ids]-A[np.ix_(ids,fixed)]@delta[fixed])
        q1=q+delta;v1=2*delta/h-v;fm=K@((q+q1)/2)
        reaction=M@(v1-v)/h+fm
        T0=.5*v@M@v;T1=.5*v1@M@v1;U0=.5*q@K@q;U1=.5*q1@K@q1
        work=float(reaction[fixed]@delta[fixed]);defect=float(T1+U1-T0-U0-work)
        fstate=State(q1.reshape(-1,3),v1.reshape(-1,3),t1,i+1);xx,FF,vv=m.fields(fstate,X)
        metrics.append(dict(time=t1,max_u=float(np.linalg.norm(xx-X,axis=1).max()),minJ=float(np.linalg.det(FF).min()),
            free_residual=float(abs(reaction[ids]).max()),boundary_work=work,energy_defect=defect,
            boundary_velocity_error=float(abs(v1[fixed]-2*.1*t1*unit[fixed]).max()),
            reaction_x=float(reaction.reshape(-1,3)[r.fixed[orig>=.75],0].sum()),kinetic=T1,potential=U1))
        q,v=q1,v1;frames.append(xx);vel.append(vv)
    assert max(a['free_residual'] for a in metrics)<1e-7
    assert max(abs(a['energy_defect']) for a in metrics)<1e-10
    # Correction requires no physical metric change; retain all prescribed rows.
    write(run/'S2/dry-probe.json',dict(scope='four-step linearized ORIGINAL formal solid; no fluid or nonlinear trajectory qualification',metrics=metrics,
        M_hash_unchanged=True,cross_free_fixed_norm=float(la.norm(M[np.ix_(ids,fixed)])),projection_work=0.,gate_M_nonlinear=False))
    np.savez_compressed(run/'S2/frames.npz',X=X,x=frames,velocity=vel,time=np.arange(5)*h,q=q.reshape(-1,3),v=v.reshape(-1,3))
    m.save(State(q.reshape(-1,3),v.reshape(-1,3),4*h,4),run/'S2/final.npz')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();start=time.perf_counter()
    a.run.joinpath('S1').mkdir(exist_ok=True);a.run.joinpath('S2').mkdir(exist_ok=True)
    m=MaterialStateModel();closure(a.run,m);dry_probe(a.run,m)
    write(a.run/'S2/resource.json',dict(seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024));print('static closure and linearized probe completed',flush=True)
