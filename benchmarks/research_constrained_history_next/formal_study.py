"""Bounded formal static audit; never promotes a dynamic model."""
import argparse,json,time,resource
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_constrained_history_next.formal import FormalCandidate,sh_axis,paired
from engine.aniso_phase1.tensor_metrics import sampling

def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);t=time.perf_counter()
    c=FormalCandidate();s=c.space;r={'identity':c.identity}
    # Interior points in BOTH entire grip volumes, including off-grid positions.
    rng=np.random.default_rng(7107)
    X=rng.uniform([.125,.375,.375],[.875,.625,.625],(128,3));X[:64,0]=rng.uniform(.125,.25,64);X[64:,0]=rng.uniform(.75,.875,64)
    N,D=c.basis(X);ids=s.free_scalar_ids
    r['grip_value_max']=float(abs(N[:,ids]).max());r['grip_gradient_max']=float(abs(D[:,ids]).max())
    probes=np.array([[.25,.5,.5],[.75,.5,.5]])
    original=paired([sampling(e,2,probes[:,a]) for a,e in enumerate(s.oldedges)])@s.oldA
    naive=paired([sh_axis(e,probes[:,a]) for a,e in enumerate(s.oldedges)])@s.oldA
    r['old_grip_max']=float(abs(original[:,c.free_carriers]).max())
    r['naive_grip_max']=float(abs(naive[:,c.free_carriers]).max())
    X=rng.uniform([.3,.39,.39],[.7,.61,.61],(24,3))
    X[:4,0]=[.260,.267,.733,.740]
    N,D=c.basis(X);errors=[]
    for eps in (1e-5,1e-6,1e-7):
        cols=[]
        for a in range(3):
            e=np.eye(3)[a]*eps
            fd=(c.basis(X+e)[0]-c.basis(X-e)[0])/(2*eps)
            cols.append(float(np.linalg.norm(fd-D[:,:,a])/max(np.linalg.norm(D[:,:,a]),1e-30)))
        errors.append(cols)
    r['derivative_relative_by_epsilon']=errors
    masses=[]
    for order in (7,8):
        tick=time.perf_counter();M=c.mass(order);masses.append(M)
        r['mass_q'+str(order)+'_seconds']=time.perf_counter()-tick
    M=masses[0];r['mass_q7_q8_relative']=float(np.linalg.norm(M-masses[1])/np.linalg.norm(masses[1]))
    P,offset,Z,report=c.stationary_coordinates(M);r['stationary']=report
    r['raw_mass_free_eigenvalues']=np.linalg.eigvalsh(M[np.ix_(s.free_scalar_ids,s.free_scalar_ids)]).tolist()
    r['full_carrier_local_cross_norm']=float(np.linalg.norm(M[:s.n,s.n:]))
    r['full_free_fixed_cross_norm']=float(np.linalg.norm(M[np.ix_(s.free_scalar_ids,s.fixed_scalar_ids)]))
    r['basis_null_relative']=float(np.linalg.norm(N@Z)/np.linalg.norm(N))
    r['gradient_null_relative']=float(np.linalg.norm(np.einsum('qna,nk->qka',D,Z))/np.linalg.norm(D))
    full=rng.normal(size=(s.ndof,3))*1e-5;v=rng.normal(size=full.shape)
    E,f,Hv=c.stabilization(full,v);eps=1e-7
    Ep,fp,_=c.stabilization(full+eps*v);Em,fm,_=c.stabilization(full-eps*v)
    r['absolute_stabilization_force_rel']=float(abs((Ep-Em)/(2*eps)-np.sum(f*v))/max(abs(np.sum(f*v)),1e-20))
    r['absolute_stabilization_action_rel']=float(np.linalg.norm((fp-fm)/(2*eps)-Hv)/np.linalg.norm(Hv))
    # Two inverse coefficient increments do NOT reconstruct the original absolute map.
    d=np.zeros_like(full);d[c.free_carriers]=rng.normal(size=(len(c.free_carriers),3))*.001
    x1=X+N@d;F1=np.eye(3)+np.einsum('qna,ni->qia',D,d)
    N1,D1=c.basis(X,x1,F1);x2=x1-N1@d
    r['inverse_coefficient_cycle_position_gap']=float(np.max(np.linalg.norm(x2-X,axis=1)))
    # Independently pushed shadow points test three compatible incremental maps.
    eps=1e-6;xp=[X+eps*np.eye(3)[a] for a in range(3)];xm=[X-eps*np.eye(3)[a] for a in range(3)]
    fp=[np.tile(np.eye(3),(len(X),1,1)) for _ in range(6)]
    xx=X.copy();FF=np.tile(np.eye(3),(len(X),1,1))
    for factor in (.5,-.3,.2):
        inc=factor*d;nn,dd=c.basis(X,xx,FF);xx=xx+nn@inc;FF=FF+np.einsum('qna,ni->qia',dd,inc)
        for a in range(3):
            for sign,arr,j in ((1,xp,a),(-1,xm,a+3)):
                nn,dd=c.basis(X+sign*eps*np.eye(3)[a],arr[a],fp[j]);arr[a]=arr[a]+nn@inc;fp[j]=fp[j]+np.einsum('qna,ni->qia',dd,inc)
    fd=np.stack([(xp[a]-xm[a])/(2*eps) for a in range(3)],axis=-1)
    r['three_step_history_derivative_relative']=float(np.linalg.norm(fd-FF)/np.linalg.norm(FF))
    r['cycle_min_detF']=float(np.linalg.det(F1).min())
    r['static_pass']=bool(r['grip_value_max']<1e-10 and r['grip_gradient_max']<1e-8 and max(errors[1])<1e-4 and r['mass_q7_q8_relative']<1e-8 and r['basis_null_relative']<1e-10 and r['gradient_null_relative']<1e-10)
    r['gate_A']=False
    r['dynamic_blocker']='Updated SH histories are path dependent in physical space. A sum of coefficient increments does not define the original absolute carrier Y. Keeping the old Ks expression alone does not close its original stabilization meaning; no dynamic promotion.'
    r['elapsed_seconds']=time.perf_counter()-t;r['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    np.savez_compressed(out/'static-operators.npz',M=M,P=P,offset=offset,null=Z)
    (out/'report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)
    return r
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',required=True);run(a.parse_args().out)
