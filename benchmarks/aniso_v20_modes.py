"""Reaction-ranked local modes with stiffness whitening, no mass shift."""
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v20_common import OUT,snapshot,lift_state,load,write,gauss_sites
from engine.aniso_phase1.separate_kinetic import KineticGeometry
from engine.aniso_phase1.endpoint_boundary import right_lift,prescribed_speed
from engine.aniso_phase1.unresolved_velocity import pack
from engine.aniso_phase1.history_increment import material_tangent
from engine.aniso_phase1.carrier_joint import gradient
from benchmarks.aniso_carrier_joint import spectrum

def gram(omega,T):
    x=omega[:,None]*T;y=omega[None,:]*T
    sinc=lambda a:np.sinc(a/np.pi)
    def s(a):return np.divide(2*np.sin(a/2)**2,a,out=np.zeros_like(a),where=abs(a)>1e-30)
    return .5*(sinc(x-y)+sinc(x+y)),.5*(sinc(x-y)-sinc(x+y)),.5*(s(x+y)+s(y-x))

def model(s,e,m,h,duration=.0125):
    g=KineticGeometry(s,e,m,h);Q=g.Q;n=Q.shape[1];K=e.tangent(s.Y,Q);Kfull=e.tangent(s.Y);Ks=la.block_diag(*([Q.T@e.Ks@Q]*3));assert spectrum(K)['passed']
    L=la.cholesky(K,lower=True);W=la.solve_triangular(L.T,np.eye(len(K)),lower=False);Jr=np.sqrt(g.metric)[:,None]*(g.J@Q)
    A=np.einsum('pi,aij->apj',Jr,W.reshape(3,n,-1)).reshape(-1,len(K));U,sv,Vh=la.svd(A,full_matrices=False);keep=sv>1e-12*sv[0];U=U[:,keep];sigma=sv[keep];phi=W@Vh[keep].T/sigma
    omega=1/sigma;lift=right_lift(g,h);ell=lift.T.ravel();full=np.einsum('ni,aij->naj',Q,phi.reshape(3,n,-1));flat=full.transpose(1,0,2).reshape(3*e.n,-1)
    kin_lift=(np.sqrt(g.metric)[:,None]*(g.J@lift)).T.ravel();cm=kin_lift@U;ck=ell@Kfull@flat;coupling=ck-omega**2*cm
    f=e.evaluate(s.Y)['force'];fmodal=phi.T@(Q.T@f).T.ravel();speed=prescribed_speed(s.time);z=pack(s.v,s.C)-g.J@lift*speed
    velocity=U.T@(np.sqrt(g.metric)[:,None]*z).T.ravel()
    if .6<s.time<1.1-1e-10:
        angle=2*np.pi*(s.time-.6);ac=.0025*np.cos(angle);ass=-.0025*np.sin(angle);rate=2*np.pi
        const=-fmodal+ck*ac;cosforce=(rate*rate*cm-ck)*ac;sinforce=(rate*rate*cm-ck)*ass
        cosine=-const/omega**2-cosforce/(omega**2-rate**2);sine=velocity/omega-sinforce*rate/(omega*(omega**2-rate**2))
    else:cosine=fmodal/omega**2;sine=velocity/omega
    cr=coupling*cosine;sr=coupling*sine;Gcc,Gss,Gcs=gram(omega,duration);power=cr*cr*np.diag(Gcc)+sr*sr*np.diag(Gss)+2*cr*sr*np.diag(Gcs);order=np.argsort(-power)
    total=float(cr@Gcc@cr+sr@Gss@sr+2*cr@Gcs@sr);records=[];F=gradient(e.B,s.Y)
    for idx in order:
        u=full[:,:,idx];norm=la.norm(u);un=u/norm;dF=gradient(e.B,un);dP=material_tangent(F,e.A,dF,e.params);mat=e.V*np.sum(dF*dP,axis=(1,2));pr=e.P@un[e.ids];stab=e.weights[:,None]*np.sum(pr*pr,axis=2);rowenergy=np.zeros(e.n);np.add.at(rowenergy,e.ids.ravel(),stab.ravel())
        vel=(g.J@un);kin=g.metric[:,None]*vel*vel;kp=np.sum(kin.reshape(4,len(m),3),axis=(0,2));grip=(s.x[:,0]<=.3125)|(s.x[:,0]>=.6875);outside=(s.Y[:,1]<.375)|(s.Y[:,1]>.625)|(s.Y[:,2]<.375)|(s.Y[:,2]>.625)|(s.Y[:,0]<.125)|(s.Y[:,0]>.875)
        den=float(mat.sum()+stab.sum());records.append(dict(mode=int(idx),omega_rad_s=float(omega[idx]),period_s=float(2*np.pi/omega[idx]),reaction_self_rms_N=float(np.sqrt(max(power[idx],0))),self_power_fraction=float(power[idx]/max(power.sum(),1e-300)),reaction_coupling=float(coupling[idx]),
            material_stiffness=float(mat.sum()),stabilization_stiffness=float(stab.sum()),effective_inertia=float(kin.sum()),stabilization_fraction=float(stab.sum()/den),kinetic_C_fraction=float(kin[len(m):].sum()/kin.sum()),grip_kinetic_fraction=float(kp[grip].sum()/kp.sum()),exterior_stabilization_fraction=float(rowenergy[outside].sum()/rowenergy.sum()),unit_K_identity_relative=abs(den-omega[idx]**2*kin.sum())/den))
    groups=[]
    for count in (1,3,10):
        ids=order[:count];rest=np.setdiff1d(np.arange(len(omega)),ids)
        def p(ii):return float(cr[ii]@Gcc[np.ix_(ii,ii)]@cr[ii]+sr[ii]@Gss[np.ix_(ii,ii)]@sr[ii]+2*cr[ii]@Gcs[np.ix_(ii,ii)]@sr[ii])
        groups.append(dict(top=count,group_rms_N=float(np.sqrt(max(p(ids),0))),remaining_rms_N=float(np.sqrt(max(p(rest),0)))))
    return dict(time=s.time,particles=len(m),static_gate=spectrum(K),rank=len(sigma),zero_inertia_modes=int((~keep).sum()),singular_ratio=float(sv[-1]/sv[0]),rank_sensitivity={str(t):int(np.sum(sv>t*sv[0])) for t in (1e-14,1e-12,1e-10,1e-8)},fastest_mode=int(np.argmax(omega)),reaction_modes=records,groups=groups,total_oscillatory_reaction_rms_N=float(np.sqrt(max(total,0))),duration=duration,
        caveat='Local frozen-coefficient linearization with prescribed cosine unloading or hold. Ranking includes inertia in reaction and actual lifted initial velocity; self powers include cross terms only in group/total results. Weak modes near numerical rank thresholds are provisional.'),dict(omega=omega,phi=phi,full=full,reaction_cos=cr,reaction_sin=sr,sv=sv,Q=Q,K=K)

def main():
    dest=OUT/'modes';dest.mkdir(exist_ok=False);rows=[]
    for t in (.85,1.1,1.4,1.6):
        s,e,m,h,meta=snapshot(t)
        for kind in ('sampled','gauss3'):
            case=(s,e,m,h) if kind=='sampled' else lift_state(s,e,m,h,meta,*gauss_sites(h,3))
            r,a=model(*case);name=f'{t:.2f}-{kind}';np.savez_compressed(dest/f'{name}.npz',**a);write(dest/f'{name}.json',r);rows.append(dict(time=t,kind=kind,top=r['reaction_modes'][:5],rank=r['rank'],total_reaction_rms_N=r['total_oscillatory_reaction_rms_N']));print(name,rows[-1],flush=True)
    write(OUT/'reaction-modes.json',dict(completed=True,records=rows,material_and_patch_energy_unchanged=True,source_snapshots={str(t):{k:v for k,v in snapshot(t)[-1].items() if k in ('path','sha256')} for t in (.85,1.1,1.4,1.6)}))
if __name__=='__main__':main()
