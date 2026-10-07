"""Actual moving Gauss-particle histories checked with moving-piece cubature."""
import time
from pathlib import Path
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v20_common import *
from benchmarks.aniso_v20_runs import setup
from benchmarks.aniso_v20_quadrature import integrate
from benchmarks.aniso_v20_modes import gram
from engine.aniso_phase1.integrated_avf import IntegratedAVF
from engine.aniso_phase1.swept_quadrature import sweep
from engine.aniso_phase1.carrier_joint import State
from engine.aniso_phase1.endpoint_boundary import right_lift,prescribed_speed
from engine.aniso_phase1.unresolved_velocity import pack
from engine.aniso_phase1.local_reference import LO,HI

def modes(s,e,g,h):
    Q=g.Q;n=Q.shape[1];K=e.tangent(s.Y,Q);L=la.cholesky(K,lower=True);W=la.solve_triangular(L.T,np.eye(len(K)),lower=False);J=np.sqrt(g.metric)[:,None]*(g.J@Q);A=np.einsum('pi,aij->apj',J,W.reshape(3,n,-1)).reshape(-1,len(K));U,sv,Vh=la.svd(A,full_matrices=False);assert sv[-1]>1e-12*sv[0];phi=W@Vh.T/sv;omega=1/sv;full=np.einsum('ni,aij->naj',Q,phi.reshape(3,n,-1));flat=full.transpose(1,0,2).reshape(3*e.n,-1);lift=g.R@right_lift(g,h);cm=(np.sqrt(g.metric)[:,None]*(g.J@lift)).T.ravel()@U;ck=lift.T.ravel()@e.tangent(s.Y)@flat;coupling=ck-omega**2*cm;f=phi.T@(Q.T@e.evaluate(s.Y)['force']).T.ravel();z=pack(s.v,s.C)-g.J@lift*prescribed_speed(s.time);velocity=U.T@(np.sqrt(g.metric)[:,None]*z).T.ravel()
    if .6<s.time<1.1-1e-10:
        angle=2*np.pi*(s.time-.6);ac=.0025*np.cos(angle);ass=-.0025*np.sin(angle);rate=2*np.pi;const=-f+ck*ac;co=(rate*rate*cm-ck)*ac;si=(rate*rate*cm-ck)*ass;c=-const/omega**2-co/(omega**2-rate**2);b=velocity/omega-si*rate/(omega*(omega**2-rate**2))
    else:c=f/omega**2;b=velocity/omega
    cr=coupling*c;sr=coupling*b;cc,ss,cs=gram(omega,.0125);power=cr*cr*np.diag(cc)+sr*sr*np.diag(ss)+2*cr*sr*np.diag(cs);ids=np.argsort(-power)[:6];v=full[:,:,ids].transpose(2,0,1);v/=la.norm(v,axis=(1,2))[:,None,None]
    return v,dict(mode_ids=ids.tolist(),omega_rad_s=omega[ids].tolist(),max_omega_rad_s=float(omega.max()),reaction_self_rms_N=np.sqrt(np.maximum(power[ids],0)).tolist())

def main():
    dest=OUT/'runtime-quadrature';dest.mkdir(exist_ok=False);scratch=Path(load(OUT/'fast-cycle/protocol.json')['scratch']);records=[]
    for t in (.85,1.1,1.4,1.6):
        k=round(t/.0000625);path=scratch/'gauss3-condensed-L3'/f'audit-{k:06d}.npz';recovered=OUT/'recovered/gauss3-condensed-L3'/path.name
        while not path.exists() and not recovered.exists():
            final=OUT/'completed-case-paths.json'
            if final.exists():
                v=load(final)['cases'].get('gauss3-condensed-L3')
                if v is not None and not v['completed']:raise RuntimeError('final finest trajectory failed')
            time.sleep(10)
        if recovered.exists():path=recovered
        # Wait for the writer to finish and the zip footer to appear.
        while True:
            try:
                with np.load(path) as z:s=State(z['x'],z['Y'],z['v'],z['C'],float(z['time']))
                break
            except (OSError,ValueError,__import__('zipfile').BadZipFile):time.sleep(1)
        _,e,m,h,meta=setup('gauss3-condensed');meta['particle_reference']=e.kinetic_reference;so=IntegratedAVF(s,e,m,h,condense=True);targets,modeinfo=modes(s,e,so.geometry,h);field=SnapshotField(s,meta['particle_reference']);edges=[]
        for j,(lo,hi) in enumerate(zip(LO,HI)):
            centers=(np.arange(-1,int(np.ceil(hi/h))+2)+.5)*h;edges.append(np.unique(np.r_[lo,hi,field.axes[j],centers[(centers>lo)&(centers<hi)]]))
        g=so.geometry;nativeM=g.J.T@(g.metric[:,None]*g.J);vel=np.einsum('pn,tna->tpa',g.J,targets);nativek=np.einsum('tpa,tpa,p->t',vel,vel,g.metric);checks=[];last=None
        rules=[(3,0),(4,0)]
        for order,refine in rules:
            X,V,info=sweep(edges,field.maps['x'],h,order,refine);M,ki=integrate(s,e,m,h,meta,X,V,targets)
            if last is not None:info.update(matrix_relative=float(la.norm(M-last[0])/la.norm(M)),mode_inertia_max_relative=float(np.max(abs(ki-last[1])/ki)))
            checks.append(info);last=M,ki;np.savez_compressed(dest/f'{t:.2f}-o{order}-r{refine}.npz',M=M,mode_inertia=ki)
            if order==4 and refine==0 and (info['matrix_relative']>1e-5 or info['mode_inertia_max_relative']>.02):rules.append((4,1))
        r=dict(time=t,source=str(path),source_sha256=sha(path),checks=checks,modes=modeinfo,native_matrix_relative=float(la.norm(nativeM-M)/la.norm(M)),native_mode_inertia_relative=(abs(nativek-ki)/ki).tolist(),cut_reference_passed=checks[-1]['matrix_relative']<1e-5 and checks[-1]['mode_inertia_max_relative']<.02,native_reaction_mode_inertia_passed=bool(np.max(abs(nativek-ki)/ki)<.02));records.append(r);write(dest/f'{t:.2f}.json',r);print(r,flush=True);write(OUT/'runtime-quadrature.json',dict(completed=len(records)==4,records=records,actual_moving_kinetic_particle_fields=True,history_remeshing=False))
if __name__=='__main__':main()
