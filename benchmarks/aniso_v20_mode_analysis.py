"""Coherent reaction clusters and midpoint phase prediction, v20."""
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v20_common import *
from benchmarks.aniso_v20_modes import gram
from engine.aniso_phase1.separate_kinetic import KineticGeometry
from engine.aniso_phase1.endpoint_boundary import right_lift

def cluster(omega,rtol=1e-6):
    ids=np.argsort(omega);groups=[]
    for i in ids:
        if not groups or omega[i]-omega[groups[-1][0]]>rtol*omega[i]:groups.append([int(i)])
        else:groups[-1].append(int(i))
    return groups

def main():
    out={}
    for path in sorted((OUT/'modes').glob('[01]*.npz')):
        z=np.load(path);w=z['omega'];c=z['reaction_cos'];s=z['reaction_sin'];r=load(path.with_suffix('.json'));cc,ss,cs=gram(w,r['duration']);groups=[]
        details={v['mode']:v for v in r['reaction_modes']}
        for ids in cluster(w):
            ix=np.ix_(ids,ids);power=c[ids]@cc[ix]@c[ids]+s[ids]@ss[ix]@s[ids]+2*c[ids]@cs[ix]@s[ids];groups.append(dict(modes=ids,omega_rad_s=[float(w[i]) for i in ids],reaction_rms_N=float(np.sqrt(max(power,0))),min_stabilization_fraction=min(details[i]['stabilization_fraction'] for i in ids),max_exterior_stabilization_fraction=max(details[i]['exterior_stabilization_fraction'] for i in ids)))
        out[path.stem]=dict(groups=sorted(groups,key=lambda r:-r['reaction_rms_N']),same_cluster_relative_band=1e-6,full_rms_N=r['total_oscillatory_reaction_rms_N'])
    predictions=[]
    for kind in ('sampled','gauss3'):
        s,e,m,h,meta=snapshot(1.4)
        if kind=='gauss3':s,e,m,h=lift_state(s,e,m,h,meta,*gauss_sites(h,3))
        g=KineticGeometry(s,e,m,h);l=right_lift(g,h)
        from engine.aniso_phase1.carrier_driven import projection
        from engine.aniso_phase1.unresolved_velocity import pack
        zz=pack(s.v,s.C);full,_=projection(g.J,g.metric,zz);free,_=projection(g.J@g.Q,g.metric,zz);initial_impulse=-float(np.sum((g.J@l)*(g.metric[:,None]*(full-free))))
        z=np.load(OUT/'modes'/f'1.40-{kind}.npz');w=z['omega'];phi=z['phi'];Q=z['Q'];f=e.evaluate(s.Y)['force'];fm=phi.T@(Q.T@f).T.ravel();u_eq=-(phi@(fm/(w*w))).reshape(3,-1).T;R0=float(np.sum(l*f)+l.T.ravel()@e.tangent(s.Y)@(Q@u_eq).T.ravel())
        for level,dt in enumerate([.0000625,.00003125,.000015625]):
            path=OUT/'snapshot-runs'/f'1.40-{kind}-fixed-{level}'/'steps.jsonl'
            if not path.exists():continue
            rows=[__import__('json').loads(v) for v in path.read_text().splitlines()];k=np.arange(len(rows))+.5;theta=2*np.arctan(w*dt/2);terms=np.cos(theta[:,None]/2)*(z['reaction_cos'][:,None]*np.cos(theta[:,None]*k)+z['reaction_sin'][:,None]*np.sin(theta[:,None]*k));pred=R0+terms.sum(0);pred[0]+=initial_impulse/dt;actual=np.array([r['reaction_N'] for r in rows]);rms=lambda v:float(np.sqrt(np.mean(v*v)))
            predictions.append(dict(kind=kind,level=level,dt=dt,rms_actual_N=rms(actual),rms_predicted_N=rms(pred),absolute_error_N=rms(pred-actual),relative_error=rms(pred-actual)/rms(actual),correlation=float(np.corrcoef(pred,actual)[0,1]),first_constraint_impulse_Ns=initial_impulse,after_first_relative_error=rms(pred[1:]-actual[1:])/rms(actual[1:])))
            np.savez_compressed(OUT/'modes'/f'prediction-{kind}-L{level}.npz',actual=actual,predicted=pred,terms=terms,time=1.4+(k+.5)*dt)
    write(OUT/'reaction-clusters.json',dict(completed=True,records=out,midpoint_predictions=predictions,caveat='Coherent near-degenerate groups reduce arbitrary eigenvector splitting. Finite-window phase sensitivity and nonlinear changes remain. Prediction uses exact discrete midpoint phase, no reaction smoothing.'))
if __name__=='__main__':main()
