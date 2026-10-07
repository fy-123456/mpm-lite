"""Separate sampled direction reconstruction from same-space quadrature error."""
import argparse,json
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_unified_lite_poro.space import Space
from engine.aniso_phase1.research_unified_lite_poro.model import seed,FrozenOperator,PARAMS
from engine.aniso_phase1.research_eulerian_poro_next.materials import make_operator

BASE=Path(__file__).resolve().parents[2]/'docs/results/unified-lite-poro/20261005T123342Z-unified-lite-poro'

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))

def exact_operator(space,order=7,kind='planar'):
    X,w,_=space.rule(order)
    angle=np.deg2rad(20+50*X[:,1]/.25+15*X[:,0])
    a=np.c_[np.cos(angle),np.sin(angle),np.zeros(len(X))]
    if kind=='nonplanar':
        tilt=.12*np.sin(2*np.pi*X[:,0])*np.cos(np.pi*X[:,2]/.25)
        a=np.c_[np.cos(angle)*np.cos(tilt),np.sin(angle)*np.cos(tilt),np.sin(tilt)]
    A=a[:,:,None]*a[:,None,:];flat=A.reshape(-1,9)
    return FrozenOperator(space,X,w,A,flat[:,:,None]*flat[:,None,:])

def diagnostic(run):
    s=Space();states={'zero':np.zeros((len(s.free),3))}
    for name,case in [('load','fixed-short-ppc4'),('unload','fixed-cycle-ppc3')]:
        with np.load(BASE/'S3'/case/'terminal.npz') as d:states[name]=d['q'].copy()
    rows=[]
    for name,q in states.items():
        ref=exact_operator(s);Er,fr,_=ref.material(q)
        for ppc in (2,4,8):
            p=seed(s,ppc)
            for rule,order in [('exact',4),('exact',5),('exact',7),('moments',4),
                               ('fixed-positive',4),('fixed-positive',7),('director-log',4),('director-log',7)]:
                if rule=='exact':op=exact_operator(s,order);info={}
                elif rule=='moments':op=FrozenOperator.from_particles(s,p);info={}
                else:op,info=make_operator(s,p,rule,order)
                E,f,_=op.material(q)
                iso=FrozenOperator(s,op.points,op.weights,op.A2,op.A4,params={**PARAMS,'k_f':0.})
                Ei,fi,_=iso.material(q)
                ir=FrozenOperator(s,ref.points,ref.weights,ref.A2,ref.A4,params={**PARAMS,'k_f':0.})
                Eir,fir,_=ir.material(q)
                rows.append(dict(state=name,ppc=ppc,rule=rule,order=order,Nq=len(op.points),
                    energy_J=E,energy_abs_error_J=abs(E-Er),force_abs_error_N=float(np.linalg.norm(f-fr)),
                    force_relative=float(np.linalg.norm(f-fr)/max(np.linalg.norm(fr),1e-10)),
                    reference_force_norm_N=float(np.linalg.norm(fr)),
                    fiber_force_relative=float(np.linalg.norm((f-fi)-(fr-fir))/max(np.linalg.norm(fr-fir),1e-10)),
                    isotropic_force_abs_error_N=float(np.linalg.norm(fi-fir)),fit=info))
    # Separate diagnostics, not a hidden paper test or a selection training set.
    nonplanar=[];q=states['unload'];ref=exact_operator(s,7,'nonplanar');fr=ref.material(q)[1]
    for ppc in (4,8):
        p=seed(s,ppc);X=p.X;angle=np.deg2rad(20+50*X[:,1]/.25+15*X[:,0]);tilt=.12*np.sin(2*np.pi*X[:,0])*np.cos(np.pi*X[:,2]/.25)
        a=np.c_[np.cos(angle)*np.cos(tilt),np.sin(angle)*np.cos(tilt),np.sin(tilt)];p.A=a[:,:,None]*a[:,None,:]
        for rule in ('fixed-positive','director-or-positive'):
            op,info=make_operator(s,p,rule)
            nonplanar.append(dict(ppc=ppc,rule=rule,force_relative=float(np.linalg.norm(op.material(q)[1]-fr)/np.linalg.norm(fr)),fit=info))
    write(run/'S1/material-decomposition.json',dict(scope='same-space fixed-state diagnostics only',rows=rows,nonplanar=nonplanar))
    # Recover the earlier 13.4% diagnostic from its old moments-cycle state.
    # Keep this separate from the preregistered states and record the actual q.
    with np.load(BASE/'S3/cycle-ppc3/terminal.npz') as d:qlegacy=d['q'].copy()
    p=seed(s,4);fr=exact_operator(s).material(qlegacy)[1]
    legacy=[]
    for rule in ('fixed-positive','director-log'):
        op,info=make_operator(s,p,rule)
        legacy.append(dict(rule=rule,force_relative=float(np.linalg.norm(op.material(qlegacy)[1]-fr)/np.linalg.norm(fr))))
    np.savez_compressed(run/'S1/legacy-diagnostic-state.npz',q=qlegacy)
    write(run/'S1/legacy-state-recheck.json',dict(source=str(BASE/'S3/cycle-ppc3/terminal.npz'),
        note='supplementary source recovery; not an unseen test',rows=legacy))
    print(json.dumps({'unload_ppc4':[r for r in rows if r['state']=='unload' and r['ppc']==4], 'nonplanar':nonplanar},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();diagnostic(a.run)
