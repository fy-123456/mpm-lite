"""Fresh-process full frozen solve or transactional checkpoint continuation."""
import argparse
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_common_kinematics_next.model import CommonOperator
from engine.aniso_phase1.research_common_kinematics_next.checkpoint import load,save
from engine.aniso_phase1.research_unified_lite_poro.solve import advance
from .run import write

def array_inventory(obj,seen=None,path='op'):
    seen=set() if seen is None else seen
    if id(obj) in seen:return []
    seen.add(id(obj))
    if isinstance(obj,np.ndarray):return [dict(path=path,shape=list(obj.shape),bytes=obj.nbytes)]
    if isinstance(obj,dict):items=obj.items()
    elif isinstance(obj,(tuple,list)):items=enumerate(obj)
    elif hasattr(obj,'__dict__'):items=vars(obj).items()
    else:return []
    return [r for k,v in items for r in array_inventory(v,seen,path+'.'+str(k))]

def residual(op,d,v0,p0,p,h,time=0.):
    V0=op.geometry(np.zeros_like(d),False)[0];V,G,H,J=op.geometry(d,False)
    H=op.geometry(d/2)[2];pm=(p+p0)/2;z=np.linalg.solve(H,op.top.B.T@pm-op.gb)
    velocity=2*d/h-v0;Gb=op.discrete_G(np.zeros_like(d),d)
    mech=op.M@(velocity-v0)/h+op.avf_force(np.zeros_like(d),d)-op.alpha*np.einsum('cni,c->ni',Gb,pm)-op.load(time+h/2)
    mass=op.capacity*(p-p0)+op.alpha*(V-V0)+h*op.top.B@z
    return np.r_[mech.ravel(),mass/h]

def frozen_replay(package,inputs,output):
    op=CommonOperator.load_package(package)
    with np.load(inputs,allow_pickle=False) as z:v=z['v'];p=z['p'];h=float(z['dt']);t=float(z['time'])
    d,v1,p1,flux,m=advance(op,np.zeros_like(v),v,p,h,t)
    rng=np.random.default_rng(21);direction=rng.normal(size=d.shape)*.01;dp=rng.normal(size=p.shape)*.01
    products=[]
    for e in (1e-5,1e-6,1e-7):
        products.append((residual(op,d+e*direction,v,p,p1+e*dp,h,t)-residual(op,d-e*direction,v,p,p1-e*dp,h,t))/(2*e))
    inventory=array_inventory(op)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output.with_suffix('.npz'),increment=d,v=v1,p=p1,flux=flux)
    write(output,dict(metrics=m,identity=op.identity,arrays=inventory,array_bytes=sum(x['bytes'] for x in inventory),
        particle_arrays_loaded=False,full_coupled_step=True,
        jvp_last_pair_relative=float(np.linalg.norm(products[-1]-products[-2])/max(np.linalg.norm(products[-1]),1e-12))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path);p.add_argument('--inputs',type=Path)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--dt',type=float,default=.00125);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    if a.checkpoint:
        b=load(a.checkpoint);before=b.state.digest();m=b.step(a.dt);save(b,a.output.with_suffix('.npz'))
        write(a.output,dict(before=before,after=b.state.digest(),metrics=m))
    else:frozen_replay(a.package,a.inputs,a.output)
