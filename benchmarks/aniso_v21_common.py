"""v21 provenance and independently evaluable static spatial metrics."""
import hashlib,json,tempfile
from pathlib import Path
import numpy as np
from benchmarks.aniso_v20_common import ROOT,BASE,controlled_case,load,write,sha
from benchmarks.aniso_boundary_reference import hessian
from benchmarks.aniso_local_q3 import gradient,chunks
OUT=BASE/'v21'
FIBER=np.array([1.,1.,0.])/np.sqrt(2)

def read_field(path,degree=None):
    with np.load(path) as z:
        e=[z[f'axis{k}'].copy() for k in range(3)];u=z['u'].copy();p=int(z['degree']) if 'degree' in z else degree
    return e,p,u

def regions(X):
    grip=(X[:,0]<=.3125)|(X[:,0]>=.6875)
    return dict(global_=np.ones(len(X),bool),grip=grip,interior=~grip,deep_interior=(X[:,0]>.375)&(X[:,0]<.625))

def compare(a,b,label='F45',order=None,indicators=False):
    ea,pa,ua=a;eb,pb,ub=b;common=[np.union1d(np.union1d(x,y),[.3125,.375,.625,.6875] if k==0 else []) for k,(x,y) in enumerate(zip(ea,eb))];order=order or max(pa,pb)+1;H=hessian(label);theta={'ISO':0.,'F0':0.,'F45':np.pi/4,'F90':np.pi/2}[label];fiber=np.array([np.cos(theta),np.sin(theta),0.]);totals={k:np.zeros(7) for k in ('global_','grip','interior','deep_interior')};hist=[np.zeros(len(e)-1) for e in eb]
    for X,V in chunks(common,order=order,size=16):
        La,Lb=gradient(X,ea,pa,ua),gradient(X,eb,pb,ub);Pa,Pb=(La.reshape(-1,9)@H.T).reshape(-1,3,3),(Lb.reshape(-1,9)@H.T).reshape(-1,3,3);aa=np.einsum('i,pij,j->p',fiber,La,fiber);ab=np.einsum('i,pij,j->p',fiber,Lb,fiber)
        vals=V[:,None]*np.column_stack((np.sum((Pa-Pb)**2,axis=(1,2)),np.sum(Pb*Pb,axis=(1,2)),(aa-ab)**2,ab*ab,np.sum((La-Lb)**2,axis=(1,2)),np.sum(Lb*Lb,axis=(1,2)),np.ones(len(X))))
        for k,m in regions(X).items():totals[k]+=vals[m].sum(0)
        if indicators:
            for k,e in enumerate(eb):hist[k]+=np.bincount(np.clip(np.searchsorted(e,X[:,k],side='right')-1,0,len(e)-2),weights=vals[:,0],minlength=len(e)-1)
    out={k.rstrip('_'):dict(stress_relative=float(np.sqrt(v[0]/v[1])),stress_absolute_rms_Pa=float(np.sqrt(v[0]/v[6])),fiber_strain_relative=float(np.sqrt(v[2]/v[3])),fiber_strain_absolute_rms=float(np.sqrt(v[2]/v[6])),gradient_relative=float(np.sqrt(v[4]/v[5])),volume=float(v[6]),stress_error_squared=float(v[0])) for k,v in totals.items()}
    return dict(regions=out,quadrature_order=order,stress_passed=all(out[k]['stress_relative']<.02 for k in ('global','grip','interior')),fiber_strain_passed=all(out[k]['fiber_strain_relative']<.02 for k in ('global','grip','interior')),axis_indicators=[a.tolist() for a in hist] if indicators else None)

def mark(edges,indicators,fraction=.125):
    marks=[];out=[]
    for k,(e,hist) in enumerate(zip(edges,indicators)):
        hist=np.array(hist);idx=np.argsort(-hist,kind='stable')[:max(1,int(np.ceil(len(hist)*fraction)))];mandatory=[]
        if k==0:
            for x in (.25,.75):
                j=np.searchsorted(e,x);mandatory.extend([j-1,j])
        else:mandatory=[0,len(e)-2]
        idx=np.union1d(idx,np.clip(mandatory,0,len(e)-2));marks.append(idx.tolist());out.append(np.union1d(e,(e[:-1]+e[1:])[idx]/2))
    return out,marks


def freeze():
    assert not OUT.exists();OUT.mkdir();old=load(BASE/'v20/artifact-sha256.json')
    for name,digest in old.items():assert sha(ROOT/name)==digest,name
    scratch=Path(tempfile.mkdtemp(prefix='mpm-lite-v21-',dir='/dev/shm'));scratch.chmod(0o700)
    from utils.resource_guard import inspect_storage
    assert not inspect_storage(scratch).paused
    write(OUT/'protocol.json',dict(prior_v20_verified=len(old),scratch=str(scratch),geometry='Original physical box and hard grips; linear static same-law tangent, unchanged patch coefficient.',scope='Independent spatial study, no new dynamic claim; no production defaults changed.',reference_plan='Old local Q2/Q3 stress-difference marks top 1/8 axis intervals plus grip/boundary neighbors. Q3/Q4 cross-order on marked mesh. A further Q3 level uses previous Q3 difference marks. Tensor closure is explicit.',gates=dict(reaction=.01,stress=.02,fiber_strain=.02),candidate_plan='Scalar local spaces shared by all three displacement components; select local residual corrections using independently evaluated stress error. Held-out finer reference is never used to construct correction vectors. Compare equal-dimension geometric and error-selected spaces.',source_sha256={n:sha(ROOT/n) for n in ['engine/aniso_phase1/tensor_reference.py','benchmarks/aniso_v21_common.py','benchmarks/aniso_v21_reference.py','tests/test_aniso_v21_reference.py']}))
if __name__=='__main__':freeze()
