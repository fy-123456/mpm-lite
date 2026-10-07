"""Static gate evidence, phase windows, and inherited formal-boundary audit."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy import sparse
from engine.aniso_phase1.research_common_kinematics_next.model import CommonBridge,CommonOperator,Space
from engine.aniso_phase1.research_common_kinematics_next.mapping import push
from engine.aniso_phase1.research_common_kinematics_next.checkpoint import load,save
from engine.aniso_phase1.research_unified_lite_poro.solve import advance
from .replay import array_inventory
from .run import write
ROOT=Path(__file__).resolve().parents[2]

def phase(run):
    # Start at the same actual committed t=.045 state; only refine its next window.
    # A full checkpoint at unload start is emitted by the driver. Never fabricate
    # material/velocity state from frame positions alone.
    b=load(run/'S3/cycle24/checkpoint-unload.npz')
    rows=[];signals=[]
    for label,dt,steps in [('coarse',.0025,4),('fine',.00125,8),('finer',.000625,16)]:
        case=load(run/'S3/cycle24/checkpoint-unload.npz');times=[];u=[];v=[];p=[];metrics=[]
        for i in range(steps):
            m=case.step(dt);metrics.append(m)
            if (i+1)%(steps//4)==0:
                times.append(case.state.time);u.append(case.state.qx-b.space.rule()[0]);v.append(case.state.qv.copy());p.append(case.state.p.copy())
        record=dict(label=label,dt=dt,steps=steps,start=.045,end=case.state.time,metrics=metrics)
        write(run/f'S3/phase-{label}.json',record);signals.append((np.array(u),np.array(v),np.array(p)))
    for i in range(2):
        entry={}
        for name,a,c,floor in zip(('displacement','velocity','pressure'),signals[i],signals[i+1],(1e-6,1e-5,1e-4)):
            absolute=float(np.sqrt(np.mean((a-c)**2)));den=max(float(np.sqrt(np.mean(c*c))),floor)
            entry[name]=dict(rms_absolute=absolute,reference_rms=den,relative=absolute/den)
        rows.append(entry)
    write(run/'S3/phase-comparison.json',dict(window=[.045,.055],same_checkpoint=True,pairs=rows,
        temporal_accuracy=False,scope='local step sensitivity from same coarse-history checkpoint; not full refined-cycle certification'))


def quadrature(run):
    with np.load(run/'S3/cycle24/frames.npz') as f:increments=f['increments'].copy()
    space=Space();rows=[]
    for order in (4,5,6):
        X,w,c=space.rule(order);x=X.copy();F=np.tile(np.eye(3),(len(X),1,1))
        for d in increments:x,F=push(space,X,x,F,d)
        angle=np.deg2rad(20+50*X[:,1]/.25+15*X[:,0]);a=np.c_[np.cos(angle),np.sin(angle),np.zeros(len(X))]
        A=a[:,:,None]*a[:,None,:];flat=A.reshape(-1,9);A4=flat[:,:,None]*flat[:,None,:]
        op=CommonOperator(space,x,F,A,A4,order);zero=np.zeros((10,3));E,force,P=op.material(zero)
        V,G,H,J=op.geometry(zero)
        rows.append(dict(order=order,points=len(X),energy=E,force_norm=float(np.linalg.norm(force)),
            force=force.tolist(),volume=V.tolist(),stress_rms=float(np.sqrt(np.einsum('q,qij,qij->',w,P,P)/w.sum())),minJ=J))
    pairs=[]
    for a,b in zip(rows,rows[1:]):
        pairs.append(dict(orders=[a['order'],b['order']],energy_relative=abs(a['energy']-b['energy'])/max(abs(b['energy']),1e-12),
            force_relative=float(np.linalg.norm(np.array(a['force'])-b['force'])/max(b['force_norm'],1e-12)),
            volume_max_absolute=float(np.max(abs(np.array(a['volume'])-b['volume'])))))
    write(run/'S1/quadrature-diagnostic.json',dict(rows=rows,pairs=pairs,
        scope='same evolved mapping, analytic registered planar director, no independent spatial solution',
        quadrature_certified=False))


def scaling(run):
    rows=[]
    for ppc in (2,3,4,8):
        b=CommonBridge(ppc=ppc);op,v,fit=b.prepare();inventory=array_inventory(op)
        op.material(np.zeros_like(v));op.geometry(np.zeros_like(v))
        rows.append(dict(particles=len(b.state.particles.X),Nq=len(op.points),free_vector=v.size,
            array_bytes=sum(r['bytes'] for r in inventory),material_evaluations=op.material_evaluations,
            geometry_calls=op.geometry_calls,prepare_seconds=fit['seconds'],inner_particle_reads=0))
    write(run/'S2/scaling.json',dict(rows=rows,scope='fixed active modal space: per-call scale only; no end-to-end speedup claim'))


def formal(run):
    from engine.aniso_phase1.tensor_metrics import sampling
    from engine.aniso_phase1.tensor_reference import coordinates
    folder=next((ROOT/'docs/results/cross-direction').glob('*/S1/candidates/cross-direction-snapshot6'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=folder/'space-package.json';expected='885a2870a9908f5fd5a0904173a5c1244695e4b7e5a80bf953ff445fda644a30'
    if sha(manifest)!=expected:raise ValueError('wrong formal144 package')
    cache=folder/'qualified-array-cache';meta=json.loads((cache/'cache.json').read_text())
    if meta['source_package_sha256']!=expected:raise ValueError('foreign cache')
    for name,h in meta['files'].items():
        if sha(cache/name)!=h:raise ValueError('cache bytes changed')
    with np.load(cache/'arrays.npz') as z:
        oldA=z['oldA'];edges=[z['oldedge'+str(i)] for i in range(3)]
        carrier=z['carrier_X'];M=z['M'];Ks=z['Ks'];free=z['free_scalar_ids'];fixed=z['fixed_scalar_ids']
    n=len(carrier);ids=free[free<n];rows=[];dx=1/64
    def values(axes):
        matrices=[sampling(e,2,a) for e,a in zip(edges,axes)]
        T=sparse.kron(sparse.kron(matrices[0],matrices[1]),matrices[2])
        return T@oldA
    for point in ([.25,.5,.5],[.75,.5,.5]):
        point=np.array(point);old=values([np.array([v]) for v in point])[0]
        base=np.floor(point/dx-.5).astype(int);f=point/dx-.5-base;new=np.zeros(n)
        for c in np.ndindex(2,2,2):
            corner=np.array(c);weight=np.prod(np.where(corner,f,1-f))
            nodes=[(base[a]+corner[a]+np.arange(2))*dx for a in range(3)]
            new+=weight*values(nodes).mean(axis=0)
        rows.append(dict(point=point.tolist(),old_free_carrier_max=float(abs(old[ids]).max()),
            naive_SH_free_carrier_max=float(abs(new[ids]).max()),basis_change_norm=float(np.linalg.norm(new-old))))
    fullfree=M[np.ix_(free,free)];diag=np.sqrt(np.diag(fullfree));ev=np.linalg.eigvalsh(fullfree/diag[:,None]/diag[None,:])
    # One targeted constrained-carrier candidate: C1 material mask, not adopted.
    # It changes the physical basis and therefore needs NEW M and stabilization
    # checks. Zeroing particle positions afterward would not be an equivalent fix.
    def smooth(t):
        z=np.clip(t,0.,1.);return z*z*(3-2*z),6*z*(1-z) if 0<t<1 else 0.
    def masked(point):
        point=np.array(point);base=np.floor(point/dx-.5).astype(int);f=point/dx-.5-base
        value=np.zeros(n);gradient=np.zeros((n,3))
        for c in np.ndindex(2,2,2):
            corner=np.array(c);factors=np.where(corner,f,1-f)
            nodal=values([(base[a]+corner[a]+np.arange(2))*dx for a in range(3)]).mean(axis=0)
            value+=np.prod(factors)*nodal
            for a in range(3):gradient[:,a]+=(2*corner[a]-1)*np.prod(np.delete(factors,a))/dx*nodal
        width=2*dx;l,dl=smooth((point[0]-.25)/width);r,dr=smooth((.75-point[0])/width)
        mask=l*r;gmask=(dl*r-l*dr)/width
        gradient=mask*gradient;gradient[:,0]+=gmask*value
        return mask*value,gradient
    point=np.array([.260,.503,.497]);mv,mg=masked(point);fd_errors=[]
    for eps in (1e-5,1e-6,1e-7):
        derivative=np.column_stack([(masked(point+np.eye(3)[a]*eps)[0]-masked(point-np.eye(3)[a]*eps)[0])/(2*eps) for a in range(3)])
        fd_errors.append(float(abs(derivative[ids]-mg[ids]).max()))
    mask_boundary=max(float(abs(masked(np.array(p))[0][ids]).max()) for p in ([.25,.5,.5],[.75,.5,.5]))
    write(run/'S3/formal144-entry-audit.json',dict(package=str(manifest),sha256=expected,
        name='cross-direction-snapshot6',local_functions=144,carrier_scalar=n,full_scalar=M.shape[0],
        free_scalar_before_condensation=len(free),mass_order=7,material_order=7,
        scaled_nullity=int(np.count_nonzero(ev<1e-10)),old_mass_cross_norm=float(np.linalg.norm(M[:n,n:])),
        old_stabilization_norm=float(np.linalg.norm(Ks)),boundary_probes=rows,
        targeted_mask_candidate=dict(boundary_free_max=mask_boundary,gradient_absolute_errors=fd_errors,adopted=False,reason="changes carrier basis, needs fresh full mass and stabilization qualification"),
        common_map_adopted=False,reason='naive SH resampling changes rigid-grip constraints; old M7 and condensation cannot be certified for the new map',
        next_action='derive and test constrained SH carrier/lift with zero free displacement on rigid volumes, then reassemble full inertia; no silent boundary clamping'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--task',choices=['phase','quadrature','scaling','formal'],required=True)
    a=p.parse_args();globals()[a.task](a.run)
