"""Offline positive material samples and conditional direction moments.

Selection uses reference positions, unoriented directions and volume only.
No energies/stresses, material parameters, or reference answers enter selection.
No production residual, tangent, or history update is changed here.
"""
import numpy as np
from scipy.spatial import cKDTree
from .material_snapshot import (center_support,moments,moment_points,response,
                               reconstructed_F,particle_response)
from .quadratic import history_polynomial

METHODS=('paired8','paired16','group2x8','group4x8','group8x8')


def partition_material(X,A,w,h,budget,spatial_fallback=True):
    """Weighted binary PCA partitions; direction first, then reference position.

    With no direction variation, grouped moments need no further subdivision.
    Actual-particle representatives continue splitting spatially up to budget.
    Leaf splits use a weighted median; all leaves retain positive volume.
    """
    if budget<1 or h<=0 or len(X)==0 or np.any(w<=0):raise ValueError('positive support, dx and budget required')
    leaves=[np.arange(len(X))]
    while len(leaves)<budget:
        candidates=[]
        for leaf_index,ids in enumerate(leaves):
            if len(ids)<2:continue
            weights=w[ids];wn=weights/weights.sum();a=A[ids].reshape(-1,9);d=a-wn@a
            cov=(d*wn[:,None]).T@d;eig,U=np.linalg.eigh(cov);directional=eig[-1]>1e-12
            if not directional:
                if not spatial_fallback:continue
                d=(X[ids]-wn@X[ids])/h;cov=(d*wn[:,None]).T@d;eig,U=np.linalg.eigh(cov)
                if eig[-1]<1e-16:continue
            axis=U[:,-1]
            # Fix the otherwise arbitrary PCA sign for reproducible ordering.
            axis*=1 if axis[np.argmax(np.abs(axis))]>=0 else -1
            score=d@axis;order=np.argsort(score,kind='stable')
            # Split only between distinct scores, never split identical fibers
            # just to hit a median. This preserves pure direction families.
            cuts=np.flatnonzero(np.diff(score[order])>1e-10)+1
            if not len(cuts):continue
            mass=np.cumsum(weights[order]);cut=int(cuts[np.argmin(abs(mass[cuts-1]-weights.sum()/2))])
            candidates.append((int(directional),float(weights.sum()*eig[-1]),leaf_index,ids[order[:cut]],ids[order[cut:]]))
        if not candidates:break
        best=max(candidates,key=lambda c:(c[0],c[1],-c[2]));i=best[2]
        leaves[i:i+1]=[best[3],best[4]]
    return leaves


def paired_samples(snapshot,ids,w,h,budget):
    leaves=partition_material(snapshot.X[ids],snapshot.A[ids],w,h,budget,True)
    selected=[];weights=[]
    for leaf in leaves:
        wl=w[leaf];wn=wl/wl.sum();X=snapshot.X[ids[leaf]]/h;a=snapshot.A[ids[leaf]].reshape(-1,9)
        distance=np.sum((X-wn@X)**2,axis=1)+np.sum((a-wn@a)**2,axis=1)
        selected.append(ids[leaf[np.argmin(distance)]]);weights.append(wl.sum())
    return np.asarray(selected),np.asarray(weights)


def build_rule(snapshot,ids,w,h,method):
    """Return world positions, reference weights and per-sample A2/A4."""
    if method.startswith('paired'):
        selected,weights=paired_samples(snapshot,ids,w,h,int(method[6:]));A=snapshot.A[selected];a=A.reshape(-1,9)
        return snapshot.x[selected],weights,A,np.einsum('pi,pj->pij',a,a)
    if method not in METHODS:raise ValueError('unknown joint sampling method')
    budget=int(method[5:method.index('x')]);leaves=partition_material(snapshot.X[ids],snapshot.A[ids],w,h,budget,False)
    positions=[];weights=[];a2=[];a4=[]
    for leaf in leaves:
        wl=w[leaf];A2,A4=moments(snapshot.A[ids[leaf]],wl)
        positions.extend(moment_points(snapshot.x[ids[leaf]],wl));weights.extend([wl.sum()/8]*8)
        a2.extend([A2]*8);a4.extend([A4]*8)
    return np.array(positions),np.array(weights),np.array(a2),np.array(a4)


def compare_joint(snapshot,h,params,methods=METHODS):
    groups=center_support(snapshot,h);tree=cKDTree(snapshot.x);ep,pp,tp=particle_response(snapshot,params)
    records={name:[] for name in methods};counts={name:[] for name in methods};full={name:0 for name in methods};details=[]
    referenceP=[];referenceT=[];volumes=[];rule_vol_error=0.;min_weight=float('inf')
    for key,ids,w in groups:
        V=w.sum();center=(key+.5)*h;volumes.append(V)
        referenceP.append(np.einsum('p,pij->ij',w/V,pp[ids]));referenceT.append(np.einsum('p,pij->ij',w/V,tp[ids]))
        coef,_=history_polynomial(snapshot.x,snapshot.X,snapshot.F,snapshot.volume,center,h,tree)
        for method in methods:
            points,W,A2,A4=build_rule(snapshot,ids,w,h,method);F=reconstructed_F(coef,points,center,h)
            E,P,tau=response(F,A2,A4,params);wn=W/V
            energy=float(W@E);p=np.einsum('p,pij->ij',wn,P);t=np.einsum('p,pij->ij',wn,tau)
            records[method].append((energy,p,t));counts[method].append(len(W))
            if method.startswith('paired') and len(W)==len(ids):full[method]+=1
            rule_vol_error=max(rule_vol_error,float(abs(W.sum()-V)));min_weight=min(min_weight,float(W.min()))
            detail=dict(method=method,i=int(key[0]),j=int(key[1]),k=int(key[2]),volume=V,support_particles=len(ids),samples=len(W),energy=energy)
            detail.update({f'P_{i}{j}':float(p[i,j]) for i in range(3) for j in range(3)});details.append(detail)
    V=np.array(volumes);refP=np.array(referenceP);refT=np.array(referenceT);Uref=float(snapshot.volume@ep);out={}
    Pden=float(np.einsum('p,pij,pij->',V,refP,refP));Tden=float(np.einsum('p,pij,pij->',V,refT,refT))
    for method,rows in records.items():
        E=sum(r[0] for r in rows);P=np.array([r[1] for r in rows]);T=np.array([r[2] for r in rows])
        out[method]=dict(energy=E,energy_relative_error=(E-Uref)/max(abs(Uref),1e-20),
            center_P_rms_relative_error=float(np.sqrt(np.einsum('p,pij,pij->',V,P-refP,P-refP)/max(Pden,1e-30))),
            center_tau_rms_relative_error=float(np.sqrt(np.einsum('p,pij,pij->',V,T-refT,T-refT)/max(Tden,1e-30))),
            integrated_P=np.einsum('p,pij->ij',V,P).tolist(),integrated_tau=np.einsum('p,pij->ij',V,T).tolist(),
            material_evaluations=sum(counts[method]),min_samples_per_center=min(counts[method]),max_samples_per_center=max(counts[method]),
            full_particle_support_centers=full[method],evaluations_per_particle=sum(counts[method])/len(snapshot.x))
    return dict(methods=out,reference_energy=Uref,particles=len(snapshot.x),centers=len(groups),
        max_rule_volume_error=rule_vol_error,min_sample_weight=min_weight,particle_support_entries=sum(len(ids) for _,ids,_ in groups)),details
