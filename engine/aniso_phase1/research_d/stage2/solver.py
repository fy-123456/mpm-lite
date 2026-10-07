"""Curvature-aware linear solves and rollback-safe static equilibration.

The frozen search matrix is an iteration aid only. All accepted iterates and
convergence decisions use the original backend energy and complete gradient.
"""
import time
import numpy as np
import scipy.linalg as la
from ..cpu import pcg


def curvature(matrix,*,floor=1e-12):
    a=np.asarray(matrix,dtype=float)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.isfinite(a).all():raise ValueError('finite square operator required')
    symmetry=float(la.norm(a-a.T)/max(la.norm(a),1e-30))
    if symmetry>1e-9:return dict(kind='general',symmetry=symmetry,positive_definite=False)
    eig=la.eigvalsh((a+a.T)/2);threshold=floor*max(abs(eig).max(),1.)
    kind='spd' if eig[0]>threshold else ('indefinite' if eig[0]<-threshold else 'near_singular')
    return dict(kind=kind,symmetry=symmetry,positive_definite=kind=='spd',minimum=float(eig[0]),maximum=float(eig[-1]),near_zero=int(np.sum(abs(eig)<=threshold)),negative=int(np.sum(eig < -threshold)))


def preconditioner(matrix,name,carrier_components):
    a=np.asarray(matrix);n=len(a);t=time.perf_counter()
    if name=='none':apply=lambda x:x.copy();size=0
    elif name=='diagonal':
        d=np.diag(a).copy()
        if np.any(d<=0):raise ValueError('nonpositive diagonal')
        apply=lambda x:x/d;size=d.nbytes
    else:
        if name=='carrier_local_block':blocks=[np.arange(carrier_components),np.arange(carrier_components,n)]
        elif name=='overlap_block':
            # Shared carrier block with overlapping consecutive local coordinates.
            local=np.arange(carrier_components,n);blocks=[np.r_[np.arange(carrier_components),local[k:k+144]] for k in range(0,len(local),96)]
        else:raise ValueError('unknown preconditioner')
        counts=np.zeros(n)
        for ids in blocks:counts[ids]+=1
        if np.any(counts==0):raise ValueError('incomplete block coverage')
        parts=[(ids,1/np.sqrt(counts[ids]),la.cho_factor(a[np.ix_(ids,ids)],lower=True)) for ids in blocks]
        def apply(x):
            y=np.zeros_like(x)
            for ids,w,f in parts:y[ids]+=w*la.cho_solve(f,w*x[ids])
            return y
        size=sum(f[0].nbytes+ids.nbytes+w.nbytes for ids,w,f in parts)
    return apply,dict(name=name,build_seconds=time.perf_counter()-t,bytes=size)


def solve_linear(matrix,b,*,name='none',carrier_components=0,rtol=1e-8,maxiter=800):
    t=time.perf_counter();a=np.asarray(matrix);b=np.asarray(b);info=curvature(a)
    if info['positive_definite']:
        M,build=preconditioner(a,name,carrier_components)
        x,record=pcg(lambda v:a@v,b,M,rtol=rtol,atol=1e-12,maxiter=maxiter)
        result=record.record();result['method']='pcg';result['preconditioner']=build
    else:
        # Rank-revealing SVD handles both indefinite and incompatible singular
        # systems; a least-squares minimizer is never automatically convergence.
        x,_,rank,_=la.lstsq(a,b,lapack_driver='gelsd')
        r=float(la.norm(a@x-b));target=max(1e-12,rtol*la.norm(b))
        result=dict(method='svd_lstsq',rank=int(rank),true_residual=r,target=target,
                    converged=bool(r<=target),status='converged' if r<=target else 'incompatible_or_rank_deficient')
    result.update(curvature=info,total_seconds=time.perf_counter()-t)
    return x,result


def equilibrate(evaluate,start,search_matrix,*,tolerance=1e-8,max_iterations=30,max_seconds=1800,checkpoint=None):
    """Frozen-tangent chord iteration with original-energy Armijo globalization.

    Uses dense direct search solves (not PCG). It works with indefinite search
    matrices and falls back to steepest descent on a non-descent direction.
    No search matrix term is added to the physical equation or potential.
    """
    begun=time.perf_counter();q=np.array(start,dtype=float,copy=True);shape=q.shape
    h=np.array(search_matrix,copy=True);c=curvature(h);rejections=[];trace=[];crossings={}
    if c['positive_definite']:
        factor=la.cho_factor(h,lower=True);inverse=lambda g:la.cho_solve(factor,g)
    else:inverse=lambda g:la.lstsq(h,g)[0]
    r=evaluate(q);status='iteration_limit'
    for iteration in range(max_iterations+1):
        g=r['force'].ravel();residual=float(la.norm(g))
        row=dict(iteration=iteration,energy=float(r['U']),residual_N=residual,min_detF=float(r['min_detF']),seconds=time.perf_counter()-begun)
        trace.append(row)
        for threshold in (1e-6,1e-8):
            if residual<=threshold and str(threshold) not in crossings:crossings[str(threshold)]=iteration
        if checkpoint:checkpoint(q,trace,rejections)
        if residual<=tolerance:status='converged';break
        if row['seconds']>=max_seconds:status='time_limit';break
        if iteration==max_iterations:break
        d=-inverse(g);slope=float(g@d)
        if not np.isfinite(d).all() or slope>=0:
            d=-g/max(float(la.norm(h,2)),1.);slope=float(g@d);row['fallback']='steepest_descent'
        accepted=False
        for backtrack in range(24):
            alpha=2.**(-backtrack);trial=q+alpha*d.reshape(shape)
            try:
                candidate=evaluate(trial)
                valid=np.isfinite(candidate['U']) and np.isfinite(candidate['force']).all() and candidate['min_detF']>0
                # Tiny roundoff allowance is fixed independently of a trial.
                descent=candidate['U']<=r['U']+1e-4*alpha*slope+1e-16
                if valid and descent:
                    accepted=True;break
                reason='energy_increase_or_nonfinite'
            except ValueError as exc:reason=str(exc)
            rejections.append(dict(iteration=iteration,alpha=alpha,reason=reason,rolled_back=True))
        if not accepted:status='line_search_failed';break
        q=trial;r=candidate;row['alpha']=alpha
    return q,r,dict(status=status,converged=status=='converged',trace=trace,rejections=rejections,
                    threshold_crossings=crossings,search_matrix_curvature=c,seconds=time.perf_counter()-begun,
                    physical_regularization=False,method='frozen exact initial tangent + Armijo')
