"""Experimental complete-quadratic reference/velocity reconstruction.

All fits happen BETWEEN time steps on the host. Newton/PCG use the existing
GPU corotated energy, exact derivatives and GN fallback with frozen arrays.
This first implementation prioritizes a reviewable discretization over speed.
No particle work, fitting or moving support occurs inside a nonlinear solve.
"""
import itertools,time
import numpy as np
from scipy.spatial import cKDTree
from engine.sp_grid import B

MODES=('quadratic','material_quadratic')
GAUSS=np.array([(0.,0.,0.)]+list(itertools.product((-1/np.sqrt(12),1/np.sqrt(12)),repeat=3)))


def polynomial(x,enriched=False):
    """Complete P2, optionally enriched by xyz to retain the Q1 corner mode."""
    x,y,z=np.asarray(x).T;n=len(x)
    P=np.array([np.ones(n),x,y,z,.5*x*x,.5*y*y,.5*z*z,x*y,x*z,y*z]+([x*y*z] if enriched else [])).T
    G=np.zeros((n,3,P.shape[1]));G[:,0,1]=1;G[:,1,2]=1;G[:,2,3]=1
    G[:,0,4]=x;G[:,1,5]=y;G[:,2,6]=z
    G[:,0,7]=y;G[:,1,7]=x;G[:,0,8]=z;G[:,2,8]=x;G[:,1,9]=z;G[:,2,9]=y
    if enriched:G[:,0,10]=y*z;G[:,1,10]=x*z;G[:,2,10]=x*y
    return P,G


def weighted_inverse(P,w):
    root=np.sqrt(w);U,s,Vt=np.linalg.svd(root[:,None]*P,full_matrices=False)
    if len(s)<P.shape[1] or s[-1]<=1e-10*s[0]:raise ValueError('quadratic patch is rank deficient')
    return (Vt.T/s)@U.T*root[None,:],float(s[0]/s[-1])


def node_patch_basis(x,center,h,tree,frame=None,queries=None):
    """Values and derivatives of the SAME frozen complete P2+xyz patch."""
    R=np.eye(3) if frame is None else frame
    for radius in (2.1,2.6,3.5):
        ids=np.array(sorted(tree.query_ball_point(center,radius*h)),dtype=int)
        z=(x[ids]-center)@R/h
        P,_=polynomial(z,True);w=np.exp(-np.sum(z*z,axis=1)/2)
        try:inverse,condition=weighted_inverse(P,w)
        except ValueError:continue
        basis,derivative=polynomial(GAUSS if queries is None else (np.asarray(queries)-center)@R/h,True)
        gradients=np.einsum('qdk,kn->qnd',derivative,inverse)/h
        gradients=gradients@R.T
        return ids,basis@inverse,gradients,condition,radius
    raise ValueError('no full-rank quadratic nodal patch within 3.5 cells')


def node_patch(x,center,h,tree,frame=None,queries=None):
    """Frozen nodal MLS derivative; preserve the production API and map."""
    ids,_,gradients,condition,radius=node_patch_basis(x,center,h,tree,frame,queries)
    return ids,gradients,condition,radius


def history_polynomial(x,X,F,volume,center,h,tree):
    """Hermite P2 fit using reference positions AND inverse-F derivatives."""
    ids=np.asarray(tree.query_ball_point(center,2.1*h),dtype=int)
    if len(ids)<8:ids=np.atleast_1d(tree.query(center,k=min(16,len(x)))[1])
    if len(ids)>128:ids=ids[np.argsort(np.linalg.norm(x[ids]-center,axis=1))[:128]]
    z=(x[ids]-center)/h;P,G=polynomial(z)
    weight=np.exp(-np.sum(z*z,axis=1)/2)*volume[ids]/np.mean(volume[ids])
    design=np.concatenate((P,G.reshape(-1,10)))
    w=np.r_[weight,np.repeat(weight,3)]
    # Fit X-x: affine rigid motion is reproduced without requiring a special case.
    inverseF=np.linalg.inv(F[ids])
    target=np.concatenate((X[ids]-x[ids],h*(inverseF-np.eye(3)).transpose(0,2,1).reshape(-1,3)))
    pinv,condition=weighted_inverse(design,w)
    return pinv@target,condition


def mapped_samples(coefficients,h,frame):
    query=GAUSS@frame.T
    _,G=polynomial(query)
    JX=np.eye(3)+np.einsum('qdk,km->qmd',G,coefficients)/h
    if not np.isfinite(JX).all():raise ValueError('quadratic reference mapping is nonfinite')
    singular=np.linalg.svd(JX,compute_uv=False)
    if np.any(np.linalg.det(JX)<=0) or singular.min()<1e-8:
        raise ValueError('quadratic reference mapping is inverted or singular')
    return np.linalg.inv(JX)


def build_patch_maps(s):
    """Build compact active-center maps and wider stencils; no solver mutation."""
    from .stabilization_probe import node_coordinates
    start=time.perf_counter();h=s.dx;nodes=node_coordinates(s)*h
    nc=int(s.n_active_centers.numpy()[0]);cd=s.cdof2bijk[:nc].numpy();blocks=s.block_xyz_by_id.numpy()
    l=cd[:,1];centers=(blocks[cd[:,0]]*B+np.stack((l//(B*B),(l//B)%B,l%B),axis=1)+.5)*h
    x,X,F,volume=(a.numpy() for a in (s.ptc_x,s.ptc_reference_x,s.ptc_F,s.ptc_vol0))
    ntree,ptree=cKDTree(nodes),cKDTree(x);patches=[];states=[];maps=[]
    node_condition=0.;history_condition=0.;expansions=0;affine_error=0.
    for center in centers:
        coef,condition=history_polynomial(x,X,F,volume,center,h,ptree);history_condition=max(history_condition,condition)
        frame=np.eye(3)
        if s.stabilization=='material_quadratic':
            Fcenter=mapped_samples(coef,h,frame)[0]
            U,_,Vt=np.linalg.svd(Fcenter);frame=U@Vt
        F0=mapped_samples(coef,h,frame)
        ids,g,condition,radius=node_patch(nodes,center,h,ntree,frame)
        node_condition=max(node_condition,condition);expansions+=radius>2.1
        affine_error=max(affine_error,float(np.max(np.abs(np.einsum('ni,qnj->qij',nodes[ids]-center,g)-np.eye(3)))))
        # g_X = F0^T g_x; F0 + dt sum(v tensor g_X) is affine in velocity.
        refg=np.einsum('qji,qnj->qni',F0,g)
        patches.append(ids);states.append(F0);maps.append(refg)
    width=max(map(len,patches));ids=np.zeros((nc,width),dtype=np.int32);gradients=np.zeros((nc,9,width,3))
    for c,(indices,g) in enumerate(zip(patches,maps)):
        ids[c,:len(indices)]=indices;gradients[c,:,:len(indices)]=g
    if affine_error>1e-7:raise ValueError('quadratic derivative lost affine reproduction')
    stats=dict(reconstruction_seconds=time.perf_counter()-start,reconstruction_nodes=width,
        reconstruction_expanded_patches=int(expansions),reconstruction_node_condition=node_condition,
        reconstruction_history_condition=history_condition,reconstruction_affine_error=affine_error)
    return ids,np.array(states),gradients,stats
