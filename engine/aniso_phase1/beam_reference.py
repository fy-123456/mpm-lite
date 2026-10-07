"""Independent small-strain Q1 FEM beam, with full/reduced quadrature audits.

Uses the tangent of the SAME Hencky + quadratic fiber law at F=I. Not a
finite-strain reference. Mass is excluded from all static rank checks.
"""
import itertools
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def reference_hessian(mu=10.,lam=20.,kf=200.,direction=(1.,0.,0.)):
    a=np.asarray(direction,dtype=float);a/=np.linalg.norm(a);A=np.outer(a,a)
    H=np.empty((9,9))
    for j,dF in enumerate(np.eye(9).reshape(9,3,3)):
        H[:,j]=(mu*(dF+dF.T)+lam*np.trace(dF)*np.eye(3)+4*kf*np.sum(A*dF)*A).ravel()
    return H


def shape_gradients(q,dx):
    corners=np.array(list(itertools.product((0,1),repeat=3)))
    gradients=np.empty((8,3))
    for i,c in enumerate(corners):
        for d in range(3):
            other=[j for j in range(3) if j!=d]
            gradients[i,d]=(1 if c[d] else -1)*np.prod([q[j] if c[j] else 1-q[j] for j in other])/dx
    return gradients


def element_stiffness(dx,H,full=True):
    coords=(.5-1/(2*np.sqrt(3)),.5+1/(2*np.sqrt(3))) if full else (.5,)
    K=np.zeros((24,24))
    for q in itertools.product(coords,repeat=3):
        grad=shape_gradients(q,dx)
        mapping=np.zeros((9,24))
        for n in range(8):
            for d in range(3):mapping[d*3:d*3+3,n*3+d]=grad[n]
        K+=dx**3/len(coords)**3*(mapping.T@H@mapping)
    return K


def beam_matrices(grid=17,kf=200.,tangent=None):
    dx=1/(grid-1);counts=np.array([round(.5/dx),round(.125/dx),round(.125/dx)])
    shape=tuple(counts+1);start=np.array([round(.25/dx),round(.4375/dx),round(.4375/dx)])
    local=np.array(list(itertools.product(*[range(n) for n in shape])))
    nodes=local+start;corners=np.array(list(itertools.product((0,1),repeat=3)))
    rows=[];cols=[];values=[[],[]];centers=[]
    H=reference_hessian(kf=kf) if tangent is None else tangent
    elements=[element_stiffness(dx,H,False),element_stiffness(dx,H,True)]
    for cell in itertools.product(*[range(n) for n in counts]):
        vertices=np.array(cell)+corners;ids=np.ravel_multi_index(vertices.T,shape)
        dofs=(3*ids[:,None]+np.arange(3)).ravel();rows.extend(np.repeat(dofs,24));cols.extend(np.tile(dofs,24))
        for data,Ke in zip(values,elements):data.extend(Ke.ravel())
        centers.append((np.array(cell)+start+.5)*dx)
    size=3*len(nodes)
    matrices=[sp.coo_matrix((v,(rows,cols)),shape=(size,size)).tocsr() for v in values]
    free=np.flatnonzero(np.repeat(local[:,0]>0,3))
    tip=np.flatnonzero(local[:,0]==counts[0])
    force=np.zeros(size);force[3*tip+1]=-1e-4/len(tip)
    return nodes,np.array(centers),free,tip,force,matrices


def beam_audit(grid=17,kf=200.,device=None):
    nodes,centers,free,tip,force,matrices=beam_matrices(grid,kf)
    result=dict(grid=grid,kf=kf,free_dofs=len(free),tip_force=-1e-4,reference='linearized same-law full-integration Q1 FEM')
    for name,K in zip(('center','full'),matrices):
        reduced=K[free][:,free].toarray();eigen,vectors=np.linalg.eigh(reduced)
        threshold=1e-9*max(eigen[-1],1.)
        null=eigen<=threshold
        result[name+'_soft_modes']=int(np.sum(null));result[name+'_min_eigenvalue']=float(eigen[0])
        result[name+'_max_eigenvalue']=float(eigen[-1])
        null_force=float(np.linalg.norm(vectors[:,null].T@force[free])/np.linalg.norm(force[free])) if np.any(null) else 0.
        result[name+'_load_in_soft_space']=null_force
        if not np.any(null):
            u=np.zeros(len(force));u[free]=spsolve(K[free][:,free],force[free])
            result[name+'_tip_displacement']=float(np.mean(u[3*tip+1]));result[name+'_strain_energy']=float(.5*u@K@u)
    # Blend study is diagnostic only, not silently added to the production solver.
    for eta in (.05,.2):
        K=(1-eta)*matrices[0]+eta*matrices[1]
        u=np.zeros(len(force));u[free]=spsolve(K[free][:,free],force[free])
        result[f'blend_{eta}_tip_displacement']=float(np.mean(u[3*tip+1]))
    if device is not None:
        import warp as wp
        from engine.types import vec3
        from .solver import AnisotropicLiteImplicitSolver
        from .types import AnisotropicMaterialParams
        from engine.sp_grid import B
        s=AnisotropicLiteImplicitSolver((grid,)*3,AnisotropicMaterialParams(10.,20.,kf),dx=1/(grid-1),gravity=0,device=device)
        s.seed_particles(centers,density=1.,vol0=s.dx**3)
        fixed=nodes[nodes[:,0]==nodes[:,0].min()]
        s.paint_boundary(fixed,np.ones(len(fixed),dtype=np.int32));s.set_dt(.01);s.step(max_iters=0,print_every=0);s.evaluate_residual()
        n=int(s.n_active_nodes.numpy()[0]);ndof=s.ndof2bijk[:n].numpy();blocks=s.block_xyz_by_id.numpy()
        actual=np.array([blocks[b]*B+np.array([l//(B*B),(l//B)%B,l%B]) for b,l in ndof])
        lookup={tuple(c):i for i,c in enumerate(nodes)};permutation=np.array([lookup[tuple(c)] for c in actual])
        p=np.random.default_rng(4).normal(size=(len(nodes),3));p[nodes[:,0]==nodes[:,0].min()]=0
        direction=wp.zeros_like(s.node_residual);out=wp.zeros_like(direction)
        wp.copy(direction,wp.array(p[permutation],dtype=vec3,device=device),count=n);s.apply_tangent(direction,out)
        masses=s.grid_m[:int(s.bcn)].numpy()
        mass=np.array([masses[b,l//(B*B),(l//B)%B,l%B] for b,l in ndof])
        Kv=(out[:n].numpy()-mass[:,None]*p[permutation])/s.dt**2
        target=(matrices[0]@p.ravel()).reshape(-1,3)[permutation]
        target[actual[:,0]==nodes[:,0].min()]=0
        result['production_stiffness_relative_error']=float(np.linalg.norm(Kv-target)/np.linalg.norm(target))
    return result
