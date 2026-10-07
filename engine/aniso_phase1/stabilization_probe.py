"""Independent beam reference and small-strain enhancement probes."""
import numpy as np
import warp as wp
from scipy.sparse.linalg import spsolve
from engine.types import vec3
from engine.sp_grid import B
from .beam_reference import beam_matrices
from .solver import AnisotropicLiteImplicitSolver
from .types import AnisotropicMaterialParams


def bent_beam_state(X,amplitude=.002):
    x=X[:,0]-.25;y=X[:,1]-.5
    w=-amplitude*(x/.5)**2;dw=-2*amplitude*x/.5**2;ddw=-2*amplitude/.5**2
    u=np.zeros_like(X);u[:,0]=-y*dw;u[:,1]=w
    F=np.tile(np.eye(3),(len(X),1,1));F[:,0,0]-=y*ddw;F[:,0,1]=-dw;F[:,1,0]=dw
    return X+u,F


def node_coordinates(s):
    n=int(s.n_active_nodes.numpy()[0]);ndof=s.ndof2bijk[:n].numpy();blocks=s.block_xyz_by_id.numpy()
    return np.array([blocks[b]*B+np.array([l//(B*B),(l//B)%B,l%B]) for b,l in ndof])


def static_audit(grid=17,device='cpu',include_corotated=False):
    nodes,centers,free,tip,force,matrices=beam_matrices(grid)
    scalar=beam_matrices(grid,tangent=20*np.eye(9))[-1]
    variants={'none':matrices[0],'supplemental':matrices[1], 'hourglass':matrices[0]+scalar[1]-scalar[0]}
    if include_corotated:variants['corotated']=matrices[1]
    results=[]
    reference=None
    for mode in (('supplemental','hourglass','none','corotated') if include_corotated else ('supplemental','hourglass','none')):
        K=variants[mode];eig=np.linalg.eigvalsh(K[free][:,free].toarray());threshold=1e-9*max(eig[-1],1.)
        r=dict(grid=grid,stabilization=mode,soft_modes=int(np.sum(eig<threshold)),min_eigenvalue=float(eig[0]))
        if r['soft_modes']==0:
            u=np.zeros(len(force));u[free]=spsolve(K[free][:,free],force[free]);disp=float(np.mean(u[3*tip+1]))
            if reference is None:reference=disp
            r.update(tip_displacement=disp,relative_to_full=abs(disp/reference-1))
        s=AnisotropicLiteImplicitSolver((grid,)*3,AnisotropicMaterialParams(10,20,200),dx=1/(grid-1),device=device,gravity=0,stabilization=mode)
        s.seed_particles(centers,density=1.,vol0=s.dx**3)
        fixed=nodes[nodes[:,0]==nodes[:,0].min()]
        s.paint_boundary(fixed,np.ones(len(fixed),dtype=np.int32));s.set_dt(.01);s.step(max_iters=0,print_every=0);s.evaluate_residual()
        actual=node_coordinates(s);lookup={tuple(c):i for i,c in enumerate(nodes)};perm=np.array([lookup[tuple(c)] for c in actual]);n=len(actual)
        p=np.random.default_rng(4).normal(size=(len(nodes),3));p[nodes[:,0]==nodes[:,0].min()]=0
        direction=wp.zeros_like(s.node_residual);out=wp.zeros_like(direction)
        wp.copy(direction,wp.array(p[perm],dtype=vec3,device=device),count=n);s.apply_tangent(direction,out)
        ndof=s.ndof2bijk[:n].numpy();m=s.grid_m[:int(s.bcn)].numpy()
        masses=np.array([m[b,l//(B*B),(l//B)%B,l%B] for b,l in ndof])
        result=(out[:n].numpy()-masses[:,None]*p[perm])/s.dt**2
        expected=(K@p.ravel()).reshape(-1,3)[perm];expected[actual[:,0]==nodes[:,0].min()]=0
        r['production_matvec_relative_error']=float(np.linalg.norm(result-expected)/np.linalg.norm(expected))
        results.append(r)
        del s
    return results
