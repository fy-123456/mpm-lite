"""Read-only probes of the production Lite transfer versus global MLS.

No production solver rule is changed. A prescribed grid velocity is passed
through the real GPU G2C/C2P kernels, isolating kinematics from Newton error.
"""
import time
import numpy as np
import warp as wp
from scipy.spatial import cKDTree
from scipy.linalg import null_space,eigvalsh
from .aligned_quadrature import evaluate
from .trace_probe import face_points,lite_values
from .quadratic import history_polynomial
from .material_snapshot import reconstructed_F
from .diagnostics import energy_density
from .operator_probe import assign_velocity
from engine.kernel.d3.kernel_lite import lite_g2c_kernel,lite_c2p_kernel
from engine.sp_grid import B
from engine.types import vec3,mat33


def transfer(s,positions,F,velocity,dt):
    """Actual node order and sparse grid; no synthetic replacement of Lite."""
    wp.copy(s.ptc_x,wp.array(positions,dtype=vec3,device=s.device))
    wp.copy(s.ptc_F,wp.array(F,dtype=mat33,device=s.device))
    s.ptc_v.zero_();s.ptc_G.zero_();s.grid_v.zero_();s.grid_v_new.zero_()
    values=wp.array(velocity,dtype=vec3,device=s.device)
    wp.launch(assign_velocity,dim=len(velocity),inputs=[s.ndof2bijk,values,s.grid_v_new],device=s.device)
    wp.synchronize_device(s.device);start=time.perf_counter()
    wp.launch(lite_g2c_kernel,dim=(s.bcn,B,B,B),inputs=[s.block_count,s.block2bid,s.block_xyz_by_id,
        s.grid_v,s.grid_v_new,s.center_v,s.center_dv,s.center_G,s.center_size,s.dx],device=s.device)
    wp.launch(lite_c2p_kernel,dim=s.n_ptc,inputs=[s.block2bid,s.ptc_x,s.ptc_v,s.ptc_k,s.ptc_F,s.ptc_G,s.ptc_dlogJ,
        s.center_m,s.center_v,s.center_dv,s.center_G,s.psi_params,s.center_size,s.dx,dt,0.],device=s.device)
    wp.synchronize_device(s.device);seconds=time.perf_counter()-start
    return s.ptc_x.numpy(),s.ptc_F.numpy(),s.ptc_G.numpy(),seconds


def history_read(positions,reference,F,volume,queries,h):
    """Same Hermite inverse-map reconstruction as production quadratic history."""
    start=time.perf_counter();tree=cKDTree(positions)
    keys,inverse=np.unique(np.floor(queries/h).astype(int),axis=0,return_inverse=True)
    result=np.empty((len(queries),3,3));max_condition=0.
    for group,key in enumerate(keys):
        center=(key+.5)*h;coef,condition=history_polynomial(positions,reference,F,volume,center,h,tree)
        ids=np.flatnonzero(inverse==group);result[ids]=reconstructed_F(coef,queries[ids],center,h)
        max_condition=max(max_condition,condition)
    return result,dict(history_seconds=time.perf_counter()-start,fit_centers=len(keys),max_condition=max_condition)


def mass_audit(blend,points,masses,grid):
    N,_=evaluate(blend,points);lumped=np.asarray(N.T@masses).ravel()
    lite=lite_values(blend.nodes,points,blend.h);old=np.asarray(lite.T@masses).ravel()
    C=evaluate(blend,face_points(grid,.25,6,True))[0].toarray();Z=null_space(C,rcond=1e-10)
    consistent=N.T@N.multiply(masses[:,None]);projected=Z.T@(consistent@Z)
    eig=eigvalsh((projected+projected.T)/2)
    threshold=1e-10*max(eig[-1],1e-30)
    return dict(total_mass=float(masses.sum()),mls_lumped_sum=float(lumped.sum()),
        negative_lumped_nodes=int(np.count_nonzero(lumped < -1e-14*masses.sum())),
        negative_mass_fraction=float(-lumped[lumped<0].sum()/masses.sum()),min_lumped=float(lumped.min()),
        lite_min_lumped=float(old.min()),consistent_mass_min=float(eig[0]),consistent_mass_max=float(eig[-1]),
        consistent_mass_soft_modes=int(np.count_nonzero(eig<=threshold)),free_scalar_dofs=Z.shape[1],
        particles=len(points),nodes=len(blend.nodes),minimum_shape_value=float(N.data.min()))
