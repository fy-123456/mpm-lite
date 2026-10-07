"""Manufactured velocity fields evaluated through the production G2C/C2P kernels."""
import itertools
import numpy as np
import warp as wp
from engine.sp_grid import B
from engine.types import real, vec3, mat33, MaterialParams, Material
from engine.kernel.d3.kernel_lite import lite_g2c_kernel, lite_c2p_kernel


def manufactured_probe(cells=8, waves=1, offset=.23, field='sine'):
    """No material solve: exact nodal samples, complete support, float64 CPU.

    The PIC derivative oracle differentiates the two-level interpolation, not
    the continuum velocity. Its finite-difference cross-check freezes topology.
    """
    dx=1./cells; size=cells+1; nb=(size+B-1)//B
    blocks=np.array(list(itertools.product(range(nb),repeat=3)),dtype=np.int32)
    block2bid=np.arange(len(blocks),dtype=np.int32).reshape((nb,)*3)
    local=np.stack(np.meshgrid(*([np.arange(B)]*3),indexing='ij'),axis=-1)
    coords=blocks[:,None,None,None,:]*B+local
    xnode=coords*dx
    A=np.array([[.08,.02,.01],[.03,-.04,.015],[.01,.02,.06]])
    bias=np.array([.13,-.04,.07]);amplitude=.01;k=2*np.pi*waves
    def velocity(x):
        if field=='affine':return x@A.T+bias
        if field=='constant':return np.broadcast_to(bias,x.shape).copy()
        v=np.zeros_like(x);v[...,0]=amplitude*np.sin(k*x[...,0]);return v
    def gradient(x):
        if field=='affine':return np.broadcast_to(A,x.shape[:-1]+(3,3)).copy()
        out=np.zeros(x.shape[:-1]+(3,3))
        if field=='sine':out[...,0,0]=amplitude*k*np.cos(k*x[...,0])
        return out
    shape=(len(blocks),B,B,B)
    block_count=wp.array([len(blocks)],dtype=int,device='cpu')
    bmap=wp.array(block2bid,dtype=int,device='cpu');bcoords=wp.array(blocks,dtype=wp.vec3i,device='cpu')
    old=wp.zeros(shape,dtype=vec3,device='cpu')
    new=wp.array(velocity(xnode),dtype=vec3,device='cpu')
    cv=wp.zeros(shape,dtype=vec3,device='cpu');cdv=wp.zeros_like(cv)
    cg=wp.zeros(shape,dtype=mat33,device='cpu');cm=wp.ones(shape,dtype=real,device='cpu')
    wp.launch(lite_g2c_kernel,dim=shape,inputs=[block_count,bmap,bcoords,old,new,cv,cdv,cg,wp.vec3i(cells,cells,cells),dx],device='cpu')
    # Include near-grip points on both sides of x=.25/.75, away from cell knots.
    axes=[(np.arange(1,cells-1)+.5+offset)*dx]*3
    xp=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    params=MaterialParams();params.mate_type=Material.elastic
    material=wp.array([params],dtype=MaterialParams,device='cpu')
    count=len(xp);px=wp.array(xp,dtype=vec3,device='cpu');pv=wp.zeros(count,dtype=vec3,device='cpu')
    pf=wp.array(np.broadcast_to(np.eye(3),(count,3,3)).copy(),dtype=mat33,device='cpu')
    pg=wp.zeros(count,dtype=mat33,device='cpu');pk=wp.zeros(count,dtype=int,device='cpu');pj=wp.zeros(count,dtype=real,device='cpu')
    wp.launch(lite_c2p_kernel,dim=count,inputs=[bmap,px,pv,pk,pf,pg,pj,cm,cv,cdv,cg,material,wp.vec3i(cells,cells,cells),dx,0.,0.],device='cpu')
    center_v=cv.numpy();center_g=cg.numpy()
    def interpolate(points,derivative=False):
        q=points/dx-.5;base=np.floor(q).astype(int);f=q-base
        out=np.zeros((len(points),3,3) if derivative else (len(points),3))
        for corner in itertools.product((0,1),repeat=3):
            c=base+corner;bid=block2bid[tuple((c//B).T)];loc=c%B
            values=center_v[bid,loc[:,0],loc[:,1],loc[:,2]]
            factors=np.where(corner,f,1-f)
            if derivative:
                dw=np.stack([(2*corner[d]-1)/dx*np.prod(np.delete(factors,d,axis=1),axis=1) for d in range(3)],axis=1)
                out+=values[:,:,None]*dw[:,None,:]
            else:out+=values*np.prod(factors,axis=1)[:,None]
        return out
    exact=gradient(xp);got=pg.numpy();pic_grad=interpolate(xp,True)
    h=dx*1e-5
    fd=np.stack([(interpolate(xp+np.eye(3)[d]*h)-interpolate(xp-np.eye(3)[d]*h))/(2*h) for d in range(3)],axis=2)
    mask=np.all(coords<cells,axis=-1);xc=(coords[mask]+.5)*dx
    rms=lambda x:float(np.sqrt(np.mean(np.sum(x*x,axis=(-2,-1)))))
    denom=rms(exact)
    return dict(cells=cells,waves=waves,offset=offset,field=field,particles=count,
        center_gradient_rms_error=rms(center_g[mask]-gradient(xc)),
        particle_gradient_rms_error=rms(got-exact),
        particle_gradient_relative_error=rms(got-exact)/max(denom,1e-30),
        particle_gradient_amplitude_gain=float(np.sum(got*exact)/max(np.sum(exact*exact),1e-30)),
        pic_derivative_rms_error=rms(pic_grad-exact),
        gradient_vs_pic_derivative_rms=rms(got-pic_grad),
        pic_derivative_fd_max_error=float(np.max(abs(pic_grad-fd))),
        pic_velocity_oracle_max_error=float(np.max(abs(pv.numpy()-interpolate(xp)))),
        particle_gradient_max_error=float(np.max(abs(got-exact))))
