"""Objective, frozen-reference-map stabilization on production sparse centers.

Xhat_i = x_i - u_i is reconstructed from particle material history BETWEEN
steps. For each center/Gauss sample, JX = dXhat/dx and grad_X N = JX^-T grad_x N
are frozen during Newton. F(v) = JX^-1 + dt sum(v_i tensor grad_X N_i).
The force includes the polar rotation derivative. The exact tangent includes
its derivative; project_pd selects the PSD Gauss--Newton approximation instead.
No particle traversal, autograd tape, finite differences or dense global matrix
is used inside a residual or matvec. Rebinning error is NOT eliminated by this.
"""
import warp as wp
from engine.types import real,vec3,mat33,mat99
from engine.sp_grid import B,unlin_IJK
from .enhancements import reference_stress,flat


@wp.func
def skew_solve(Kinv: mat33,T: mat33):
    """Solve U Z + Z U = T for skew T, Kinv=(tr(U)I-U)^-1."""
    w=Kinv@vec3(T[2,1],T[0,2],T[1,0])
    return mat33(real(0),-w[2],w[1],w[2],real(0),-w[0],-w[1],w[0],real(0))


@wp.func
def polar_data(F: mat33):
    L,s,V=wp.svd3(F)
    R=L@wp.transpose(V)
    U=wp.transpose(R)@F
    U=real(.5)*(U+wp.transpose(U))
    Kinv=wp.inverse(wp.trace(U)*wp.identity(3,dtype=real)-U)
    return R,U,Kinv


@wp.func
def admissible(F: mat33):
    L,s,V=wp.svd3(F)
    return wp.isfinite(wp.determinant(F)) and wp.determinant(F)>real(0) and wp.min(s)>real(1e-8) and wp.max(s)<real(1e8)


@wp.kernel
def prepare_reference(ids: wp.array(dtype=int,ndim=2),u: wp.array(dtype=vec3),
                      g: wp.array(dtype=vec3,ndim=2),F0: wp.array(dtype=mat33,ndim=2),
                      B0: wp.array(dtype=vec3,ndim=3),invalid: wp.array(dtype=int)):
    c,q=wp.tid();JX=wp.identity(3,dtype=real)
    for i in range(8):
        n=ids[c,i]
        if n<0:
            wp.atomic_max(invalid,0,1)
            return
        JX-=wp.outer(u[n],g[q,i])
    if not admissible(JX):
        wp.atomic_max(invalid,0,1)
        return
    f=wp.inverse(JX);F0[c,q]=f
    for i in range(8):B0[c,q,i]=wp.transpose(f)@g[q,i]


@wp.func
def gradient(c: int,q: int,ids: wp.array(dtype=int,ndim=2),
             B0: wp.array(dtype=vec3,ndim=3),v: wp.array(dtype=vec3),dt: real):
    result=mat33(real(0))
    for i in range(ids.shape[1]):result+=dt*wp.outer(v[ids[c,i]],B0[c,q,i])
    return result


@wp.kernel
def residual(cdof: wp.array(dtype=wp.vec2i),ids: wp.array(dtype=int,ndim=2),
             F0: wp.array(dtype=mat33,ndim=2),B0: wp.array(dtype=vec3,ndim=3),v: wp.array(dtype=vec3),
             A: wp.array(dtype=mat33,ndim=4),M: wp.array(dtype=mat99),slots: wp.array(dtype=int),
             volume: wp.array(dtype=real,ndim=4),out: wp.array(dtype=vec3),energy: wp.array(dtype=real),
             invalid: wp.array(dtype=int),dt: real,eta: real,mu: real,lam: real,kf: real,fourth: int):
    c,z=wp.tid();q=z+1;b,l=cdof[c][0],cdof[c][1];i,j,k=unlin_IJK(l)
    Fc=F0[c,0]+gradient(c,0,ids,B0,v,dt)
    Fq=F0[c,q]+gradient(c,q,ids,B0,v,dt)
    if not admissible(Fc) or not admissible(Fq):
        wp.atomic_max(invalid,0,1)
        return
    R,U,Kinv=polar_data(Fc)
    D=wp.transpose(R)@(Fq-Fc)
    m=mat99(real(0))
    if fourth!=0:m=M[slots[flat(b,i,j,k)]-1]
    S=reference_stress(D,A[0,b,i,j*B+k],m,mu,lam,kf,1,fourth)
    # Adjoint of dR: Z=L_U^-1(D S^T-S D^T), not a frozen-rotation force.
    Z=skew_solve(Kinv,D@wp.transpose(S)-S@wp.transpose(D))
    Pq=R@S;Pc=R@(Z-S)
    weight=eta*volume[0,b,i,j*B+k]/real(8)
    for n in range(ids.shape[1]):
        wp.atomic_add(out,ids[c,n],dt*weight*(Pq@B0[c,q,n]+Pc@B0[c,0,n]))
    e=real(.5)*weight*wp.trace(wp.transpose(D)@S)
    wp.atomic_add(energy,0,e)
    # A scale for roundoff in the original potential's Armijo comparison.
    scale=weight*wp.sqrt(wp.trace(wp.transpose(S)@S))*(real(1)+wp.sqrt(wp.trace(wp.transpose(Fq)@Fq))+wp.sqrt(wp.trace(wp.transpose(Fc)@Fc)))
    wp.atomic_add(energy,1,wp.abs(e)+scale)


@wp.kernel
def tangent(cdof: wp.array(dtype=wp.vec2i),ids: wp.array(dtype=int,ndim=2),
            F0: wp.array(dtype=mat33,ndim=2),B0: wp.array(dtype=vec3,ndim=3),v: wp.array(dtype=vec3),p: wp.array(dtype=vec3),
            A: wp.array(dtype=mat33,ndim=4),M: wp.array(dtype=mat99),slots: wp.array(dtype=int),
            volume: wp.array(dtype=real,ndim=4),out: wp.array(dtype=vec3),
            dt: real,eta: real,mu: real,lam: real,kf: real,fourth: int,project_pd: int):
    c,z=wp.tid();q=z+1;b,l=cdof[c][0],cdof[c][1];i,j,k=unlin_IJK(l)
    Fc=F0[c,0]+gradient(c,0,ids,B0,v,dt)
    Fq=F0[c,q]+gradient(c,q,ids,B0,v,dt)
    R,U,Kinv=polar_data(Fc)
    D=wp.transpose(R)@(Fq-Fc)
    dFc=gradient(c,0,ids,B0,p,dt);dFq=gradient(c,q,ids,B0,p,dt)
    dA=wp.transpose(R)@dFc
    Omega=skew_solve(Kinv,dA-wp.transpose(dA))
    dD=wp.transpose(R)@(dFq-dFc)-Omega@D
    m=mat99(real(0))
    if fourth!=0:m=M[slots[flat(b,i,j,k)]-1]
    a=A[0,b,i,j*B+k]
    dS=reference_stress(dD,a,m,mu,lam,kf,1,fourth)
    # Gauss--Newton: L^T H0 L, with the full first derivative L including dR.
    dZ=skew_solve(Kinv,D@wp.transpose(dS)-dS@wp.transpose(D))
    dPq=R@dS;dPc=R@(dZ-dS)
    if project_pd==0:
        S=reference_stress(D,a,m,mu,lam,kf,1,fourth)
        Z=skew_solve(Kinv,D@wp.transpose(S)-S@wp.transpose(D))
        dU=dA-Omega@U;dU=real(.5)*(dU+wp.transpose(dU))
        rhs=dD@wp.transpose(S)+D@wp.transpose(dS)-dS@wp.transpose(D)-S@wp.transpose(dD)-dU@Z-Z@dU
        dZ=skew_solve(Kinv,rhs)
        dPq=R@(Omega@S+dS)
        dPc=R@(Omega@(Z-S)+dZ-dS)
    weight=eta*volume[0,b,i,j*B+k]/real(8)
    for n in range(ids.shape[1]):
        wp.atomic_add(out,ids[c,n],dt*weight*(dPq@B0[c,q,n]+dPc@B0[c,0,n]))
