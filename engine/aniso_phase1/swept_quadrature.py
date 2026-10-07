"""Positive nested Gauss cubature cut at moved grid-kernel crossings.

The supplied physical map is trilinear on each reference box. Sweep z/y/x:
cut outer lines at box-edge crossings, then cut inner x lines at ALL physical
coordinate planes. Curved intersection topology is verified by order and
optional outer-cell refinement; exactness for a general deformed map is not
assumed. Reference volume, not deformed density, supplies the weights.
"""
import itertools
import numpy as np

def line_breaks(lo,hi,ends,h):
    cuts=[lo,hi]
    for a,b in np.asarray(ends).reshape(-1,2,3):
        for j in range(3):
            if abs(b[j]-a[j])<1e-15:continue
            low,high=sorted((a[j],b[j]));planes=(np.arange(int(np.floor(low/h-.5))-1,int(np.ceil(high/h-.5))+2)+.5)*h
            for plane in planes:
                root=lo+(plane-a[j])/(b[j]-a[j])*(hi-lo)
                if lo+1e-12*(hi-lo)<root<hi-1e-12*(hi-lo):cuts.append(root)
    return np.unique(cuts)

def points(edges,order):
    q,w=np.polynomial.legendre.leggauss(order);q=(q+1)/2;w=w/2;d=np.diff(edges)
    return (edges[:-1,None]+d[:,None]*q).ravel(),(d[:,None]*w).ravel()

def sweep(edges,physical,h,order=3,outer_refine=0):
    axes=[np.asarray(e) for e in edges]
    for _ in range(outer_refine):
        for j in (1,2):axes[j]=np.sort(np.r_[axes[j],(axes[j][1:]+axes[j][:-1])/2])
    out=[];weights=[];cuts=0
    for ijk in itertools.product(*[range(len(e)-1) for e in axes]):
        lo=np.array([axes[j][i] for j,i in enumerate(ijk)]);hi=np.array([axes[j][i+1] for j,i in enumerate(ijk)])
        corners=np.array(list(itertools.product(*[(lo[j],hi[j]) for j in range(3)])));physical_corners=physical(corners).reshape(4,2,3)
        ez=line_breaks(lo[2],hi[2],physical_corners,h);zz,wz=points(ez,order);cuts+=len(ez)-2
        for z,vz in zip(zz,wz):
            yy_corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1])]);ey=line_breaks(lo[1],hi[1],physical(yy_corners).reshape(2,2,3),h);yy,wy=points(ey,order);cuts+=len(ey)-2
            for y,vy in zip(yy,wy):
                ends=np.array([[lo[0],y,z],[hi[0],y,z]]);ex=line_breaks(lo[0],hi[0],physical(ends).reshape(1,2,3),h);xx,wx=points(ex,order);cuts+=len(ex)-2
                out.append(np.column_stack((xx,np.full(len(xx),y),np.full(len(xx),z))));weights.append(wx*vy*vz)
    X=np.concatenate(out);V=np.concatenate(weights)
    return X,V,dict(points=len(X),line_cuts=cuts,order=order,outer_refine=outer_refine,reference_volume=float(V.sum()))
