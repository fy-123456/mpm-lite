"""Small independent Q1 + cell-bubble carrier for transfer/flow contracts.

Reference coordinates are fixed. Particles physically move; this module does
not implement Eulerian grid relocation or inherit the 144-function certificate.
"""
import itertools
import numpy as np
from numpy.polynomial.legendre import leggauss

CORNERS = np.array(list(itertools.product((0, 1), repeat=3)))

class Space:
    def __init__(self, shape=(2, 1, 1), bubbles=None, lengths=(1., .25, .25)):
        self.shape = tuple(shape)
        if len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError('three positive integer cell counts required')
        self.lengths = np.asarray(lengths, float)
        if self.lengths.shape != (3,) or np.any(self.lengths <= 0):
            raise ValueError('positive dimensions required')
        self.h = self.lengths / shape
        self.cells = int(np.prod(shape))
        self.cuts = tuple(np.linspace(0, self.lengths[d], shape[d]+1) for d in range(3))
        self.nodes = np.array(list(np.ndindex(*(n+1 for n in shape)))) * self.h
        self.bubbles = tuple(range(self.cells)) if bubbles is None else tuple(bubbles)
        if len(set(self.bubbles)) != len(self.bubbles) or any(b < 0 or b >= self.cells for b in self.bubbles):
            raise ValueError('unique valid bubble cells required')
        self.ncarrier = len(self.nodes)
        self.nscalar = self.ncarrier + len(self.bubbles)
        self.free = np.r_[np.flatnonzero(self.nodes[:, 0] > 0),
                          np.arange(self.ncarrier, self.nscalar)]
        self.ndof = 3*len(self.free)

    def rule(self, order=4, particles=False):
        if type(order) is not int or order < 1: raise ValueError('positive integer order')
        if particles:
            a=(np.arange(order)+.5)/order;w=np.ones(order)/order
        else:
            a,w=leggauss(order);a=(a+1)/2;w=w/2
        ids=np.array(list(itertools.product(range(order),repeat=3)))
        local=a[ids];weights=np.prod(w[ids],axis=1)*np.prod(self.h)
        X=np.concatenate([(np.array(c)+local)*self.h for c in np.ndindex(self.shape)])
        return X,np.tile(weights,self.cells),np.repeat(np.arange(self.cells),len(local))

    def basis(self, X, free=True):
        X=np.asarray(X,float)
        if X.ndim!=2 or X.shape[1]!=3 or not np.isfinite(X).all():raise ValueError('finite positions required')
        z=X/self.h
        if np.any(z < -1e-10) or np.any(z > np.array(self.shape)+1e-10):
            raise ValueError('points outside reference carrier')
        cell=np.minimum(np.maximum(np.floor(z).astype(int),0),np.array(self.shape)-1)
        s=z-cell;ids=np.ravel_multi_index(cell.T,self.shape)
        N=np.zeros((len(X),self.nscalar));D=np.zeros((len(X),self.nscalar,3))
        for corner in CORNERS:
            col=np.ravel_multi_index((cell+corner).T,tuple(n+1 for n in self.shape))
            f=np.where(corner,s,1-s);N[np.arange(len(X)),col]=np.prod(f,axis=1)
            for a in range(3):
                D[np.arange(len(X)),col,a]=(2*corner[a]-1)*np.prod(f[:,[k for k in range(3) if k!=a]],axis=1)/self.h[a]
        for j,c in enumerate(self.bubbles):
            ix=np.flatnonzero(ids==c);t=s[ix];g=4*t*(1-t)
            N[ix,self.ncarrier+j]=np.prod(g,axis=1)
            for a in range(3):
                D[ix,self.ncarrier+j,a]=4*(1-2*t[:,a])*np.prod(g[:,[k for k in range(3) if k!=a]],axis=1)/self.h[a]
        if free:return N[:,self.free],D[:,self.free]
        return N,D

    def identity(self):
        return dict(shape=list(self.shape),lengths=self.lengths.tolist(),bubbles=list(self.bubbles),
                    carrier_scalar=self.ncarrier,local_scalar=len(self.bubbles),free_vector=self.ndof,
                    coordinate='fixed material reference',stabilization='none; full mass, rank checked')
