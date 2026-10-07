"""Householder application of weighted orthogonal projectors without tall Q.

The same QR+SVD rank threshold as the sealed implementation is retained.
Only the three velocity right-hand sides receive Householder transformations.
"""
import numpy as np
import scipy.linalg as la
from scipy.linalg.lapack import get_lapack_funcs

class ThinProjector:
    def __init__(self,A):
        (self.qr,self.tau),R=la.qr(A,mode='raw');U,sv,_=la.svd(R,full_matrices=False);self.U=U[:,sv>1e-12*sv[0]];self.rank=self.U.shape[1];self.n=R.shape[0];self.ormqr=get_lapack_funcs('ormqr',(self.qr,))
    def apply_q(self,z,transpose=False):
        trans='T' if transpose else 'N';_,w,info=self.ormqr('L',trans,self.qr,self.tau,np.asfortranarray(z),-1);assert info==0
        a,_,info=self.ormqr('L',trans,self.qr,self.tau,np.asfortranarray(z),int(w[0]),overwrite_c=0);assert info==0;return a
    def project(self,z):
        small=self.apply_q(z,True)[:self.n];v=np.zeros_like(z);v[:self.n]=self.U@(self.U.T@small);return self.apply_q(v)

def projection(J,q,z):
    root=np.sqrt(q)[:,None];p=ThinProjector(root*J);return p.project(root*z)/root,p.rank
