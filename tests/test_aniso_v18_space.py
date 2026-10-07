"""Independent checks for the new modal-stress continuum reference."""
import unittest
import numpy as np
from benchmarks.aniso_v18_compatible_reference import Targets,solve_projection,fields_at
from benchmarks.aniso_v18_space import gauss_sites
from benchmarks.aniso_compatible_diagnosis import maps
from benchmarks.aniso_boundary_reference import hessian
from engine.aniso_phase1 import local_reference as ref

class Manufactured:
    labels=['quadratic_bending','quadratic_shear']
    def values(self,X):
        q=np.clip((X[:,0]-.25)/.5,0,1);f=q*(1-q)
        out=np.zeros((2,len(X),3));out[0,:,0]=f*(X[:,1]-.5);out[0,:,1]=.4*f*(X[:,2]-.5);out[1,:,2]=f*(X[:,1]-.5)*(X[:,2]-.5)
        return out
    def strain(self,X):
        q=np.clip((X[:,0]-.25)/.5,0,1);f=q*(1-q);df=(1-2*q)*2*((X[:,0]>.25)&(X[:,0]<.75))
        E=np.zeros((len(X),2,3,3));E[:,0,0,0]=df*(X[:,1]-.5);E[:,0,0,1]=f;E[:,0,1,0]=.4*df*(X[:,2]-.5);E[:,0,1,2]=.4*f
        E[:,1,2,0]=df*(X[:,1]-.5)*(X[:,2]-.5);E[:,1,2,1]=f*(X[:,2]-.5);E[:,1,2,2]=f*(X[:,1]-.5)
        return E
    def stress(self,X):return np.einsum('ab,ptb->pta',hessian('F45'),self.strain(X).reshape(len(X),2,9)).reshape(len(X),2,3,3)

class SpaceReferenceTests(unittest.TestCase):
    def test_modal_gradient_matches_original_sparse_map(self):
        target=Targets();X,V=gauss_sites(target.h,3);_,G=maps(X,np.rint(target.nodes/target.h).astype(int),target.h)
        expected=np.stack([np.stack([g@field for g in G],axis=-1) for field in target.fields],axis=1)
        np.testing.assert_allclose(target.strain(X),expected,atol=2e-12,rtol=2e-11)
        P=target.stress(X);norm=np.sqrt(np.einsum('ptab,ptab,p->t',P,P,V)/V.sum());np.testing.assert_allclose(norm,1,atol=1e-12)
    def test_manufactured_compatible_fields_recovered_without_stabilization(self):
        edges=[np.linspace(a,b,round((b-a)*16)+1) for a,b in zip(ref.LO,ref.HI)];edges[1]=np.union1d(edges[1],[.40625,.59375]);target=Manufactured()
        nodes,values,record=solve_projection(edges,2,target)
        np.testing.assert_allclose(values,target.values(nodes),atol=2e-10,rtol=1e-7)
        X=np.random.default_rng(18).uniform(ref.LO,ref.HI,(113,3));U,E=fields_at(X,edges,2,values)
        np.testing.assert_allclose(U,target.values(X).transpose(1,0,2),atol=2e-10)
        np.testing.assert_allclose(E,target.strain(X),atol=2e-9)
        for row in record['records']:self.assertLess(abs(row['load_work']-2*row['elastic_energy_J']),1e-10)
if __name__=='__main__':unittest.main()
