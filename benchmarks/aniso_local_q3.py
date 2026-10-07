"""Cubic cross-order extension: same tensor FEM assembly, independent basis polynomials.

Only physical material is meshed. Exact Gauss integration; no MPM maps, mass,
or stabilization. Local refinement means splitting selected physical intervals;
tensor closure extends a marked interval through the corresponding slab.
"""
import itertools
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator
from engine.aniso_phase1.boundary_reference import LO, HI, tensor_matrix


def basis(q,degree):
    # Independently constructed Lagrange basis, including cubic order.
    q=np.asarray(q);points=np.linspace(0.,1.,degree+1);N=[];D=[]
    for i,x in enumerate(points):
        poly=np.poly1d([1.])
        for j,y in enumerate(points):
            if j!=i:poly*=np.poly1d([1.,-y])/(x-y)
        N.append(poly(q));D.append(np.polyder(poly)(q))
    return np.stack(N,axis=-1),np.stack(D,axis=-1)


def axis(edges,degree,order=None):
    edges=np.asarray(edges);h=np.diff(edges);p=degree;n=len(h)
    z,w=np.polynomial.legendre.leggauss(order or p+1);q,w=(z+1)/2,w/2;N,D=basis(q,p)
    blocks=[h[:,None,None]*(N.T@(w[:,None]*N)),(D.T@(w[:,None]*D))/h[:,None,None],np.broadcast_to(D.T@(w[:,None]*N),(n,p+1,p+1))]
    ids=p*np.arange(n)[:,None]+np.arange(p+1);rows=np.repeat(ids,p+1,axis=1).ravel();cols=np.tile(ids,(1,p+1)).ravel()
    coords=np.r_[np.concatenate([edges[:-1]+k/p*h for k in range(p)]).reshape(p,n).T.ravel(),edges[-1]]
    return coords,[sp.csr_matrix((b.ravel(),(rows,cols)),shape=(n*p+1,)*2) for b in blocks]


def assemble(edges,degree,H):
    axes=[axis(e,degree) for e in edges];coords=[a[0] for a in axes];grams={}
    for i in range(3):
        for j in range(3):
            factors=[a[1][0] for a in axes]
            if i==j:factors[i]=axes[i][1][1]
            else:factors[i],factors[j]=axes[i][1][2],axes[j][1][2].T
            grams[i,j]=tensor_matrix(factors)
    blocks=[]
    for a in range(3):
        row=[]
        for b in range(3):
            terms=[H[3*a+i,3*b+j]*grams[i,j] for i in range(3) for j in range(3) if H[3*a+i,3*b+j]!=0]
            row.append(sum(terms) if terms else sp.csr_matrix(grams[0,0].shape))
        blocks.append(row)
    return np.array(list(itertools.product(*coords))),sp.bmat(blocks,format='csr')


def solve(edges,degree,H):
    nodes,K=assemble(edges,degree,H);n=len(nodes)
    fixed=(nodes[:,0]<=.25+1e-12)|(nodes[:,0]>=.75-1e-12);right=nodes[:,0]>=.75-1e-12
    free=np.flatnonzero(np.tile(~fixed,3));u=np.zeros(3*n);u[:n][right]=.005
    rhs=-(K@u)[free];A=K[free][:,free].tocsr();inv=1/A.diagonal();count=0
    def callback(_):
        nonlocal count
        count+=1
    u[free],info=cg(A,rhs,M=LinearOperator(A.shape,matvec=lambda v:inv*v),rtol=2e-11,atol=1e-14,maxiter=30000,callback=callback)
    force=K@u;residual=float(np.linalg.norm(force[free])/np.linalg.norm(rhs));energy=float(.5*u@force);R=float(force[:n][right].sum())
    return nodes,u.reshape(3,n).T,dict(degree=degree,cells=[len(e)-1 for e in edges],nodes=n,free_dofs=len(free),iterations=count,relative_residual=residual,linear_info=int(info),reaction_N=R,energy_J=energy,work_identity_relative=abs(energy-.5*.005*R)/energy,passed=bool(info==0 and residual<1e-8),mass_included=False,stiffness_shift=0.)


def gradient(X,edges,degree,u):
    p=degree;triples=np.array(list(itertools.product(range(p+1),repeat=3)));cell=[];q=[];hs=[]
    for k,e in enumerate(edges):
        e=np.asarray(e);idx=np.clip(np.searchsorted(e,X[:,k],side='right')-1,0,len(e)-2)
        h=e[idx+1]-e[idx];cell.append(idx);q.append((X[:,k]-e[idx])/h);hs.append(h)
    cell=np.array(cell).T;q=np.array(q).T;hs=np.array(hs).T
    ids=np.ravel_multi_index((p*cell[:,None,:]+triples).reshape(-1,3).T,tuple(p*(len(e)-1)+1 for e in edges)).reshape(-1,len(triples))
    N,D=basis(q,p);v=u[ids];out=np.empty((len(X),3,3))
    for k in range(3):
        w=np.ones((len(X),len(triples)))/hs[:,k,None]
        for j in range(3):w*=(D if j==k else N)[:,j,triples[:,j]]
        out[:,:,k]=np.einsum('pna,pn->pa',v,w)
    return out


def chunks(edges,order=3,size=512):
    q,w=np.polynomial.legendre.leggauss(order);q,w=(q+1)/2,w/2
    q=np.array(list(itertools.product(q,repeat=3)));w=np.prod(np.array(list(itertools.product(w,repeat=3))),axis=1)
    cells=np.array(list(itertools.product(*[range(len(e)-1) for e in edges])))
    for start in range(0,len(cells),size):
        c=cells[start:start+size];lo=np.column_stack([edges[k][c[:,k]] for k in range(3)]);hi=np.column_stack([edges[k][c[:,k]+1] for k in range(3)]);h=hi-lo
        yield (lo[:,None,:]+h[:,None,:]*q).reshape(-1,3),(h.prod(axis=1)[:,None]*w).ravel()


def compare(a,b,H):
    ea,pa,ua,ra=a;eb,pb,ub,rb=b;common=[np.union1d(x,y) for x,y in zip(ea,eb)]
    totals={k:np.zeros(4) for k in ('whole','near_grip','interior','deep_interior')};peaks=np.zeros(2)
    for X,V in chunks(common):
        La,Lb=gradient(X,ea,pa,ua),gradient(X,eb,pb,ub);Pa,Pb=(La.reshape(-1,9)@H.T).reshape(-1,3,3),(Lb.reshape(-1,9)@H.T).reshape(-1,3,3)
        norm=lambda x:np.sum(x*x,axis=(1,2))
        values=np.column_stack((norm(La-Lb),norm(Lb),norm(Pa-Pb),norm(Pb)))*V[:,None]
        dist=np.minimum(abs(X[:,0]-.25),abs(X[:,0]-.75));masks=dict(whole=np.ones(len(X),bool),near_grip=dist<.0625,interior=(X[:,0]>.3125)&(X[:,0]<.6875),deep_interior=(X[:,0]>.375)&(X[:,0]<.625))
        for k,m in masks.items():totals[k]+=values[m].sum(axis=0)
        peaks=np.maximum(peaks,[np.sqrt(norm(Pa)).max(),np.sqrt(norm(Pb)).max()])
    regions={k:dict(F_relative=float(np.sqrt(v[0]/v[1])),P_relative=float(np.sqrt(v[2]/v[3]))) for k,v in totals.items()}
    R=abs(ra['reaction_N']-rb['reaction_N'])/abs(rb['reaction_N'])
    return dict(reaction_relative=R,regions=regions,near_grip_share=float(totals['near_grip'][2]/totals['whole'][2]) if totals['whole'][2]>0 else 0.,sampled_peak_P=peaks.tolist(),reaction_passed=R<=.01,interior_passed=max(regions['interior'].values())<=.02,boundary_passed=max(regions['near_grip'].values())<=.02,global_passed=max(regions['whole'].values())<=.02)

import argparse,hashlib,json,time,unittest
from pathlib import Path
from benchmarks.aniso_boundary_reference import hessian,write
from benchmarks.aniso_local_reference import uniform
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'docs/results/lite-aniso-mainline';OUT=BASE/'v11-reference-q3'


class CubicTests(unittest.TestCase):
    def test_cubic_field_exact_and_gauss4_matches5(self):
        e=uniform(16);e[0]=np.union1d(e[0],[.265625,.734375]);x,K=assemble(e,3,hessian('F45'))
        u=np.column_stack((x[:,0]**3,x[:,0]*x[:,1]*x[:,2],x[:,1]**2*x[:,2]));X=np.random.default_rng(215).uniform(LO,HI,(20,3))
        L=gradient(X,e,3,u);T=np.zeros_like(L);T[:,0,0]=3*X[:,0]**2;T[:,1,0]=X[:,1]*X[:,2];T[:,1,1]=X[:,0]*X[:,2];T[:,1,2]=X[:,0]*X[:,1];T[:,2,1]=2*X[:,1]*X[:,2];T[:,2,2]=X[:,1]**2
        np.testing.assert_allclose(L,T,atol=3e-12)
        for a,b in zip(axis(e[0],3,4)[1],axis(e[0],3,5)[1]):self.assertLess(np.linalg.norm((a-b).data),1e-10)
        W=np.array([[0,.1,0],[-.1,0,.2],[0,-.2,0]]);self.assertLess(np.linalg.norm(K@(x@W.T).T.ravel()),1e-10)
    def test_generated_q2_basis_matches_previous(self):
        from engine.aniso_phase1.local_reference import assemble as old
        e=uniform(16);x,K=assemble(e,2,hessian('F45'));y,J=old(e,2,hessian('F45'))
        np.testing.assert_array_equal(x,y);self.assertLess(np.linalg.norm((K-J).data),1e-11)


def hashes():
    files=['benchmarks/aniso_local_q3.py','engine/aniso_phase1/boundary_reference.py','engine/aniso_phase1/beam_reference.py','engine/aniso_phase1/local_reference.py','benchmarks/aniso_local_reference.py']
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('freeze','tests','run','analyze'));a=p.parse_args();OUT.mkdir(exist_ok=True,parents=True)
    if a.action=='freeze':
        if (OUT/'protocol.json').exists():raise RuntimeError('preserve protocol')
        prior=json.loads((BASE/'v11-reference/protocol.json').read_text());base=uniform(64);edges=[]
        for k,(e,marks) in enumerate(zip(base,prior['marked_intervals'])):
            x=uniform(32)[k]
            for i in marks:x=np.union1d(x,np.linspace(e[i],e[i+1],5))
            edges.append(x.tolist())
        write(OUT/'protocol.json',dict(source_sha256=hashes(),selection_protocol_sha256=hashlib.sha256((BASE/'v11-reference/protocol.json').read_bytes()).hexdigest(),cases={'uniform32':[e.tolist() for e in uniform(32)],'local2':edges},degree=3,quadrature_order=4,reason='Q1/Q2 interior stress cross-order difference remains large; test higher order without changing physical model',gates=dict(reaction=.01,fields=.02)))
        return
    protocol=json.loads((OUT/'protocol.json').read_text());assert protocol['source_sha256']==hashes()
    if a.action=='tests':
        with (OUT/'tests.log').open('x') as stream:r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CubicTests))
        write(OUT/'tests.json',dict(passed=r.wasSuccessful(),tests=r.testsRun));raise SystemExit(0 if r.wasSuccessful() else 2)
    if a.action=='run':
        assert json.loads((OUT/'tests.json').read_text())['passed'];dest=OUT/'cases';dest.mkdir(exist_ok=False);records=[]
        for name,axes in protocol['cases'].items():
            e=[np.array(x) for x in axes];start=time.monotonic();x,u,r=solve(e,3,hessian('F45'));r.update(seconds=time.monotonic()-start,name=name)
            write(dest/(name+'.json'),r);np.savez_compressed(dest/(name+'.npz'),nodes=x,u=u,**{f'axis{k}':v for k,v in enumerate(e)});records.append(r);print(name,r,flush=True);assert r['passed']
        write(OUT/'runs.json',dict(completed=True,records=records));return
    def load(folder,name,p):
        with np.load(folder/(name+'.npz')) as z:return [z[f'axis{k}'].copy() for k in range(3)],p,z['u'].copy(),json.loads((folder/(name+'.json')).read_text())
    a=load(OUT/'cases','uniform32',3);b=load(OUT/'cases','local2',3);q=load(BASE/'v11-reference/cases','F45-local2-q2',2)
    # Gauss4 integrates cubic-gradient squared exactly on common cells.
    global chunks
    old_chunks=chunks
    def chunks(edges,order=4,size=256):return old_chunks(edges,order,size)
    rows=[]
    for name,l,r in [('q3_uniform_to_local',a,b),('q2_q3_local_cross',q,b)]:
        result=compare(l,r,hessian('F45'));result['name']=name;rows.append(result);print(name,result,flush=True)
    write(OUT/'summary.json',dict(completed=True,pairs=rows,full_stress_reference_certified=all(r['global_passed'] for r in rows),reaction_reference_passed=all(r['reaction_passed'] for r in rows),interior_reference_passed=all(r['interior_passed'] for r in rows)))


if __name__=='__main__':main()
