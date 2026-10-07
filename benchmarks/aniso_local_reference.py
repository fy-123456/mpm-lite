"""v11: frozen local Q1/Q2 hard-grip reference refinement."""
import argparse,hashlib,itertools,json,time,unittest
from pathlib import Path
import numpy as np
from engine.aniso_phase1 import local_reference as ref
from benchmarks.aniso_boundary_reference import hessian,write
from benchmarks.aniso_q2_reference import gradient as old_gradient,assemble as old_assemble
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/results/lite-aniso-mainline/v11-reference'
OLD=ROOT/'docs/results/lite-aniso-mainline/v10-reference-q2/cases'


def uniform(n):return [np.linspace(a,b,round((b-a)*n)+1) for a,b in zip(ref.LO,ref.HI)]

def load(n):
    with np.load(OLD/f'F45-q2-n{n}.npz') as z:u=z['u'].copy()
    return uniform(n),2,u,json.loads((OLD/f'F45-q2-n{n}.json').read_text())


def source_hashes():
    files=['engine/aniso_phase1/local_reference.py','benchmarks/aniso_local_reference.py','benchmarks/aniso_q2_reference.py','engine/aniso_phase1/boundary_reference.py','engine/aniso_phase1/beam_reference.py']
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}


class LocalReferenceTests(unittest.TestCase):
    def test_uniform_reproduces_independent_q2(self):
        H=hessian('F45');x,K=ref.assemble(uniform(16),2,H);y,J=old_assemble(16,H)
        np.testing.assert_array_equal(x,y);self.assertLess(np.linalg.norm((K-J).data),1e-11)
    def test_nonuniform_polynomial_and_exact_integration(self):
        edges=uniform(16);edges=[np.union1d(e,(e[:-1]+e[1:])[::2]/2) for e in edges]
        x,K=ref.assemble(edges,2,hessian('F45'));u=np.column_stack((x[:,0]**2,x[:,1]*x[:,2],x[:,2]**2))
        X=np.random.default_rng(15).uniform(ref.LO,ref.HI,(31,3));L=ref.gradient(X,edges,2,u);T=np.zeros_like(L)
        T[:,0,0]=2*X[:,0];T[:,1,1]=X[:,2];T[:,1,2]=X[:,1];T[:,2,2]=2*X[:,2]
        np.testing.assert_allclose(L,T,atol=2e-13)
        for e in edges:
            for A,B in zip(ref.axis(e,2,3)[1],ref.axis(e,2,4)[1]):self.assertLess(np.linalg.norm((A-B).data),1e-11)
    def test_nonuniform_rigid_and_work(self):
        edges=uniform(16);edges[1]=np.union1d(edges[1],[.40625,.59375])
        for p in (1,2):
            x,K=ref.assemble(edges,p,hessian('F45'));W=np.array([[0,.1,0],[-.1,0,.2],[0,-.2,0]])
            self.assertLess(np.linalg.norm(K@(x@W.T).T.ravel()),1e-11)
            _,_,r=ref.solve(edges,p,hessian('F45'));self.assertTrue(r['passed']);self.assertLess(r['work_identity_relative'],1e-8)


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve frozen reference')
    a,b=load(32),load(64);edges=uniform(64);hist=[np.zeros(len(e)-1) for e in edges];H=hessian('F45')
    for X,V in ref.chunks(edges):
        dP=(old_gradient(X,32,a[2])-old_gradient(X,64,b[2])).reshape(-1,9)@H.T
        error=V*np.sum(dP*dP,axis=1)
        for k,e in enumerate(edges):hist[k]+=np.bincount(np.searchsorted(e,X[:,k],side='right')-1,weights=error,minlength=len(e)-1)
    marks=[]
    for k,e in enumerate(edges):
        idx=np.argsort(-hist[k],kind='stable')[:max(1,int(np.ceil(len(hist[k])/8)))]
        if k==0:
            mid=(e[:-1]+e[1:])/2;idx=np.union1d(idx,np.flatnonzero(np.minimum(abs(mid-.25),abs(mid-.75))<1/64))
        marks.append(idx.astype(int))
    levels=[]
    for lev in (1,2):
        axes=[]
        for e,idx in zip(edges,marks):
            extra=np.concatenate([np.linspace(e[i],e[i+1],2**lev+1)[1:-1] for i in idx]);axes.append(np.union1d(e,extra).tolist())
        levels.append(axes)
    inputs={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for n in (32,64) for f in (OLD/f'F45-q2-n{n}.json',OLD/f'F45-q2-n{n}.npz')}
    write(out/'protocol.json',dict(source_sha256=source_hashes(),input_sha256=inputs,base_cells_per_unit=64,selection='top 1/8 intervals by integrated Q2 32/64 stress-difference squared, plus two clamp-adjacent x intervals per clamp; fixed before solving',axis_indicators=[h.tolist() for h in hist],marked_intervals=[m.tolist() for m in marks],levels=levels,cases=[[l,p] for l in (1,2) for p in (1,2)],Q1='split every Q2 interval in half for equal nodal coordinates',boundary='original hard grips',gates=dict(reaction=.01,whole=.02,interior=.02,near_grip=.02),adaptive_after_results=False))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('freeze','tests','run','analyze'));a=p.parse_args();out=OUT;out.mkdir(exist_ok=True,parents=True)
    if a.action=='freeze':freeze(out);return
    protocol=json.loads((out/'protocol.json').read_text());assert protocol['source_sha256']==source_hashes()
    if a.action=='tests':
        with (out/'tests.log').open('x') as stream:r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LocalReferenceTests))
        write(out/'tests.json',dict(passed=r.wasSuccessful(),tests=r.testsRun));raise SystemExit(0 if r.wasSuccessful() else 2)
    if a.action=='run':
        assert json.loads((out/'tests.json').read_text())['passed'];dest=out/'cases';dest.mkdir(exist_ok=False);records=[]
        for lev,p in protocol['cases']:
            edges=[np.array(e) for e in protocol['levels'][lev-1]]
            if p==1:edges=[np.union1d(e,(e[:-1]+e[1:])/2) for e in edges]
            start=time.monotonic();nodes,u,r=ref.solve(edges,p,hessian('F45'));r.update(level=lev,seconds=time.monotonic()-start)
            name=f'F45-local{lev}-q{p}';write(dest/(name+'.json'),r);np.savez_compressed(dest/(name+'.npz'),nodes=nodes,u=u,**{f'axis{k}':e for k,e in enumerate(edges)})
            records.append(r);print(name,r,flush=True);assert r['passed']
        write(out/'runs.json',dict(completed=True,records=records));return
    data={}
    for lev,p in protocol['cases']:
        name=f'F45-local{lev}-q{p}'
        with np.load(out/'cases'/(name+'.npz')) as z:data[lev,p]=([z[f'axis{k}'].copy() for k in range(3)],p,z['u'].copy(),json.loads((out/'cases'/(name+'.json')).read_text()))
    pairs=[]
    for name,a,b in [('uniform_to_local1_q2',load(64),data[1,2]),('local1_to_local2_q2',data[1,2],data[2,2]),('local1_to_local2_q1',data[1,1],data[2,1]),('cross_order_local2',data[2,1],data[2,2])]:
        r=ref.compare(a,b,hessian('F45'));r['name']=name;pairs.append(r);print(name,r,flush=True)
    write(out/'summary.json',dict(completed=True,pairs=pairs,full_stress_reference_certified=all(r['global_passed'] for r in pairs[-3:]),reaction_reference_passed=all(r['reaction_passed'] for r in pairs[-3:])))


if __name__=='__main__':main()
