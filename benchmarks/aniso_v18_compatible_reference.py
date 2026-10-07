"""Q1/Q2 projection of Lite modal stress onto physical compatible strains.

Solve min_u 1/2 int (grad u-Etarget):H:(grad u-Etarget) with original
homogeneous hard grips. No MPM stabilization or exterior material enters FE.
This is a compatibility/space-resolution test, not an eigenfrequency truth.
"""
import itertools,time
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import cg,LinearOperator
from benchmarks.aniso_v18_runs import OUT,ROOT,load,write,sha
from benchmarks.aniso_v18_space import gauss_sites
from benchmarks.aniso_boundary_reference import hessian
from engine.aniso_phase1 import local_reference as ref


def shapes(X,edges,p):
    triples=np.array(list(itertools.product(range(p+1),repeat=3)));cell=[];q=[];hs=[]
    for k,e in enumerate(edges):
        idx=np.clip(np.searchsorted(e,X[:,k],side='right')-1,0,len(e)-2);h=e[idx+1]-e[idx];cell.append(idx);q.append((X[:,k]-e[idx])/h);hs.append(h)
    cell=np.array(cell).T;q=np.array(q).T;hs=np.array(hs).T;ids=np.ravel_multi_index((p*cell[:,None,:]+triples).reshape(-1,3).T,tuple(p*(len(e)-1)+1 for e in edges)).reshape(len(X),-1)
    N,D=ref.basis(q,p);weights=np.ones_like(ids,dtype=float);derivatives=[]
    for j in range(3):weights*=N[:,j,triples[:,j]]
    for k in range(3):
        d=np.ones_like(weights)/hs[:,k,None]
        for j in range(3):d*=(D if j==k else N)[:,j,triples[:,j]]
        derivatives.append(d)
    return ids,weights,np.stack(derivatives,axis=-1)


class Targets:
    def __init__(self):
        with np.load(OUT/'space-mode-targets.npz') as z:self.nodes=z['nodes'];self.fields=z['fields'];self.labels=z['labels'].tolist();self.h=float(z['h'])
        ijk=np.rint(self.nodes/self.h).astype(int);self.lo=ijk.min(0);self.shape=ijk.max(0)-self.lo+1;self.cshape=self.shape-1
        corners=np.array(list(itertools.product((0,1),repeat=3)));centers=np.array(list(itertools.product(*[range(n) for n in self.cshape])))
        ids=np.ravel_multi_index((centers[:,None,:]+corners).reshape(-1,3).T,tuple(self.shape)).reshape(-1,8)
        self.center=np.einsum('tcba,bk->ctak',self.fields[:,ids],(2*corners-1)/(4*self.h));self.corners=corners;self.H=hessian('F45')
        X,V=gauss_sites(self.h,3);E=self.strain(X);P=np.einsum('ab,ptb->pta',self.H,E.reshape(len(X),len(self.labels),9))
        self.scale=np.sqrt(np.einsum('pta,pta,p->t',P,P,V)/V.sum());self.fields/=self.scale[:,None,None];self.center/=self.scale[None,:,None,None]
    def strain(self,X):
        q=X/self.h-.5;base=np.floor(q).astype(int);f=q-base;cs=base[:,None,:]+self.corners-self.lo
        ids=np.ravel_multi_index(cs.reshape(-1,3).T,tuple(self.cshape)).reshape(-1,8)
        weights=np.prod(np.where(self.corners[None,:,:],f[:,None,:],1-f[:,None,:]),axis=2)
        return np.einsum('pc,pctab->ptab',weights,self.center[ids])
    def stress(self,X):
        E=self.strain(X);return np.einsum('ab,ptb->pta',self.H,E.reshape(len(X),len(self.labels),9)).reshape(E.shape)


def solve_projection(edges,degree,targets):
    H=hessian('F45');nodes,K=ref.assemble(edges,degree,H);n=len(nodes);nt=len(targets.labels);rhs=np.zeros((n,3,nt));start=time.monotonic()
    for X,V in ref.chunks(edges,order=3,size=64):
        ids,N,D=shapes(X,edges,degree);P=targets.stress(X)
        local=np.einsum('pnj,ptaj,p->pnat',D,P,V,optimize=True);np.add.at(rhs,ids.ravel(),local.reshape(-1,3,nt))
    fixed=(nodes[:,0]<=.25+1e-12)|(nodes[:,0]>=.75-1e-12);free=np.flatnonzero(np.tile(~fixed,3));A=K[free][:,free].tocsr();inv=1/A.diagonal();pre=LinearOperator(A.shape,matvec=lambda x:inv*x)
    values=np.zeros((nt,n,3));records=[]
    for j,label in enumerate(targets.labels):
        b=rhs[:,:,j].T.ravel()[free];count=[0]
        def cb(_):count[0]+=1
        u,info=cg(A,b,M=pre,rtol=2e-10,atol=1e-14,maxiter=20000,callback=cb);res=float(la.norm(A@u-b)/la.norm(b));assert info==0 and res<1e-8
        flat=np.zeros(3*n);flat[free]=u;values[j]=flat.reshape(3,n).T
        records.append(dict(label=label,iterations=count[0],relative_residual=res,elastic_energy_J=.5*float(u@(A@u)),load_work=float(u@b)))
    return nodes,values,dict(degree=degree,cells=[len(e)-1 for e in edges],nodes=n,free_dofs=len(free),seconds=time.monotonic()-start,records=records)


def fields_at(X,edges,degree,values):
    ids,N,D=shapes(X,edges,degree);local=values[:,ids].transpose(1,0,2,3)
    U=np.einsum('pn,ptna->pta',N,local,optimize=True);E=np.einsum('pnj,ptna->ptaj',D,local,optimize=True)
    return U,E


def compare_cases(cases,targets,finest_name="q2-n48"):
    # Integrate each polynomial piece separately, including all Q1/Q2 interfaces.
    edges=[np.unique(np.concatenate([c[0][k] for c in cases.values()])) for k in range(3)];H=hessian('F45');nt=len(targets.labels)
    sums={name:{region:np.zeros((nt,5)) for region in ('global','grip','interior')} for name in cases};pair_sums={name:{region:np.zeros((nt,2)) for region in ('global','grip','interior')} for name in cases}
    finest=cases[finest_name]
    for X,V in ref.chunks(edges,order=3,size=32):
        target=targets.strain(X);Pt=targets.stress(X);_,Ef=fields_at(X,*finest);Pf=np.einsum('ab,ptb->pta',H,Ef.reshape(len(X),nt,9)).reshape(Ef.shape)
        masks=dict(global_=np.ones(len(X),bool),grip=(X[:,0]<=.3125)|(X[:,0]>=.6875),interior=(X[:,0]>.3125)&(X[:,0]<.6875));masks['global']=masks.pop('global_')
        for name,args in cases.items():
            U,E=fields_at(X,*args);P=np.einsum('ab,ptb->pta',H,E.reshape(len(X),nt,9)).reshape(E.shape);dP=P-Pt;dE=E-target
            vals=np.stack((np.sum(dP*dP,axis=(2,3)),np.sum(Pt*Pt,axis=(2,3)),np.sum(dE*dP,axis=(2,3)),np.sum(target*Pt,axis=(2,3)),np.sum(U*U,axis=2)),axis=-1)*V[:,None,None]
            for region,mask in masks.items():sums[name][region]+=vals[mask].sum(axis=0)
            pair_values=np.stack((np.sum((P-Pf)**2,axis=(2,3)),np.sum(Pf*Pf,axis=(2,3))),axis=-1)*V[:,None,None]
            for region,mask in masks.items():pair_sums[name][region]+=pair_values[mask].sum(axis=0)
    out={}
    for name,regions in sums.items():
        out[name]=[]
        for j,label in enumerate(targets.labels):
            rec=dict(label=label,regions={key:dict(stress_mismatch_relative=float(np.sqrt(v[j,0]/v[j,1])),strain_energy_mismatch_relative=float(np.sqrt(max(v[j,2],0)/v[j,3]))) for key,v in regions.items()},
                stress_vs_finest_relative=float(np.sqrt(pair_sums[name]['global'][j,0]/pair_sums[name]['global'][j,1])),
                stress_vs_finest_regions={key:float(np.sqrt(v[j,0]/v[j,1])) for key,v in pair_sums[name].items()})
            saved=load(OUT/'compatible-reference'/f'{name}.json');E=saved['records'][j]['elastic_energy_J'];rec['compatible_field_rayleigh_rad_s']=float(np.sqrt(2*E/regions['global'][j,4]));out[name].append(rec)
    return out


def main():
    assert not (OUT/'compatible-reference.json').exists();assert load(OUT/'space-quadrature.json')['completed'];dest=OUT/'compatible-reference';dest.mkdir(exist_ok=False);targets=Targets();specs=[(1,32),(2,16),(1,64),(2,32),(2,48)]
    write(OUT/'compatible-reference-protocol.json',dict(source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'benchmarks/aniso_v18_compatible_reference.py',ROOT/'engine/aniso_phase1/local_reference.py']},
        meshes=specs,targets=targets.labels,boundary='original physical hard grips, homogeneous motion',method='energy-norm compatible-strain projection, no mass in solve or stabilization',
        relative_space_target=.02,scope='Compatibility and FE discretization check. Rayleigh quotient of projected field is not a natural frequency certificate.'))
    cases={}
    for p,n in specs:
        edges=[np.linspace(a,b,round((b-a)*n)+1) for a,b in zip(ref.LO,ref.HI)];name=f'q{p}-n{n}'
        x,u,r=solve_projection(edges,p,targets);np.savez_compressed(dest/f'{name}.npz',nodes=x,u=u,**{f'axis{k}':v for k,v in enumerate(edges)});write(dest/f'{name}.json',r);cases[name]=(edges,p,u);print(name,r,flush=True)
    comparisons=compare_cases(cases,targets)
    write(OUT/'compatible-reference.json',dict(completed=True,comparisons=comparisons,
        finite_element_refinement_passed=all(r['stress_vs_finest_relative']<.02 for r in comparisons['q2-n32']),
        physical_mode_accuracy_accepted=False,scope='Even a converged compatible projection may differ from the target Lite modal stress; time error is absent from this test.'))
    print('REFERENCE COMPLETE',comparisons,flush=True)
if __name__=='__main__':main()
