"""Time-free integration and continuum compatibility checks for key modes."""
import itertools
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_apic_frequency import Oracle
from benchmarks.aniso_boundary_reference import hessian
from benchmarks.aniso_compatible_diagnosis import LO,HI,maps,gradient
from benchmarks.aniso_carrier_joint import spectrum
from benchmarks.aniso_v17_modes import controlled_case,ModalModel,snapshot_model
from benchmarks.aniso_v18_runs import ROOT,OUT,load,write,sha
from engine.aniso_phase1.selective_patch import scalar_matrix


def gauss_sites(h,order):
    # Split at every center-interpolation kink, as well as physical boundaries.
    edges=[]
    for a,b in zip(LO,HI):
        centers=(np.arange(-2,int(np.ceil(1/h))+3)+.5)*h
        edges.append(np.r_[a,centers[(centers>a+1e-12)&(centers<b-1e-12)],b])
    q,w=np.polynomial.legendre.leggauss(order);q=(q+1)/2;w=w/2;axes=[];weights=[]
    for e in edges:
        axes.append((e[:-1,None]+np.diff(e)[:,None]*q).ravel());weights.append((np.diff(e)[:,None]*w).ravel())
    X=np.array(list(itertools.product(*axes)));V=np.prod(np.array(list(itertools.product(*weights))),axis=1)
    return X,V


def assemble(h,order,label='F45'):
    X,V=gauss_sites(h,order);o=Oracle(X,V,h);nodes=o.nodes*h;T=o.S@o.H;G=[o.S@o.D[:,:,j] for j in range(3)]
    Ks,ids,P=scalar_matrix(nodes,o.c,o.S.T@V,h);Ks=Ks.toarray();H=hessian(label)
    free=(nodes[:,0]>.25+1e-12)&(nodes[:,0]<.75-1e-12);B=[g[:,free] for g in G];Ksr=Ks[free][:,free]
    blocks=[[sum(H[3*a+i,3*b+j]*(B[i].T@(V[:,None]*B[j])) for i in range(3) for j in range(3) if H[3*a+i,3*b+j]!=0) for b in range(3)] for a in range(3)]
    for a in range(3):blocks[a][a]+=Ksr
    K=np.block(blocks);K=(K+K.T)/2
    f=X/h-.5;f-=np.floor(f);D=h*h*(f*(1-f)+.25)
    J=np.vstack([np.sqrt(V)[:,None]*T[:,free]]+[np.sqrt(V*D[:,j])[:,None]*B[j] for j in range(3)])
    mass=J.T@J;M=la.block_diag(mass,mass,mass)
    return dict(h=h,order=order,label=label,X=X,V=V,nodes=nodes,free=free,G=G,T=T,K=K,M=M,J=J,Ks=la.block_diag(Ksr,Ksr,Ksr),H=H)


def modes(a,probe):
    J=a['J'];_,sv,Vh=la.svd(J,full_matrices=False);keep=sv>1e-12*sv[0]
    W=la.block_diag(*([Vh[keep].T/sv[keep]]*3));Z=la.block_diag(*([Vh[~keep].T]*3));K=a['K']
    if Z.shape[1]:W-=Z@la.solve(Z.T@K@Z,Z.T@K@W,assume_a='pos')
    H=W.T@K@W;lam,U=la.eigh((H+H.T)/2);assert lam.min()>0;phi=W@U;omega=np.sqrt(lam)
    nf=int(a['free'].sum());full=np.zeros((len(a['nodes']),3,len(omega)));full[a['free']]=phi.reshape(3,nf,-1).transpose(1,0,2)
    _,Gp=maps(probe,np.rint(a['nodes']/a['h']).astype(int),a['h'])
    dF=np.stack([np.einsum('pn,nai->pai',g.toarray(),full) for g in Gp],axis=2)
    DP=np.einsum('ab,pbi->pai',a['H'],dF.reshape(len(probe),9,-1)).reshape(len(probe),3,3,-1).transpose(3,0,1,2)
    residual=float(la.norm(K@phi-(a['M']@phi)*lam)/la.norm(K@phi));assert residual<1e-7
    return dict(omega=omega,phi=phi,full=full,DP=DP,null_modes=3*int((~keep).sum()),residual=residual,gate=spectrum(K))


def main():
    assert not (OUT/'space-quadrature.json').exists()
    write(OUT/'space-protocol.json',dict(source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'benchmarks/aniso_v18_space.py',ROOT/'benchmarks/aniso_boundary_reference.py',ROOT/'engine/aniso_phase1/local_reference.py']},
        grids=[1/8,1/10,1/12],quadrature_orders=[2,3,4],state='undeformed F45, same physical volume and hard grips',
        integration='cells split at Lite center-interpolation kinks; Q3 integrates degree-4 kinetic polynomials, Q4 cross-checks',
        production_particle_limit_unchanged=True,scope='independent offline matrix assembly, not a moving solver with more particles'))
    ref=snapshot_model(1.6);probe=ref.particle_reference;V=ref.e.V/ref.e.V.sum();targets=[int(ref.order[0]),211];records=[];checks=[];base=None
    for h in (1/8,1/10,1/12):
        previous=None
        for order in (2,3,4):
            a=assemble(h,order);b=modes(a,probe);matches=[]
            for target in targets:
                p=ref.D[target];norm=np.einsum('ipab,ipab,p->i',b['DP'],b['DP'],V);dot=np.einsum('ipab,pab,p->i',b['DP'],p,V);rn=np.einsum('pab,pab,p->',p,p,V)
                mac=dot*dot/(norm*rn);mac[norm<1e-16*norm.max()]=0;i=int(np.argmax(mac));vec=b['phi'][:,i]
                matches.append(dict(target=target,mode=i,omega_rad_s=float(b['omega'][i]),stress_MAC=float(mac[i]),stabilization_fraction=float(vec@a['Ks']@vec/b['omega'][i]**2)))
            record=dict(h=h,order=order,points=len(a['X']),volume=float(a['V'].sum()),zero_inertia_modes=b['null_modes'],gate=b['gate'],eigen_residual=b['residual'],matches=matches)
            assert b['gate']['passed'];records.append(record);print('quadrature',record,flush=True)
            if previous is not None:
                pa,pb=previous
                check=dict(h=h,orders=[order-1,order],K_relative=float(la.norm(a['K']-pa['K'])/la.norm(a['K'])),M_relative=float(la.norm(a['M']-pa['M'])/la.norm(a['M'])),frequencies_relative=float(la.norm(b['omega']-pb['omega'])/la.norm(b['omega'])))
                checks.append(check)
                if order==4:assert max(check['K_relative'],check['M_relative'],check['frequencies_relative'])<1e-7,check
            previous=(a,b)
            if h==.125 and order==3:base=(a,b)
    # Save both the original sampled modes and quadrature-converged alternatives
    # for the physically compatible Q1/Q2 projection experiment.
    s,e,m,h,meta=controlled_case();rest=ModalModel(s,e,m,h,meta['particle_reference'],meta['carrier_reference']);full=[];labels=[]
    for target in targets:
        norm=np.einsum('ipab,ipab,p->i',rest.D,rest.D,V);dot=np.einsum('ipab,pab,p->i',rest.D,ref.D[target],V);idx=int(np.argmax(dot*dot/np.maximum(norm,1e-30)))
        full.append(rest.Q@rest.phi[:,idx].reshape(3,rest.Q.shape[1]).T);labels.append(f'sampled-target{target}')
    a,b=base
    for target in targets:
        record=next(r for r in records if r['h']==.125 and r['order']==3);idx=next(v['mode'] for v in record['matches'] if v['target']==target)
        full.append(b['full'][:,:,idx]);labels.append(f'gauss3-target{target}')
    np.savez_compressed(OUT/'space-mode-targets.npz',nodes=meta['carrier_reference'],fields=np.array(full),labels=np.array(labels),h=.125)
    write(OUT/'space-quadrature.json',dict(completed=True,records=records,comparisons=checks,
        exact_quadrature_passed=True,spatial_accuracy_accepted=False,
        scope='Convergence of quadrature for this discrete Lite energy/mass does not establish continuum accuracy; matched shapes may mix across grids.'))
if __name__=='__main__':main()
