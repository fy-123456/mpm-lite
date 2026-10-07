"""Independent archived-state audit using legacy sparse geometry and material oracle."""
import copy,json,time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from benchmarks.aniso_v20_common import *
from benchmarks.aniso_v20_runs import setup
from benchmarks.aniso_dynamic_check import pk1
from benchmarks.aniso_compatible_controls import stiffness
from benchmarks.aniso_carrier_joint import spectrum
from engine.aniso_phase1.carrier_joint import State,gradient,current_gradient
from engine.aniso_phase1.unresolved_velocity import maps as apic_maps
from engine.aniso_phase1.material_patch import carrier_map
from types import SimpleNamespace
from engine.aniso_phase1.integrated_avf import material_null_condensation
from engine.aniso_phase1.separate_kinetic import KineticGeometry
from engine.aniso_phase1.diagnostics import energy_density
from engine.aniso_phase1.unresolved_velocity import pack
from engine.aniso_phase1.endpoint_boundary import prescribed_speed

def independent_geometry(s,B,m,h):
    data=apic_maps(s.x,m,h);nodes=data['nodes'];_,_,Ns=carrier_map(s.Y,nodes,h);N=Ns.toarray();E=la.solve(N,np.eye(len(N)));T=data['T'].T@E;inv=np.linalg.inv(gradient(B,s.Y));L=[sum(b*inv[:,k,j,None] for k,b in enumerate(B)) for j in range(3)]
    assert np.max(abs(N@E-np.eye(len(N))))<1e-10
    return SimpleNamespace(nodes=nodes,N=N,E=E,T=T,J=np.vstack([T]+L),metric=data['metric'])

def audit_case(folder,kind):
    s,e,m,h,meta=setup(kind);rows=[json.loads(v) for v in (folder/'steps.jsonl').read_text().splitlines()];frames=np.load(folder/'stress.npz');records=[];previousY=s.Y.copy();previousF=gradient(e.B,s.Y);H,R,Z,C=material_null_condensation(e,s.Y,h) if kind.endswith('condensed') else (np.eye(e.n),np.eye(e.n),None,np.empty((0,e.n)))
    for path in sorted(folder.glob('audit-*.npz')):
        k=int(path.stem.split('-')[-1]);row=rows[k-1]
        with np.load(path) as z:s=State(z['x'],z['Y'],z['v'],z['C'],float(z['time']));savedF=z['F'];Q=z['Q']
        ec=copy.copy(e);ec.B=current_gradient(getattr(e,'kinetic_reference',meta['particle_reference']),np.rint(meta['carrier_reference']/h).astype(int),h);g=independent_geometry(s,ec.B,m,h);runtime=KineticGeometry(s,e,m,h)
        F=gradient(e.B,s.Y);P=pk1(F,e.A,200.);Um=float(e.V@energy_density(F,e.A,e.params));patch=e.P@s.Y[e.ids];Us=.5*float(np.sum(e.weights[:,None,None]*patch*patch));zv=pack(s.v,s.C);K=.5*float(np.sum(g.metric[:,None]*zv*zv));fixed=(g.nodes[:,0]*h<=.25)|(g.nodes[:,0]*h>=.75);ell=np.zeros((len(g.nodes),3));ell[g.nodes[:,0]*h>=.75,0]=1.;lift=R@(g.N@ell);end=np.sqrt(g.metric)[:,None]*(zv-g.J@lift*prescribed_speed(s.time))
        def project(A,x):
            U,sv,_=la.svd(A,full_matrices=False,lapack_driver='gesvd');U=U[:,sv>1e-12*sv[0]];return U@(U.T@x)
        root=np.sqrt(g.metric)[:,None];constraint=la.norm(project(root*(g.J@H),end)-project(root*(g.J@Q),end));Km,_=stiffness(F,e.A,e.V,[sp.csr_matrix(b@Q) for b in e.B],200.);Ks=Q.T@e.Ks@Q;Kstatic=Km.toarray()+la.block_diag(Ks,Ks,Ks);gate=spectrum(Kstatic);rel=float(la.norm(Kstatic-e.tangent(s.Y,Q))/la.norm(Kstatic));errors=dict(F=float(np.max(abs(F-savedF))),P=float(np.max(abs(P-frames['P'][k]))),history_between_snapshots=float(np.max(abs(F-previousF-gradient(e.B,s.Y-previousY)))),material_J=abs(Um-row['material_J']),stabilization_J=abs(Us-row['stabilization_J']),kinetic_J=abs(K-row['kinetic_J']),total_J=abs(Um+Us+K-row['total_J']),independent_J=float(np.max(abs(g.J-runtime.J))),independent_metric=float(np.max(abs(g.metric-runtime.metric))),endpoint_velocity_constraint=float(constraint),free_grip_basis=float(np.max(abs(g.E[fixed]@Q))),material_null_history=float(la.norm(C@s.Y)))
        assert max(errors.values())<1e-7 and max(v for k,v in errors.items() if k.endswith('_J'))<1e-12 and gate['passed'] and rel<1e-6,(path,errors)
        records.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),errors=errors,static=gate,independent_tangent_relative=rel));previousY=s.Y.copy();previousF=F.copy()
    return records

def main():
    done={}
    for cache in (OUT/'audits').glob('*.json'):
        old=load(cache)
        for r in old['records']:assert sha(ROOT/r['path'])==r['sha256']
        done[cache.stem]=old['records']
    cases={p['name']:p for p in load(OUT/'fast-cycle/cycle-protocol.json')['cases']}
    while len(done)<len(cases):
        manifest=OUT/'completed-case-paths.json'
        if manifest.exists():
            for name,p in load(manifest)['cases'].items():
                if name in done:continue
                if not p['completed']:raise RuntimeError((name,'failed final candidate'))
                folder=ROOT/p['path'];records=audit_case(folder,cases[name]['kind']);done[name]=records;write(OUT/'audits'/f'{name}.json',dict(completed=True,records=records));print('audited',name,len(records),flush=True)
                write(OUT/'independent-audits.json',dict(completed=len(done)==len(cases),records=done,total_snapshots=sum(map(len,done.values())),scope='Independent sparse geometry, material stress/energy, kinetic norm, boundary subspace, inter-snapshot material history, and mass-free tangent; step work is separately checked by full ledger.'))
        if len(done)<len(cases):time.sleep(10)
if __name__=='__main__':main()
