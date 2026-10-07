"""Independent limiting-space identities and stress-error components."""
import numpy as np
from benchmarks.aniso_v20_common import *
from benchmarks.aniso_boundary_reference import hessian
from benchmarks.aniso_local_q3 import gradient
from engine.aniso_phase1 import local_reference as ref

def main():
    data=load(OUT/'local-relaxation.json');floor=data['patch_only_minimum_reaction_N'];identities=[]
    for resolution in (16,32):
        case=OUT/'local-relaxation'/f'r{resolution}-w0.25.npz';independent=BASE/'v19/space'/f'F45-reference{resolution}.npz';u=np.load(case)['u'];v=np.load(independent)['u'];R=data['records'][f'r{resolution}-w0.25']['reaction_N'];old=load(BASE/'v19/space/F45.json')
        # Archived v19 stores the fine reaction in JSON; coarse reaction follows
        # independently by assembling material-only stiffness on that same mesh.
        with np.load(BASE/'v19/space'/f'reconstruction{resolution}.npz') as z:edges=[z[f'axis{k}'] for k in range(3)]
        X,K=ref.assemble(edges,2,hessian('F45'));f=(K@v.T.ravel()).reshape(3,-1).T;Rref=float(f[X[:,0]>=.75-1e-12,0].sum());r=dict(resolution=resolution,displacement_max=float(np.max(abs(u-v))),reaction_identity_error_N=abs(R-Rref-floor),material_only_reaction_N=Rref,patch_only_reaction_N=floor,independent_reference_sha256=sha(independent));assert r['displacement_max']<1e-10 and r['reaction_identity_error_N']<1e-10;identities.append(r)
    source=BASE/'v11-reference-q3/cases/local2.npz'
    with np.load(source) as z:edges=[z[f'axis{k}'] for k in range(3)];ur=z['u']
    with np.load(BASE/'v19/space/reconstruction32.npz') as z:own=[z[f'axis{k}'] for k in range(3)]
    files={'compatible':OUT/'space/F45-r32-e0.npz','local150':OUT/'space/F45-r32-e3.npz','full_strips':OUT/'local-relaxation/r32-w0.1875.npz','full_interior':OUT/'local-relaxation/r32-w0.25.npz'};solutions={n:np.load(p)['u'] for n,p in files.items()};totals={n:{r:np.zeros((3,3)) for r in ('global','grip','interior')} for n in files};rotated={n:{r:np.zeros((3,3)) for r in ('global','grip','interior')} for n in files};T=np.array([[1.,-1.,0.],[1.,1.,0.],[0.,0.,np.sqrt(2)]])/np.sqrt(2);H=hessian('F45');common=[np.union1d(a,b) for a,b in zip(edges,own)]
    for X,V in ref.chunks(common,order=4,size=32):
        P=(gradient(X,edges,3,ur).reshape(-1,9)@H.T).reshape(-1,3,3);grip=(X[:,0]<=.3125)|(X[:,0]>=.6875)
        for name,u in solutions.items():
            p=(ref.gradient(X,own,2,u).reshape(-1,9)@H.T).reshape(-1,3,3);values=V[:,None,None]*(p-P)**2;dv=np.einsum('ai,pab,bj->pij',T,p-P,T);rv=V[:,None,None]*dv*dv
            for r,mask in [('global',np.ones(len(X),bool)),('grip',grip),('interior',~grip)]:totals[name][r]+=values[mask].sum(0);rotated[name][r]+=rv[mask].sum(0)
    components={n:{r:dict(error_squared_Pa2_m3=v.tolist(),component_error_power_fraction=(v/v.sum()).tolist(),fiber_frame_error_power_fraction=(rotated[n][r]/v.sum()).tolist(),fiber_axial_error_power_fraction=float(rotated[n][r][0,0]/v.sum())) for r,v in region.items()} for n,region in totals.items()}
    write(OUT/'space-diagnosis.json',dict(completed=True,limiting_space_identities=identities,error_components=components,interpretation='With full interior relaxation the material displacement exactly reproduces the independent Q2 solve; remaining extra reaction equals the unchanged carrier patch minimum. Component power fractions describe error to the archived Q3 comparison, not certified continuum error.'))
if __name__=='__main__':main()
