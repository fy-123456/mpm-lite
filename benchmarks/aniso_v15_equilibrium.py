"""Frozen-map nonlinear relaxation: distinguish stored from inaccessible energy.

These are counterfactual equilibria, not trajectories or proposed energy resets.
Both constitutive F and carrier Y vary with the same frozen grid increment used
by the respective solver. Clamp increments stay zero; inertia is omitted.
"""
import json
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from benchmarks.aniso_compatible_diagnosis import BASE, maps, gradient, scalar_K, write
from benchmarks.aniso_compatible_controls import stiffness
from benchmarks.aniso_dynamic_check import pk1
from engine.aniso_phase1.material_patch import carrier_map
from engine.aniso_phase1.diagnostics import energy_density
from engine.aniso_phase1.types import AnisotropicMaterialParams


def relax(z,cfg,kind):
    h=1/(cfg['grid']-1);F=z['particle_F_after'];Y=z['marker_after'];A=z['particle_A0'];V=z['particle_volume']
    _,G=maps(z['particle_x_after'],z['native_nodes'],h)
    _,G0=maps(z['particle_reference_x'],np.rint(z['patch_X']/h).astype(int),h)
    _,_,N=carrier_map(Y,z['native_nodes'],h);R=np.linalg.solve(gradient(G0,Y),F)
    B=[sum(G[j].multiply(F[:,j,k,None]) for j in range(3)).tocsr() for k in range(3)] if kind=='baseline' else [sum((G0[j]@N).multiply(R[:,j,k,None]) for j in range(3)).tocsr() for k in range(3)]
    n=len(z['native_nodes']);fixed=(z['native_nodes'][:,0]*h<=.25)|(z['native_nodes'][:,0]*h>=.75)
    free=np.flatnonzero(np.tile(~fixed,3));u=np.zeros((n,3));params=AnisotropicMaterialParams(10,20,cfg['kf']);Ks=N.T@scalar_K(z)@N
    def evaluate(u):
        Ft=F+gradient(B,u);Yt=Y+N@u
        r=np.einsum('cij,cja->cia',z['patch_P'],Yt[z['patch_ids']])
        Um=float(V@energy_density(Ft,A,params));Us=float(.5*np.sum(z['patch_weight'][:,None,None]*r*r))
        cf=np.zeros_like(Y);ff=z['patch_weight'][:,None,None]*np.einsum('cji,cja->cia',z['patch_P'],r)
        np.add.at(cf,z['patch_ids'].ravel(),ff.reshape(-1,3));Ps=pk1(Ft,A,cfg['kf'])
        force=sum(b.T@(V[:,None]*Ps[:,:,k]) for k,b in enumerate(B))+N.T@cf
        return Um+Us,Um,Us,Ft,force
    initial=evaluate(u);trace=[]
    for step in range(12):
        E,Um,Us,Ft,f=evaluate(u);r=f.T.ravel()[free];trace.append(dict(iteration=step,energy_J=E,force_norm_N=float(np.linalg.norm(r))))
        if np.linalg.norm(r)<1e-11:break
        Km,_=stiffness(Ft,A,V,B,cfg['kf']);K=Km+sp.block_diag([Ks]*3,format='csr');H=K[free][:,free].toarray()
        du=np.zeros(3*n);du[free]=np.linalg.solve(H,-r);descent=float(r@du[free]);assert descent<0
        accepted=False
        for j in range(20):
            scale=2.**(-j);trial=u+scale*du.reshape(3,n).T
            if evaluate(trial)[0]<=E+1e-4*scale*descent:u=trial;accepted=True;break
        assert accepted,'line search failed'
    E,Um,Us,Ft,f=evaluate(u);norm=lambda P:float(np.sqrt(np.mean(np.sum(P*P,axis=(1,2)))))
    assert np.linalg.norm(f.T.ravel()[free])<1e-10
    return dict(mode=kind,initial_J=initial[0],relaxed_J=E,relaxed_material_J=Um,relaxed_stabilization_J=Us,
        releasable_fraction=1-E/initial[0],initial_P_rms_Pa=norm(pk1(F,A,cfg['kf'])),relaxed_P_rms_Pa=norm(pk1(Ft,A,cfg['kf'])),
        max_grid_displacement_m=float(np.max(np.linalg.norm(u,axis=1))),free_force_N=float(np.linalg.norm(f.T.ravel()[free])),trace=trace)


def main():
    out=BASE/'v15/frozen-relaxation';out.mkdir(exist_ok=False);p=json.loads((BASE/'v14/protocol.json').read_text());cfg=p['configs']['material-fourth'];records=[]
    for t in (1.1,1.2,1.4,1.6):
        with np.load(BASE/'v14/cases/material-fourth'/f'audit-{round(t/cfg["dt"]):05d}.npz') as f:z={k:f[k] for k in f.files}
        for kind in ('baseline','compatible'):
            result=relax(z,cfg,kind);result['time']=t;records.append(result);print(t,kind,result,flush=True)
    write(out/'summary.json',dict(completed=True,records=records,scope=__doc__))

if __name__=='__main__':main()
