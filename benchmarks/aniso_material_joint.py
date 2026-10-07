"""Exact frozen linear coupled dynamics versus filter splitting.

The real composite P2G map and F45 massless stiffness define an auxiliary
energy-consistent linear system. This is a mechanism diagnostic, not a new
nonlinear production integrator. expm_multiply integrates mechanics and weak
velocity dissipation jointly. Frozen geometry has an exact exponential anchor.
"""
import hashlib
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from benchmarks.aniso_material_history import ROOT,BASE,LEVELS,load,write
from benchmarks.aniso_material_controls import SOURCE
from benchmarks.aniso_apic_frequency import Oracle
from benchmarks.aniso_static_space import stiffness
from benchmarks.aniso_boundary_reference import hessian
from engine.aniso_phase1.selective_patch import scalar_matrix
from engine.aniso_phase1.unresolved_velocity import VelocityFilter,pack
OUT=BASE/'v14-exploration'

def main():
    with np.load(SOURCE) as f:z={k:f[k].copy() for k in f.files}
    x=z['particle_x_after'];m=z['particle_mass'];h=.125;f=VelocityFilter(x,m,h,.001,np.sqrt(10)/h,'weak');d=f.data;n=len(d['nodes']);np_=len(x)
    o=Oracle(x,m,h);V=o.S.T@z['particle_volume'];G=[sp.csr_matrix(o.S@o.D[:,:,k]) for k in range(3)];H=hessian('F45')
    # Oracle grid ordering equals the production filter's lexicographic order.
    assert np.array_equal(o.nodes,d['nodes'])
    S,_,_=scalar_matrix(o.xn,o.c,V,h);K=stiffness(G,z['particle_volume'],H)+sp.block_diag([S]*3,format='csr')
    J=d['normalized']/np.sqrt(d['mass'][:,None]);fixed=(o.xn[:,0]<=.25)|(o.xn[:,0]>=.75);J[fixed]=0
    A=sp.block_diag([sp.csr_matrix(J)]*3,format='csr');nu=3*n;ny=12*np_;zero_u=sp.csr_matrix((nu,nu));zero_y=sp.csr_matrix((ny,ny))
    L=sp.bmat([[zero_u,A],[-A.T@K,zero_y]],format='csr')
    Q=f.affine;R=f.right;rates=f.rate*np.maximum(0,1-f.singular**2/.1)**2
    Ds=f.rate*(np.eye(4*np_)-Q@Q.T)-R.T@((f.rate-rates)[:,None]*R);Ds=(Ds+Ds.T)/2
    D=sp.block_diag([sp.csr_matrix(Ds)]*3,format='csr');B=sp.block_diag([zero_u,-D],format='csr');joint=L+B
    y=(f.sqrt_metric[:,None]*pack(z['particle_velocity_after'],z['particle_C_after'])).T.ravel();initial=np.r_[np.zeros(nu),y];T=.04
    ref=expm_multiply(joint*T,initial);conservative=expm_multiply(L*T,initial)
    energy=lambda a:float(.5*(a[:nu]@(K@a[:nu])+a[nu:]@a[nu:]));grid=lambda a:A@a[nu:]
    filter_cache={}
    def damp(a,dt,mode='weak'):
        if (dt,mode) not in filter_cache:filter_cache[dt,mode]=VelocityFilter(x,m,h,dt,f.rate,mode)
        ff=filter_cache[dt,mode];Y=a[nu:].reshape(3,4*np_).T;aff=Q@(Q.T@Y);c=R@(Y-aff);null=Y-aff-R.T@c
        b=a.copy();b[nu:]=(aff+R.T@(ff.alpha[:,None]*c)+ff.alpha_null*null).T.ravel();return b
    rows=[]
    for mode in ('weak','null'):
        anchor=ref if mode=='weak' else damp(conservative,.001,'null')
        for order in ('joint','pre','post','symmetric'):
            for level,dt in LEVELS.items():
                a=initial.copy();steps=round(T/dt)
                for _ in range(steps):
                    if order=='joint':
                        a=expm_multiply((joint if mode=='weak' else L)*dt,a)
                        if mode=='null':a=damp(a,dt,mode)
                    else:
                        if order in ('pre','symmetric'):a=damp(a,dt if order=='pre' else dt/2,mode)
                        a=expm_multiply(L*dt,a)
                        if order in ('post','symmetric'):a=damp(a,dt if order=='post' else dt/2,mode)
                r=dict(mode=mode,order=order,level=level,dt=dt,relative_state_error=float(np.linalg.norm(a-anchor)/np.linalg.norm(anchor)),relative_grid_velocity_error=float(np.linalg.norm(grid(a)-grid(anchor))/np.linalg.norm(grid(anchor))),final_energy_J=energy(a));rows.append(r);print(r,flush=True)
    # Smooth physical fields with their exact gradients: affine, quadratic bending,
    # and a sinusoidal velocity. Small visibility does not imply nonphysical motion.
    probes=[];q=x-np.average(x,axis=0,weights=m)
    for field in ('affine','quadratic_bending','smooth_sine'):
        v=np.zeros_like(x);C=np.zeros((np_,3,3))
        if field=='affine':
            C[:]=np.array([[.02,.03,0],[-.03,.01,.02],[0,-.02,-.01]]);v=q@C[0].T+.01
        elif field=='quadratic_bending':v[:,1]=q[:,0]**2;C[:,1,0]=2*q[:,0]
        else:v[:,1]=np.sin(2*np.pi*q[:,0]);C[:,1,0]=2*np.pi*np.cos(2*np.pi*q[:,0])
        for mode in ('null','weak'):
            ff=VelocityFilter(x,m,h,T,f.rate,mode);vp,Cp,stats=ff.apply(v,C)
            probes.append(dict(field=field,mode=mode,duration=T,energy_loss_fraction=-stats['dissipation_delta']/stats['kinetic_before_dissipation'],grid_momentum_change_relative=float(np.linalg.norm(ff.data['momentum_map']@(pack(vp,Cp)-pack(v,C)))/np.linalg.norm(ff.data['momentum_map']@pack(v,C))),affine_change_max=stats['protected_affine_change_max']))
    assert energy(ref)<=energy(initial)+1e-14
    assert max(r['relative_state_error'] for r in rows if r['order']=='joint')<1e-9
    assert max(r['relative_state_error'] for r in rows if r['mode']=='null')<1e-9
    write(OUT/'joint-linear-control.json',dict(completed=True,records=rows,smooth_fields=probes,initial_energy_J=energy(initial),joint_final_energy_J=energy(ref),no_dissipation_final_energy_J=energy(conservative),joint_vs_no_dissipation_grid_relative=float(np.linalg.norm(grid(ref)-grid(conservative))/np.linalg.norm(grid(conservative))),source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),scope='fixed real P2G support with auxiliary F45 linear elastic dynamics, stationary grips; exact exponential anchor; not nonlinear production solve; moving nonlinear comparison in v14-controls'))

if __name__=='__main__':main()
