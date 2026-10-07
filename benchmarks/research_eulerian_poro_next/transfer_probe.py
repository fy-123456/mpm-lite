"""Actual original Lite kernels: crossing, C/L semantics and map compatibility.

Prescribed-velocity transfer probes only: no enriched/poro solve is implied.
"""
import argparse,json,gc
from pathlib import Path
import numpy as np
from .diagnose import write


def run_probe(run):
    import warp as wp
    wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
    from engine.aniso_phase1.solver import AnisotropicLiteImplicitSolver
    from engine.aniso_phase1.types import AnisotropicMaterialParams
    from engine.types import vec3
    from engine.sp_grid import B
    from engine.aniso_phase1.affine_transfer import prepare_incremental,finish_transfer
    from engine.kernel.d3.kernel_lite import lite_c2g,lite_g2c_kernel,lite_c2p_kernel
    from benchmarks.aniso_apic_frequency import Oracle,CORNERS
    from engine.aniso_phase1.research_eulerian_poro_next.transfer import capture_lite_kinematics
    dx=1/64
    s=AnisotropicLiteImplicitSolver((65,17,17),AnisotropicMaterialParams(10,20,200),
       dx=dx,device='cpu',gravity=0.,boundary_impulse_transfer=True,apic_transfer='incremental')
    X=np.stack(np.meshgrid([.451,.464,.477],[.085,.10,.115],[.085,.10,.115],indexing='ij'),axis=-1).reshape(-1,3)
    s.seed_particles(X,vol0=1e-6,density=1.)
    rows=[]
    for name in ('translation','rotation','affine','nonaffine'):
        s.ptc_x.assign(X);s.ptc_F.assign(np.tile(np.eye(3),(len(X),1,1)))
        s.ptc_A0.assign(np.tile(np.diag([1.,0.,0.]),(len(X),1,1)))
        if name=='translation':v=np.tile([.045,0,0.],(len(X),1));C=np.zeros((len(X),3,3));dt=1.
        elif name in ('rotation','affine'):
            L=(np.array([[0,-.02,0],[.02,0,0],[0,0,0.]]) if name=='rotation'
               else np.array([[.02,.01,0],[0,-.01,.004],[.003,0,0.]]))
            v=(X-X.mean(axis=0))@L.T;C=np.tile(L,(len(X),1,1));dt=.01
        else:
            k=50.;v=np.zeros_like(X);v[:,0]=.01*np.sin(k*(X[:,0]-.45))
            C=np.zeros((len(X),3,3));C[:,0,0]=.01*k*np.cos(k*(X[:,0]-.45));dt=.01
        s.ptc_v.assign(v);s.ptc_C.assign(C);s.ptc_L.assign(C)
        for step in range(2 if name=='translation' else 1):
            x=s.ptc_x.numpy().copy();v0=s.ptc_v.numpy().copy();C0=s.ptc_C.numpy().copy();F0=s.ptc_F.numpy().copy()
            s.activate_sparse_grid();coords0=s.block_xyz_by_id.numpy()[:s.bcn].copy();s.reset_grid()
            if not s._transfer_to_centers():raise ValueError('original history resampling failed')
            lite_c2g(s.block_count,s.block2bid,s.block_xyz_by_id,s.bc_block2bid,s.bc_type,s.bc_norm,s.bc_velo,
                s.hf_bc_p,s.hf_bc_n,s.hf_bc_v,s.hf_bc_type,s.num_hf,s.center_m,s.center_v,s.center_G,s.center_vol,
                s.center_tau,s.grid_m,s.grid_v,s.grid_v_new,s.grid_size,s.center_size,0.,s.dx,0.,s.n_psi,s.device,
                explicit_force=False,enable_apic=True)
            if s.grid_v_raw is None or s.grid_v_raw.shape[0]<s.bcn:
                s.grid_v_raw=wp.zeros((s.bcn,B,B,B),dtype=vec3,device='cpu')
            wp.copy(s.grid_v_raw,s.grid_v,count=s.bcn*B**3)
            wp.copy(s.grid_v_new,s.grid_v)
            oracle=Oracle(x,s.ptc_m.numpy(),dx);b=oracle.nodes//B;loc=oracle.nodes%B
            bid=s.block2bid.numpy()[b[:,0],b[:,1],b[:,2]]
            ids=bid*B**3+loc[:,0]*B**2+loc[:,1]*B+loc[:,2]
            nv=s.grid_v[:s.bcn].numpy().reshape(-1,3)[ids]
            Lexpected=np.einsum('pc,cnj,ni->pij',oracle.S,oracle.D,nv)
            vpic=oracle.S@oracle.H@nv
            # Derivative of the actual PIC advection map uses grad(S), not S*D.
            base=np.floor(x/dx-.5).astype(int);f=x/dx-.5-base
            center_lookup={tuple(c):i for i,c in enumerate(oracle.c)}
            gradS=np.zeros((len(x),len(oracle.c),3))
            for p in range(len(x)):
                for corner in CORNERS:
                    j=center_lookup[tuple(base[p]+corner)]
                    factors=np.where(corner,f[p],1-f[p])
                    for a in range(3):gradS[p,j,a]=(2*corner[a]-1)*np.prod(np.delete(factors,a))/dx
            Lmap=np.einsum('pcj,ci->pij',gradS,oracle.H@nv)
            prepare_incremental(s)
            wp.launch(lite_g2c_kernel,dim=(s.bcn,B,B,B),inputs=[s.block_count,s.block2bid,s.block_xyz_by_id,
                s.grid_v_raw,s.grid_v_new,s.center_v,s.center_dv,s.center_G,s.center_size,s.dx],device='cpu')
            wp.launch(lite_c2p_kernel,dim=s.n_ptc,inputs=[s.block2bid,s.ptc_x,s.ptc_v,s.ptc_k,s.ptc_F,s.ptc_G,
                s.ptc_dlogJ,s.center_m,s.center_v,s.center_dv,s.center_G,s.psi_params,s.center_size,dx,dt,s.flip_ratio],device='cpu')
            finish_transfer(s);state=capture_lite_kinematics(s)
            ferror=float(np.max(abs(state['F']-(np.eye(3)+dt*Lexpected)@F0)))
            xerror=float(np.max(abs(state['x']-(x+dt*vpic))))
            Cexpected=Lexpected+s.flip_ratio*(C0-Lexpected)
            cerr=float(np.max(abs(state['C']-Cexpected)))
            s.activate_sparse_grid();coords1=s.block_xyz_by_id.numpy()[:s.bcn].copy()
            row=dict(case=name,step=step+1,particles=len(x),dt=dt,
                F_kernel_error=ferror,advection_kernel_error=xerror,C_kernel_error=cerr,
                C_L_difference=float(np.max(abs(state['C']-state['L']))),
                advection_history_gradient_gap=float(np.linalg.norm(Lmap-Lexpected)),
                relative_gradient_gap=float(np.linalg.norm(Lmap-Lexpected)/max(np.linalg.norm(Lexpected),1e-10)),
                min_detF=float(np.linalg.det(state['F']).min()),
                crossed_cells=int(np.count_nonzero(np.any(np.floor(x/dx)!=np.floor(state['x']/dx),axis=1))),
                crossed_blocks=int(np.count_nonzero(np.any(np.floor(x/(B*dx))!=np.floor(state['x']/(B*dx)),axis=1))),
                blocks_before=coords0.tolist(),blocks_after=coords1.tolist(),
                momentum_change=float(np.linalg.norm(s.ptc_m.numpy()@(state['v']-v0))))
            if max(ferror,xerror,cerr)>1e-7 or row['min_detF']<=0:raise AssertionError(row)
            rows.append(row)
    assert sum(r['crossed_blocks'] for r in rows)>0
    assert max(r['relative_gradient_gap'] for r in rows if r['case']=='nonaffine')>.05
    write(run/'S2/original-kernel-probes.json',dict(scope='original transfer only; no poro/Eulerian integration',
        rows=rows,G1_passed=False,reason='nonaffine particle advection derivative differs from stored history gradient'))
    print(json.dumps(rows,indent=2));del s;gc.collect()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run_probe(a.run)
