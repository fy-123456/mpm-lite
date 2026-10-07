"""Frozen-state force visibility under actual two-level APIC transfers.

Algebraic attribution only: low P2G visibility is not permission to damp a mode.
The existing real-kernel oracle anchors the P2G map on each saved input state.
"""
import json
import numpy as np
from benchmarks.aniso_compatible_diagnosis import BASE,write,maps
from benchmarks.aniso_apic_frequency import Oracle


def main():
    p=json.loads((BASE/'v14/protocol.json').read_text());cfg=p['configs']['material-fourth'];h=1/(cfg['grid']-1);records=[]
    for t in (.5,.85,1.1,1.2,1.4,1.6):
        with np.load(BASE/'v14/cases/material-fourth'/f'audit-{round(t/cfg["dt"]):05d}.npz') as f:z={k:f[k] for k in f.files}
        with np.load(BASE/'v15/diagnosis'/f'fourth-{t:g}.npz') as f:d={k:f[k] for k in f.files}
        o=Oracle(z['particle_x_before'],z['particle_mass'],h);T=o.S@o.H
        r=o.xn[:,None,:]-o.x[None,:,:];D=np.einsum('pn,npj,npk->pjk',T,r,r).diagonal(axis1=1,axis2=2)
        q=np.concatenate([o.m]+[o.m*D[:,k] for k in range(3)])
        A=np.concatenate([T.T*o.m[None,:]]+[T.T*o.m[None,:]*r[:,:,k] for k in range(3)],axis=1)
        W=A/np.sqrt(o.mn[:,None]*q[None,:]);lam,U=np.linalg.eigh(W@W.T);lam=np.maximum(lam,0)
        old=np.concatenate([z['particle_velocity_before']]+[z['particle_C_before'][:,:,k] for k in range(3)])
        raw=A@old/o.mn[:,None];lookup={tuple(n):i for i,n in enumerate(z['native_nodes'])};perm=np.array([lookup[tuple(n)] for n in o.nodes])
        # Saved raw velocities are in the ledger order, separately permuted.
        saved={tuple(n):i for i,n in enumerate(z['grid_nodes'])};savedperm=np.array([saved[tuple(n)] for n in o.nodes])
        anchor=float(np.max(abs(raw-z['grid_velocity_raw'][savedperm])));assert anchor<1e-12
        # Actual impulse return: delta v_p=T delta v_g, delta C_p=G delta v_g.
        _,G=maps(o.x,o.nodes,h);J=np.vstack([T]+[g.toarray() for g in G])
        return_map=np.sqrt(o.mn[:,None])*(A@J/o.mn[:,None])/np.sqrt(o.mn[None,:])
        free=(o.xn[:,0]>.25)&(o.xn[:,0]<.75)
        # Constrained node increments are zero; project both input and output.
        return_map[~free]=0.;return_map[:,~free]=0.
        modes={}
        for label,f in [('material',d['material_force'][perm]),('stabilization',d['stabilization_force'][perm]),('total',(d['material_force']+d['stabilization_force'])[perm])]:
            force=f.copy();force[~free]=0.;a=force/np.sqrt(o.mn[:,None]);c=U.T@a;den=max(float(np.sum(c*c)),1e-30)
            modes[label]=dict(acceleration_mass_norm_squared=float(np.sum(a*a)),left_null_fraction=float(np.sum(c[lam<=1e-12]**2)/den),
                weak_nonzero_fraction=float(np.sum(c[(lam>1e-12)&(lam<.1)]**2)/den),resolved_fraction=float(np.sum(c[lam>=.1]**2)/den),
                impulse_return_norm_ratio=float(np.linalg.norm(return_map@a)/max(np.linalg.norm(a),1e-30)),
                impulse_return_alignment=float(np.sum(a*(return_map@a))/max(np.sum(a*a),1e-30)))
        records.append(dict(time=t,p2g_real_snapshot_anchor_max=anchor,left_null_modes=int(np.sum(lam<=1e-12)),weak_nonzero_modes=int(np.sum((lam>1e-12)&(lam<.1))),modes=modes))
        print(t,modes,flush=True)
    write(BASE/'v15/force-visibility.json',dict(completed=True,records=records,scope=__doc__,weak_threshold=.1,
        threshold_role='diagnostic inherited v14 visibility category, not a new dissipation parameter',
        formula='W=M_grid^-1/2 A M_APIC^-1/2; fhat=M_grid^-1/2 f_free; actual impulse return M_grid^1/2 M_grid^-1 A [T;G] M_grid^-1/2'))

if __name__=='__main__':main()
