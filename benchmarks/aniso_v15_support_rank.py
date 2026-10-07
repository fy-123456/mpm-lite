"""Check whether support expansion introduces massless modes in common history."""
import itertools,json
import numpy as np
import scipy.sparse as sp
import warp as wp
from benchmarks.aniso_compatible_diagnosis import BASE,write
from benchmarks.aniso_compatible_controls import stiffness
from benchmarks.aniso_compatible_history import CommonLedger
from engine.aniso_phase1.compatible_patch import CompatiblePatchLiteSolver,CompatiblePatchEnhancements
from engine.aniso_phase1.material_patch import MaterialPatchEnhancements
from demos.aniso import Config


class RejectedCandidateEnhancements(CompatiblePatchEnhancements):
    """Two-step diagnostic only: reproduce the original, unrestricted candidate."""
    prepare=MaterialPatchEnhancements.prepare


def main():
    wp.config.kernel_cache_dir=json.loads((BASE/'v15/storage-protocol.json').read_text())['temporary']+'/mpm-lite-warp-cache'
    h=1/16;dt=.005;v=np.array([.1,.025,-.02]);x=np.array(list(itertools.product(.34375+np.arange(4)*h/2,repeat=3)))
    s=CompatiblePatchLiteSolver((17,)*3,params=Config('tensile',17,dt,45.).params,dx=h,device='cpu',gravity=0.,ppc=1,
        flip_ratio=.9**(dt/.001),energy_diagnostics=True,boundary_impulse_transfer=True,apic_transfer='incremental',affine_flip_ratio=1.,velocity_dissipation='null')
    s.enhancements=RejectedCandidateEnhancements(s)
    s.seed_particles(x,density=1.,vol0=h**3/8,velocity=np.broadcast_to(v,x.shape));s.set_dt(dt);s.energy_ledger=CommonLedger();s.energy_ledger.begin(s);records=[]
    for i in range(2):
        assert s.step(max_iters=16,print_every=0,v_tol=1e-10,cg_tol=1e-4,cg_atol=1e-12,newton_atol=1e-10,max_cg_iters=1000,reaction_force_atol=1e-7),s.last_step_stats
        e=s.enhancements;state=e.state();ids=state['ids'];P=state['P'];m=ids.shape[1];n=e.N.shape[1]
        Ky=sp.csr_matrix(((state['weights'][:,None,None]*(P.swapaxes(1,2)@P)).ravel(),(np.repeat(ids,m,axis=1).ravel(),np.tile(ids,(1,m)).ravel())),shape=(len(e.origin),len(e.origin)))
        Km,_=stiffness(s.ptc_F.numpy(),s.ptc_A0.numpy(),s.ptc_vol0.numpy(),s.local_host_maps,200.)
        Ks=e.N.T@Ky@e.N;K=Km+sp.block_diag([Ks]*3,format='csr');lam=np.linalg.eigvalsh(K.toarray());tol=1e-9*max(float(abs(K).sum(axis=1).max()),1.)
        record=dict(step=i+1,carriers=e.N.shape[0],grid_nodes=n,zero_modes=int(np.sum(lam<=tol)),physical_rigid_modes=6,
            extra_modes=int(np.sum(lam<=tol))-6,min_eigenvalue=float(lam.min()),mass_included=False)
        G=tuple(s.mapped_S@d for d in s.mapped_D);F0=s.mapped_old_F
        oldB=[sum(g.multiply(F0[:,j,k,None]) for j,g in enumerate(G)).tocsr() for k in range(3)]
        oldKm,_=stiffness(s.ptc_F.numpy(),s.ptc_A0.numpy(),s.ptc_vol0.numpy(),oldB,200.)
        oldK=oldKm+sp.block_diag([Ks]*3,format='csr');oldvals=np.linalg.eigvalsh(oldK.toarray())
        oldtol=1e-9*max(float(abs(oldK).sum(axis=1).max()),1.)
        record['v14_material_map_extra_modes']=int(np.sum(oldvals<=oldtol))-6
        records.append(record);print(record,flush=True)
    write(BASE/'v15/support-rank-diagnostic.json',dict(completed=True,records=records,passed=all(r['extra_modes']==0 for r in records),scope=__doc__,diagnostic_only_support_guard_bypassed=True))

if __name__=='__main__':main()
