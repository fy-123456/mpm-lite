"""Legacy pre-guard rigid-translation diagnostic, not a stiffness certificate.

Delivered guarded solver is expected to reject step 2 (64 carriers, 80 nodes).
The archived 180-step record used the separately sealed pre-guard source.
Use aniso_v15_support_rank for the two-step unrestricted counterexample and
test_support_expansion_rejected_before_physical_commit for delivered behavior.
"""
import hashlib,itertools
import numpy as np
import warp as wp
from benchmarks.aniso_material_history import BASE,write
from demos.aniso import Config
from engine.aniso_phase1.compatible_patch import CompatiblePatchLiteSolver
from benchmarks.aniso_compatible_history import CommonLedger as MaterialLedger
import json

def main():
    wp.config.kernel_cache_dir=json.loads((BASE/'v15/storage-protocol.json').read_text())['temporary']+'/mpm-lite-warp-cache';h=1/16;dt=.005;v=np.array([.1,.025,-.02]);x=np.array(list(itertools.product(.34375+np.arange(4)*h/2,repeat=3)))
    s=CompatiblePatchLiteSolver((17,)*3,params=Config('tensile',17,dt,45.).params,dx=h,device='cpu',gravity=0.,ppc=1,flip_ratio=.9**(dt/.001),energy_diagnostics=True,force_discretization='variational',history_mode='particle_resample',boundary_impulse_transfer=True,apic_transfer='incremental',affine_flip_ratio=1.,velocity_dissipation='null')
    s.seed_particles(x,density=1.,vol0=h**3/8,velocity=np.broadcast_to(v,x.shape));s.set_dt(dt);s.energy_ledger=MaterialLedger();s.energy_ledger.begin(s);rows=[];support=None;changes=0;Y0=None
    for i in range(180):
        s.energy_ledger.audit_next=i in (0,59,119,179);assert s.step(max_iters=16,print_every=0,v_tol=1e-10,cg_tol=1e-4,cg_atol=1e-12,newton_atol=1e-10,max_cg_iters=1000,linear_solver='pcg',reaction_force_atol=1e-7),s.last_step_stats
        e=s.enhancements
        if Y0 is None:Y0=e.last_origin.copy()
        new_support=set(map(tuple,s.mapped_nodes));changes+=int(support is not None and support!=new_support);support=new_support
        r=dict(step=i+1,x_error=float(np.max(abs(s.ptc_x.numpy()-(x+(i+1)*dt*v)))),F_error=float(np.max(abs(s.ptc_F.numpy()-np.eye(3)))),v_error=float(np.max(abs(s.ptc_v.numpy()-v))),C_error=float(np.max(abs(s.ptc_C.numpy()))),carrier_error=float(np.max(abs(e.origin.numpy()-(Y0+(i+1)*dt*v)))),energy_rebuild_J=abs(s.energy_ledger.rows[-1]['stabilization_rebuild_delta']),stage_budget_J=abs(s.energy_ledger.rows[-1]['stage_budget_error']))
        rows.append(r)
    maxima={k:max(r[k] for r in rows) for k in rows[0] if k!='step'};assert changes>0 and max(maxima.values())<1e-10,maxima
    write(BASE/'v15/rigid-support-crossing.json',dict(passed=True,steps=180,rigid_translation_cells=(180*dt*v/h).tolist(),support_changes=changes,errors=maxima,source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),scope='actual moving particles and virtual markers; active-node membership changes; no prescribed-position reset'))
    print(changes,maxima,flush=True)

if __name__=='__main__':main()
