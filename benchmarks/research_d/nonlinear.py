"""Same-state production nonlinear checks and complete representative scenes.

This is evidence for the existing moving solver, NOT a tensor-GPU dynamic port.
"""
from __future__ import annotations
import gc
import time
import numpy as np
import warp as wp
from engine.aniso_phase1.research_d.identity import digest
from benchmarks.research_d.protocol import GATES


def snapshot(s):
    fields={k:getattr(s,k).numpy().copy() for k in ('ptc_x','ptc_v','ptc_F','ptc_C','ptc_A0')}
    fields.update(sim_steps=s.sim_steps,sim_time=float(s.sim_time))
    if int(s.bcn):
        fields['committed_F']=s.aniso_committed_F[:,:int(s.bcn)].numpy().copy()
        fields['state_valid']=s.aniso_state_valid[:,:int(s.bcn)].numpy().copy()
    return fields


def implicit_checks(device):
    from engine.aniso_phase1.operator_probe import SparseProbe
    records=[]; raw={}
    for name,F in [('identity',np.eye(3)),('compression',np.diag([.65,.85,1.])),
                   ('repeated',np.diag([1.1,1.1,1.1]))]:
        p=SparseProbe(device,200.,.005); start=time.perf_counter()
        from engine.aniso_phase1.research_d.snapshot import canonical_variational_check
        r,matrices=canonical_variational_check(p,F); mass=matrices['mass']
        raw[name+'_exact']=matrices['exact']; raw[name+'_modified']=matrices['modified']; raw[name+'_mass']=mass
        raw[name+'_nodes']=matrices['nodes']; raw[name+'_velocity']=matrices['velocity']
        r.update(name=name,input_sha256=r['canonical_input_sha256'],device=device,seconds=time.perf_counter()-start,
                 passed=bool(min(r['potential_fd_errors'].values())<=GATES['energy_fd_relative'] and
                     min(r['tangent_fd_errors'].values())<=GATES['tangent_fd_relative'] and
                     r['exact']['symmetry_relative']<=GATES['symmetry_relative']))
        if name=='compression':
            r['passed'] &= r['exact']['min_eigenvalue']<0 and r['modified']['min_eigenvalue']>0
            p.set_deformation(F)
            accepted=bool(p.s.step(max_iters=80,max_cg_iters=500,cg_tol=1e-5,cg_atol=1e-12,newton_atol=1e-9,print_every=0))
            trace=p.s.last_newton_trace
            r['nonlinear']=dict(accepted=accepted,stats=p.s.last_step_stats,trace=trace,
                energy_descent=all(t['slope']<0 and t['potential_after']<=t['potential_before']+
                                  1e-4*t['alpha']*t['slope']+t['armijo_slack'] for t in trace))
            r['passed'] &= accepted and r['nonlinear']['energy_descent'] and p.s.last_step_stats['curvature_failures']>0
            raw['compression_final_x']=p.s.ptc_x.numpy(); raw['compression_final_F']=p.s.ptc_F.numpy()
        records.append(r); del p; gc.collect()
    p=SparseProbe(device,200.,.005)
    p.set_deformation(np.diag([-1.,1.,1.]))
    before=snapshot(p.s)
    accepted=bool(p.s.step(max_iters=5,print_every=0))
    after=snapshot(p.s)
    identical=all(np.array_equal(before[k],after[k]) for k in before)
    records.append(dict(name='invalid_jacobian_rollback',accepted=accepted,state_unchanged=identical,
                        before_sha256=digest(before),after_sha256=digest(after),passed=not accepted and identical))
    # Inject failure on a valid frozen state too: zero Newton budget must not commit.
    p.set_deformation(np.eye(3)); before=snapshot(p.s)
    accepted=bool(p.s.step(max_iters=0,print_every=0)); after=snapshot(p.s)
    identical=all(np.array_equal(before[k],after[k]) for k in before)
    records.append(dict(name='zero_budget_rollback',accepted=accepted,state_unchanged=identical,
                        before_sha256=digest(before),after_sha256=digest(after),passed=not accepted and identical))
    del p; gc.collect()
    return records,raw


def scene_run(device,name='fixed',steps=60,samples=2,angle=45.,*,dt=.0005,speed=.01):
    from demos.aniso import Config,Scene
    from engine.aniso_phase1.history_increment import material_response
    config=Config(scene=name,grid=9 if name=='tensile' else 8,dt=dt,
        fiber_angle=angle,samples=samples,loading_time=.01,loading_speed=.01,
        residual_atol=1e-9,v_tol=1e-9)
    scene=Scene(config,device); rows=[]; frames=[]; stresses=[]
    # Tensile uses an explicit load / hold / unload / end hold schedule below.
    from engine.aniso_phase1.tensile import set_grip_velocity,grip_reactions
    start=time.perf_counter()
    for i in range(steps):
        if name=='tensile':
            phase=min(i//(steps//4),3); velocity=(speed,0.,-speed,0.)[phase]
            set_grip_velocity(scene.solver,scene.boundary_indices,velocity)
            accepted=bool(scene.solver.step(max_iters=24,max_cg_iters=1000,cg_tol=1e-5,
                cg_atol=1e-12,newton_atol=1e-9,print_every=0))
            reaction=grip_reactions(scene.solver) if accepted else {}
        else:
            phase=0; accepted=scene.step(); reaction={}
        s=scene.solver; x,F=scene.frame()
        finite=bool(np.isfinite(x).all() and np.isfinite(F).all()); minimum=float(np.linalg.det(F).min())
        metrics=dict(step=i+1,phase=phase,accepted=accepted,finite=finite,min_det=minimum,
            residual=float(s.last_step_stats['last_residual_norm']),newton_iterations=s.last_step_stats['newton_iterations'],
            step_seconds=s.last_step_stats['step_seconds'],**reaction)
        if name=='tensile':
            right=scene.reference[:,0]>=.75;left=scene.reference[:,0]<=.25
            commanded=speed*config.dt*max(0,min(i+1,steps//4,3*(steps//4)-(i+1)))
            measured=float((x-scene.reference)[right,0].mean()-(x-scene.reference)[left,0].mean())
            metrics.update(commanded_displacement_m=commanded,measured_grip_displacement_m=measured,
                           grip_displacement_error_m=abs(measured-commanded))
        rows.append(metrics); frames.append(x.copy())
        if finite and minimum>0:
            _,P=material_response(F,s.ptc_A0.numpy(),s.aniso_params); stresses.append(P)
        if not accepted or not finite or minimum<=0:break
    arrays=dict(positions=np.array(frames),stresses=np.array(stresses),final_F=F.copy(),reference=scene.reference.copy())
    result=dict(name=name,device=device,steps_requested=steps,steps_completed=len(rows),samples=samples,
        particles=len(scene.reference),angle=angle,dt=config.dt,speed=speed,rows=rows,seconds=time.perf_counter()-start,
        passed=bool(len(rows)==steps and all(r['accepted'] and r['finite'] and r['min_det']>0 for r in rows)),
        algorithm='unchanged production moving solver; not the new tensor backend',
        scope='device equivalence and stable cycle; not temporal convergence certification')
    del scene; gc.collect(); return result,arrays
