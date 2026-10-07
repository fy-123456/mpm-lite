"""Velocity-controlled grid grips and their discrete momentum-balance reactions."""
import numpy as np
from .constitutive import pk1
from .diagnostics import center_snapshot
from engine.sp_grid import B


def loading_displacement(t, speed, loading_time, cycles=1, smooth=False):
    """Triangular displacement, then held at zero; exact at reversal/end."""
    if t < 0 or t >= 2*loading_time*cycles-1e-12:
        return 0.
    phase = t % (2*loading_time)
    if smooth:
        return .5*speed*loading_time*(1-np.cos(np.pi*phase/loading_time))
    return speed * max(0., min(phase, 2*loading_time-phase))


def grid_values(s, field, coords):
    block = coords // B
    bid = s.block2bid.numpy()[block[:, 0], block[:, 1], block[:, 2]]
    if np.any(bid < 0):
        raise RuntimeError('reaction node absent from sparse grid')
    local = coords % B
    return field[:int(s.bcn)].numpy()[bid, local[:, 0], local[:, 1], local[:, 2]]


def grip_reactions(s, left=.25, right=.75):
    """Actuator-on-body force, including C2G projection impulse.

    F_act = sum_grip [m (v_new-v_raw)/dt + sum_c V tau grad(w)].
    Elastic and inertial parts are reported separately. This is the force of
    the selected discrete residual, not a continuum traction estimate.
    """
    ledger = s.energy_ledger
    if hasattr(s, 'projected_internal_force'):
        block=ledger.nodes//B;local=ledger.nodes%B
        bid=s.block2bid.numpy()[block[:,0],block[:,1],block[:,2]]
        dof=s.node2dof.numpy()[bid,local[:,0],local[:,1],local[:,2]]
        internal=s.projected_internal_force()[dof]
    elif getattr(s,'quadrature_kind','center')=='group4x8':
        block=ledger.nodes//B;local=ledger.nodes%B
        bid=s.block2bid.numpy()[block[:,0],block[:,1],block[:,2]]
        dof=s.node2dof.numpy()[bid,local[:,0],local[:,1],local[:,2]]
        internal=ledger.group_internal[dof].copy()
    elif getattr(s,'quadrature_kind','center')=='particle':
        F=s.ptc_F.numpy();A=s.ptc_A0.numpy()
        pullback=ledger.step_start_particle_F if s.force_discretization=='variational' else F
        tau=np.array([pk1(f,a,s.aniso_params)@f0.T for f,a,f0 in zip(F,A,pullback)])
        force=np.einsum('p,pij,pqj->pqi',s.ptc_vol0.numpy(),tau,s.particle_grads.numpy())
        ids=s.particle_node_ids.numpy();valid=ids>=0
        by_dof=np.zeros((s.MAX_DOF,3));np.add.at(by_dof,ids[valid],force[valid])
        block=ledger.nodes//B;local=ledger.nodes%B
        bid=s.block2bid.numpy()[block[:,0],block[:,1],block[:,2]]
        dof=s.node2dof.numpy()[bid,local[:,0],local[:,1],local[:,2]]
        internal=by_dof[dof]
    else:
        coords, volumes, _, active = center_snapshot(s)
        n = int(s.bcn)
        F = s.aniso_committed_F[:, :n].numpy()[0].reshape(-1, 3, 3)[active]
        A = s.aniso_A0[:, :n].numpy()[0].reshape(-1, 3, 3)[active]
        pullback = ledger.step_start_F[0].reshape(-1, 3, 3)[active] if s.force_discretization == "variational" else F
        tau = np.array([pk1(f, a, s.aniso_params)@f0.T for f, a, f0 in zip(F, A, pullback)])
        corners = ledger.corners
        grad = (2*corners-1)/(4*s.dx)
        forces = np.einsum('p,pij,cj->pci', volumes, tau, grad)
        internal = np.stack([np.bincount(ledger.inverse, weights=forces[:, :, d].reshape(-1)) for d in range(3)], axis=1)
    if getattr(s,'enhancements',None) is not None:
        block=ledger.nodes//B;local=ledger.nodes%B
        bid=s.block2bid.numpy()[block[:,0],block[:,1],block[:,2]]
        dof=s.node2dof.numpy()[bid,local[:,0],local[:,1],local[:,2]]
        internal+=s.enhancements.extra.numpy()[dof]/s.dt
    vnew = grid_values(s, s.grid_v_new, ledger.nodes)
    inertia = ledger.node_m[:, None]*(vnew-ledger.raw_v)/s.dt
    result = {}
    for name, mask in [('left', ledger.nodes[:, 0]*s.dx <= left+1e-12),
                       ('right', ledger.nodes[:, 0]*s.dx >= right-1e-12)]:
        result[name+'_force'] = float((internal[mask]+inertia[mask])[:, 0].sum())
        result[name+'_elastic_force'] = float(internal[mask, 0].sum())
        result[name+'_inertial_force'] = float(inertia[mask, 0].sum())
    free = (ledger.nodes[:, 0]*s.dx > left+1e-12) & (ledger.nodes[:, 0]*s.dx < right-1e-12)
    result['free_force_residual_norm'] = float(np.linalg.norm((internal+inertia)[free]))
    result['momentum_balance_error'] = float(result['left_force']+result['right_force']-inertia[:, 0].sum())
    # Independent particle endpoint momentum; do not infer it from grid balance.
    grip = ~free
    external = (internal+inertia)[grip].sum(axis=0)
    particle_rate = (ledger.particle_momentum_end-ledger.particle_momentum_start)/s.dt
    particle_error = external-particle_rate
    result["particle_momentum_balance_error"] = float(particle_error[0])
    result["particle_momentum_balance_error_norm"] = float(np.linalg.norm(particle_error))
    for axis, value in zip("xyz", particle_error):
        result["particle_momentum_balance_error_"+axis] = float(value)
    return result


def set_grip_velocity(s, nodes, velocity):
    """Update existing grip storage without reallocating 64 boundary blocks."""
    import warp as wp
    from engine.types import vec3
    from engine.boundary_utils import paint_bc_kernel
    values = np.zeros((len(nodes), 3))
    values[nodes[:, 0]*s.dx >= .75-1e-12, 0] = velocity
    wp.launch(paint_bc_kernel, dim=len(nodes), inputs=[
        wp.array(nodes, dtype=wp.vec3i, device=s.device),
        wp.ones(len(nodes), dtype=int, device=s.device),
        wp.zeros(len(nodes), dtype=vec3, device=s.device),
        wp.array(values, dtype=vec3, device=s.device),
        s.bc_block2bid, s.bc_type, s.bc_norm, s.bc_velo], device=s.device)
