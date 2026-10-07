"""Offline, atomic migration audit; no automatic adaptation during Newton."""
import numpy as np
import scipy.linalg as la

from .model import DynamicState, quadrature


def migrate(old_model, new_model, state, *, field_atol=1e-9, energy_rtol=1e-5,
            energy_atol=1e-10, momentum_atol=1e-9):
    """Only return a proposed state after common-probe and conservation checks.

    Energy is evaluated with each rule separately; signed and absolute jumps
    are returned. Caller commits model/state/rule together AFTER this call.
    Nonrecoverable prestress/internal histories are not accepted by this API.
    """
    if old_model.density != new_model.density:
        raise ValueError('migration must preserve physical density')
    X, _ = quadrature((7, 4, 4))
    old = old_model.space.kinematics(X, state.q, state.velocity)
    N, D = new_model.space.basis(X)
    A = np.vstack((N, D.transpose(0, 2, 1).reshape(-1, new_model.space.n)))
    rhs = np.vstack((old['x']-X, (old['F']-np.eye(3)).transpose(0, 2, 1).reshape(-1, 3)))
    q = la.lstsq(A, rhs, cond=1e-12)[0]
    # Preserve both velocity and its reference gradient, not only kinetic norm.
    oldN, oldD = old_model.space.basis(X)
    gradv = np.einsum('pnj,ni->pij', oldD, state.velocity)
    vrhs = np.vstack((oldN@state.velocity, gradv.transpose(0, 2, 1).reshape(-1, 3)))
    velocity = la.lstsq(A, vrhs, cond=1e-12)[0]
    proposed = DynamicState(q, velocity, state.time, state.step)
    new = new_model.space.kinematics(X, q, velocity)
    errors = {name: float(np.max(abs(new[name]-old[name]))) for name in ('x', 'F', 'v', 'C')}
    from ..carrier_driven import displacement
    from ..endpoint_boundary import prescribed_speed
    fixed, lift = new_model.space.fixed, new_model.space.lift
    errors['boundary'] = max(float(np.max(abs((q-lift*displacement(state.time))[fixed]))),
                             float(np.max(abs((velocity-lift*prescribed_speed(state.time))[fixed]))))
    p0, l0 = old_model.momentum(state); p1, l1 = new_model.momentum(proposed)
    U0 = old_model.evaluate(state.q)['U']; U1 = new_model.evaluate(q)['U']
    K0 = old_model.kinetic(state.velocity); K1 = new_model.kinetic(velocity)
    jumps = dict(material_J=U1-U0, stabilization_J=0., kinetic_J=K1-K0)
    change_kind = 'space' if old_model.space.version != new_model.space.version else 'rule'
    report = dict(kind=change_kind, from_space=old_model.space.version, to_space=new_model.space.version,
                  from_rule=old_model.rule_version, to_rule=new_model.rule_version, errors=errors,
                  signed_jumps=jumps, absolute_jumps={k: abs(v) for k, v in jumps.items()},
                  linear_momentum_error=float(la.norm(p1-p0)), angular_momentum_error=float(la.norm(l1-l0)))
    passed = (max(errors.values()) <= field_atol and
              max(la.norm(p1-p0), la.norm(l1-l0)) <= momentum_atol and
              abs(U1-U0) <= energy_atol+energy_rtol*max(abs(U0), abs(U1)) and
              abs(K1-K0) <= energy_atol+energy_rtol*max(abs(K0), abs(K1)))
    report['accepted'] = bool(passed)
    return (proposed if passed else None), report
