"""Bounded exact-matrix LU reuse for the original AVF search, without SPD claims."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import copy
from pathlib import Path
import time
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator,gmres
from ..research_d.stage2.contracts import array_digest
from ..research_d.identity import digest
from .integrator import ValidatedAVF


class StaticFactors:
    def __init__(self,*,max_entries=8,max_bytes=64<<20):
        if max_entries<1 or max_bytes<1:raise ValueError('positive factor cache limits required')
        self.max_entries=max_entries;self.max_bytes=max_bytes;self.entries=OrderedDict()
        self.bytes=0;self.peak_bytes=0;self.hits=0;self.misses=0;self.evictions=0
        self.factor_seconds=0.;self.lookup_seconds=0.
        self.source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def get(self,matrix,dt,model_identity):
        started=time.perf_counter();H=np.asarray(matrix)
        if H.ndim!=2 or H.shape[0]!=H.shape[1] or not np.isfinite(H).all() or not np.isfinite(dt) or dt<=0:
            raise ValueError('finite square search matrix and positive dt required')
        key=digest(dict(model=model_identity,dt=float(dt).hex(),matrix=array_digest(H),source=self.source))
        self.lookup_seconds+=time.perf_counter()-started
        if key in self.entries:
            self.entries.move_to_end(key);self.hits+=1;return self.entries[key]
        self.misses+=1
        needed=H.nbytes+H.shape[0]*8
        if needed>self.max_bytes:raise MemoryError('factor exceeds cache budget')
        while self.entries and (len(self.entries)>=self.max_entries or self.bytes+needed>self.max_bytes):
            _,value=self.entries.popitem(last=False);self.bytes-=sum(a.nbytes for a in value);self.evictions+=1
        begun=time.perf_counter();factor=la.lu_factor(H,check_finite=True);self.factor_seconds+=time.perf_counter()-begun
        if not np.isfinite(factor[0]).all() or np.any(np.diag(factor[0])==0):raise ValueError('singular AVF search matrix')
        for a in factor:a.setflags(write=False)
        self.entries[key]=factor;self.bytes+=sum(a.nbytes for a in factor);self.peak_bytes=max(self.peak_bytes,self.bytes)
        return factor

    def report(self):
        return dict(hits=self.hits,misses=self.misses,evictions=self.evictions,entries=len(self.entries),
            bytes=self.bytes,peak_bytes=self.peak_bytes,max_bytes=self.max_bytes,
            factor_seconds=self.factor_seconds,lookup_seconds=self.lookup_seconds)


class ReusedAVF(ValidatedAVF):
    """Only the factor provider differs from the sealed original AVF body.

    The derived _advance below is a source copy, rather than a runtime patch.
    Its unchanged remainder is checked against the sealed method by the tests.
    """
    def __init__(self,model,config,state=None):
        self.factors=StaticFactors()
        super().__init__(model,config,state)

    def _factor(self,H,dt):
        return self.factors.get(H,dt,self.model.signature)

    def _advance(self, s, dt, max_iters, external_force, inject):
        start = time.perf_counter()
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('positive finite step required')
        m = self.model; m.validate(s)
        t1 = s.time+dt
        if m.boundary.hold is None and t1 > 1.6+1e-12:
            raise ValueError('step exceeds frozen cycle endpoint')
        if inject is not None:
            inject('before_material', s)
        initial = m.evaluate(s.q); K0 = m.kinetic(s.velocity)
        Wb = (m.boundary.lift(t1)-m.boundary.lift(s.time))/dt
        W = Wb.copy()
        W[m.free] = s.velocity[m.free] if s.predictor is None else s.predictor[m.free]
        ext = np.zeros_like(W) if external_force is None else np.array(external_force, dtype=float)
        if ext.shape != W.shape or not np.isfinite(ext).all():
            raise ValueError('finite interval generalized force required')
        scale = max(la.norm((m.M@s.velocity)[m.free]),
                    dt*la.norm((initial['force']-ext)[m.free]), 1e-8)
        tolerance = self.atol+self.rtol*scale
        H = 2*m.M3ff
        if m.rest_K is not None:
            H = H+.5*dt**2*m.rest_K[np.ix_(m.ids, m.ids)]
        factor = self._factor(H, dt)  # same matrix; only its exact LU is reused
        calls = initial['material_calls']; backs = 0; krylov = 0

        def residual(velocity):
            nonlocal calls
            out = self.path(s.q, velocity, dt)
            calls += out['material_calls']
            impulse = 2*m.M@(velocity-s.velocity)+dt*(out['force']-ext)
            return impulse[m.free].ravel(), out, impulse

        r, out, midpoint = residual(W)
        for iteration in range(max_iters+1):
            norm = float(la.norm(r))
            if norm <= tolerance:
                break
            if iteration == max_iters:
                raise ValueError(f'true AVF residual {norm:g} exceeds {tolerance:g}')
            if inject is not None:
                inject('before_search', s)
            if iteration < 4 and m.rest_K is not None:
                update = la.lu_solve(factor, -r)
            else:
                def action(vector):
                    nonlocal calls
                    d = np.zeros_like(W); d[m.free] = vector.reshape(-1, 3)
                    p = self.path(s.q, W, dt, d); calls += p['material_calls']
                    return (2*m.M@d+dt**2*p['action'])[m.free].ravel()
                def count(_):
                    nonlocal krylov
                    krylov += 1
                A = LinearOperator(H.shape, matvec=action, dtype=float)
                P = LinearOperator(H.shape, matvec=lambda x: la.lu_solve(factor, x), dtype=float)
                update, info = gmres(A, -r, M=P, atol=0., rtol=1e-5,
                    restart=20, maxiter=3, callback=count, callback_type='pr_norm')
                if info:
                    raise ValueError(f'exact tangent GMRES failed: {info}')
            for ls in range(12):
                candidate = W.copy(); candidate[m.free] += (2.**-ls)*update.reshape(-1, 3)
                try:
                    newr, newout, newmid = residual(candidate)
                except ValueError:
                    continue
                if la.norm(newr) < norm:
                    W, r, out, midpoint = candidate, newr, newout, newmid
                    backs += ls
                    break
            else:
                raise ValueError('residual line search failed')

        q1 = s.q+dt*W
        velocity, endpoint, end = m.endpoint(2*W-s.velocity, t1)
        candidate = s.clone()
        candidate.q, candidate.velocity = q1, velocity
        candidate.time, candidate.step = t1, s.step+1
        candidate.predictor = W.copy()
        m.validate(candidate)
        final = m.evaluate(q1); calls += final['material_calls']
        K1 = m.kinetic(velocity)
        midpoint_work = float(np.sum(midpoint*Wb))
        external_work = dt*float(np.sum(ext*W))
        path_error = final['U']-initial['U']-dt*float(np.sum(out['force']*W))
        residual_work = float(np.sum(midpoint*(W-Wb)))
        delta = K1+final['U']-K0-initial['U']
        balance = delta-midpoint_work-end['endpoint_boundary_work_J']-external_work+end['constraint_kinetic_loss_J']
        closure = balance-path_error-residual_work
        if not np.isfinite(closure) or abs(closure) > self.ledger_atol:
            raise ValueError('step energy ledger failed')
        if abs(end['endpoint_identity_error_J']) > self.ledger_atol or end['endpoint_free_impulse'] > 1e-9:
            raise ValueError('endpoint impulse identity failed')
        boundary = m.boundary.unit
        reaction = lambda f: float(np.sum(f*boundary))
        P0, _ = m.wrench(m.M@s.velocity); P1, L1 = m.wrench(m.M@velocity, q1)
        total_impulse, _ = m.wrench(midpoint+endpoint+dt*ext)
        row = dict(step=candidate.step, time=t1, dt=dt,
            reaction_N=reaction(midpoint+endpoint)/dt,
            midpoint_reaction_N=reaction(midpoint)/dt, endpoint_reaction_N=reaction(endpoint)/dt,
            material_reaction_N=reaction(out['material']),
            stabilization_reaction_N=reaction(out['force']-out['material']),
            inertia_reaction_N=reaction(2*m.M@(W-s.velocity))/dt,
            external_reaction_N=-reaction(ext), material_J=final['material_U'],
            stabilization_J=final['stabilization_U'], kinetic_J=K1, total_J=K1+final['U'],
            delta_total_J=delta, midpoint_boundary_work_J=midpoint_work,
            boundary_work_J=midpoint_work+end['endpoint_boundary_work_J'], external_work_J=external_work,
            path_quadrature_error_J=path_error, solve_work_error_J=residual_work,
            energy_balance_J=balance, budget_defect_J=closure, **end,
            true_residual=norm, residual_tolerance=tolerance, newton_iterations=iteration,
            line_search_backtracks=backs, krylov_iterations=krylov, material_point_calls=calls,
            min_detF=min(final['min_detF'], out['min_detF']),
            linear_momentum=P1.tolist(), angular_momentum=L1.tolist(),
            linear_momentum_error=float(la.norm(P1-P0-total_impulse)),
            displacement_constraint=float(m.boundary.validate(candidate)[0]),
            velocity_constraint=float(m.boundary.validate(candidate)[1]),
            material_rule=m.rule.signature, path_order=self.path_order,
            accepted=True, wall_seconds=time.perf_counter()-start)
        candidate.child_states['last_ledger'] = copy.deepcopy(row)
        candidate.child_states['cumulative_abs_closure_J'] = (
            s.child_states.get('cumulative_abs_closure_J', 0.)+abs(closure))
        return candidate, row
