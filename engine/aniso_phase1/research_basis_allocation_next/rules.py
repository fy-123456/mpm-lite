"""Exact adjacent-grid material permissions and authenticated rule-only branches."""
import copy
import numpy as np


def contiguous_slice(proposed, certified, atol=1e-10):
    a=np.asarray(proposed,dtype=float);b=np.asarray(certified,dtype=float)
    if a.ndim!=1 or b.ndim!=1 or len(a)<2 or len(b)<len(a):return False
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(np.diff(a)<=0) or np.any(np.diff(b)<=0):return False
    starts=np.where(abs(b-a[0])<=atol)[0]
    return any(i+len(a)<=len(b) and np.max(abs(b[i:i+len(a)]-a))<=atol for i in starts)


def permission(q,cfg,*,numerical_sources=None):
    if q.get('schema')!='basis-allocation-q5-qualification-v1':return False,'certificate has no current explicit path contract'
    if not q.get('qualified',False):return False,'qualification not passed'
    scope=q['scope']
    if q.get('numerical_source_sha256')!=numerical_sources:return False,'numerical implementation differs from certificate'
    if scope['physical_space_sha256']!=cfg['physical_space']['sha256']:return False,'space differs'
    if scope['mass_order']!=cfg['mass_order'] or scope['full_order']!=cfg['post_release']['full_order']:return False,'quadrature model differs'
    if scope['peak_m']!=cfg['scenario']['peak_m'] or scope['fiber_angle_degrees']!=cfg['scenario']['fiber_angle_degrees']:return False,'loading/material differs'
    if not contiguous_slice(cfg['times'],scope['time_grid_s']):return False,'not a contiguous certified time path'
    entry=cfg.get('initial_state')
    if entry is None:
        if cfg['times'][0]!=0 or not scope.get('rest_start',False):return False,'authenticated window initial state required'
    else:
        allowed=scope.get('initial_states',[])
        if not any(x['sha256']==entry['sha256'] and abs(x['time_s']-cfg['times'][0])<=1e-10 for x in allowed):
            return False,'initial-state provenance outside certificate'
    return True,'qualified exact trajectory/window'


def rule_branch(state,compact,full,cfg,source):
    """New state, never rewriting the source; all original histories remain owned."""
    full.validate(state,material=True)
    if compact.identity.keys()!=full.identity.keys() or any(compact.identity[k]!=full.identity[k] for k in full.identity if k!='material'):
        raise ValueError('only material quadrature may change in a rule branch')
    if not np.array_equal(compact.M,full.M):raise ValueError('rule branch changed mass')
    delta=float(compact.evaluate(state.q)['U']-full.evaluate(state.q)['U'])
    cumulative=state.child_states.get('cumulative_abs_rule_switch_error_J',0.)+abs(delta)
    budget=cfg['acceptance']['energy_fraction']*cfg['acceptance']['energy_scale_J']
    if not np.isfinite(delta) or cumulative>budget:raise ValueError('branch material energy exceeds budget')
    result=state.clone();result.child_states['identity']=copy.deepcopy(compact.identity)
    result.child_states['cumulative_abs_rule_switch_error_J']=cumulative
    result.child_states['material_scope_bridge']=dict(source,source_state_digest=state.digest(),material_energy_difference_J=delta,
        q_v_predictor_unchanged=True,source_model=copy.deepcopy(full.identity),target_model=copy.deepcopy(compact.identity),new_branch=True)
    compact.validate(result,material=True)
    return result
