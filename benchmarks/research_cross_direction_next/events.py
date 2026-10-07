"""Raw events with machine-scale boundary allowance and anchored cycle matching."""
import copy
import numpy as np
from benchmarks.research_basis_allocation_next.time_study import events,difference as legacy_difference
KINDS=('maximum','minimum','up_zero','down_zero')

def allowance(a,b):
    values=[1.]+[abs(float(t)) for s in (a,b) for t in s.get('raw_times',[])]+[abs(float(t)) for s in (a,b) for e in s['events'] for t in e['bracket_s']]
    return 32*np.finfo(float).eps*max(values)

def roundoff_only(a,b,budget=.00625):
    result=copy.deepcopy(legacy_difference(a,b,budget));tol=allowance(a,b)
    for pair in result['pairs']:
        lo,hi=pair['offset_interval_s'];pair['resolved_within_budget']=bool(lo>=-budget-tol and hi<=budget+tol and not pair['ambiguous'])
    good=a['resolved'] and b['resolved'] and result['same_event_counts'] and result['pairs'] and result['unambiguous'] and all(p['resolved_within_budget'] for p in result['pairs'])
    result.update(status='passed_scoped' if good else 'unresolved_or_shifted',roundoff_allowance_s=tol)
    return result

def difference(a,b,budget=.00625):
    """One-to-one ordered matching anchored to the common initial history.

    Each match must be unique within the smaller of the physical offset budget
    and 45% of adjacent same-kind event spacing. Wide/aliased brackets remain
    unresolved; no smoothing, deletion or cycle shifting is allowed.
    """
    if budget<=0 or not np.isfinite(budget):raise ValueError('positive finite event budget required')
    tol=allowance(a,b);pairs=[];counts=True;unique=True
    for kind in KINDS:
        aa=[v for v in a['events'] if v['kind']==kind];bb=[v for v in b['events'] if v['kind']==kind]
        counts &=len(aa)==len(bb)
        def guard(seq,j):
            gaps=[seq[k+1]['time_s']-seq[k]['time_s'] for k in (j-1,j) if 0<=k<len(seq)-1]
            return min([budget]+[.45*g for g in gaps])
        for j,(x,y) in enumerate(zip(aa,bb)):
            gate=min(guard(aa,j),guard(bb,j));lo=x['bracket_s'][0]-y['bracket_s'][1];hi=x['bracket_s'][1]-y['bracket_s'][0]
            alternatives=[k for k,v in enumerate(bb) if x['bracket_s'][0]-v['bracket_s'][1]<=gate+tol and x['bracket_s'][1]-v['bracket_s'][0]>=-gate-tol]
            reverse=[k for k,v in enumerate(aa) if v['bracket_s'][0]-y['bracket_s'][1]<=gate+tol and v['bracket_s'][1]-y['bracket_s'][0]>=-gate-tol]
            unambiguous=alternatives==[j] and reverse==[j];unique &=unambiguous
            pairs.append(dict(kind=kind,offset_estimate_s=x['time_s']-y['time_s'],offset_interval_s=[lo,hi],match_guard_s=gate,
                ambiguous=not unambiguous,resolved_within_budget=bool(unambiguous and lo>=-budget-tol and hi<=budget+tol)))
    passed=bool(a['resolved'] and b['resolved'] and counts and pairs and unique and all(x['resolved_within_budget'] for x in pairs))
    return dict(status='passed_scoped' if passed else 'unresolved_or_shifted',same_event_counts=bool(counts),unambiguous=bool(unique),pairs=pairs,
        roundoff_allowance_s=tol,budget_s=budget,phase_alignment_applied=False,method='same-kind order; unique bilateral match with local-period guard')
