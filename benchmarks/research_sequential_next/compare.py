"""Physical comparisons at shared times and impulse-conserving intervals."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from .checkpoint import GenerationStore
from .provenance import read,write,serial_lock


def impulse_average(rows,start,end,key='reaction_N'):
    """Require exact interval coverage; never interpolate a per-step force."""
    selected=[r for r in rows if r['time']>start+1e-11 and r['time']<=end+1e-11]
    cursor=start;impulse=0.
    for row in selected:
        left=row['time']-row['dt']
        if abs(left-cursor)>1e-10:raise ValueError('reaction intervals do not exactly cover comparison range')
        impulse+=row['dt']*row[key];cursor=row['time']
    if abs(cursor-end)>1e-10:raise ValueError('reaction interval endpoint missing')
    return impulse/(end-start)


def metric(a,b,absolute,relative,weights=None):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('invalid field comparison')
    if weights is None:
        error=float(np.linalg.norm((a-b).ravel()));scale=float(np.linalg.norm(b.ravel()))
    else:
        weights=np.asarray(weights,dtype=float)
        if not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:raise ValueError('invalid physical weights')
        extra=tuple(range(weights.ndim,a.ndim))
        delta=(a-b)**2;value=b**2
        if extra:delta=delta.sum(axis=extra);value=value.sum(axis=extra)
        error=float(np.sqrt(np.sum(weights*delta)/weights.sum()))
        scale=float(np.sqrt(np.sum(weights*value)/weights.sum()))
    return dict(absolute=error,reference_norm=scale,relative=error/scale if scale>1e-14 else None,
        near_zero=scale<=absolute,budget=absolute+relative*scale,passed=error<=absolute+relative*scale)


def regions(X):
    axes=[X[:,0,0,0],X[0,:,0,1],X[0,0,:,2]];weights=np.ones(X.shape[:-1])
    for axis,x in enumerate(axes):
        w=np.empty_like(x);w[0]=(x[1]-x[0])/2;w[-1]=(x[-1]-x[-2])/2
        w[1:-1]=(x[2:]-x[:-2])/2
        shape=[1,1,1];shape[axis]=len(x);weights*=w.reshape(shape)
    fraction=(X[...,0]-axes[0][0])/(axes[0][-1]-axes[0][0])
    return dict(global_domain=weights,clamps=weights*((fraction<=.125)|(fraction>=.875)),
        transition=weights*(((fraction>.125)&(fraction<=.25))|((fraction>=.75)&(fraction<.875))),
        interior=weights*((fraction>.25)&(fraction<.75)))


def compare_cases(run,coarse,fine):
    from .run import load_model,probe_frame
    run=Path(run);a=run/'cases'/coarse;b=run/'cases'/fine
    ca=read(a/'execution-protocol.json');cb=read(b/'execution-protocol.json')
    for key in ('probe_shape','mass_order','physics'):
        if ca[key]!=cb[key]:raise ValueError('incompatible physical evaluation setting: '+key)
    ma,mb=read(a/'identity.json'),read(b/'identity.json')
    if ma['model']['model']!=mb['model']['model'] or ma['model']['boundary']!=mb['model']['boundary']:
        raise ValueError('different physical space or boundary')
    ha=GenerationStore(a,ma).history();hb=GenerationStore(b,mb).history()
    if ha[0]['state'].digest()!=hb[0]['state'].digest():
        # Backend/rule identities differ; compare only explicit physical arrays
        sa,sb=ha[0]['state'],hb[0]['state']
        if sa.time!=sb.time or not np.array_equal(sa.q,sb.q) or not np.array_equal(sa.velocity,sb.velocity):
            raise ValueError('different physical initial states')
    lookup={round(x['state'].time,10):x for x in hb}
    if any(round(x['state'].time,10) not in lookup for x in ha):raise ValueError('fine history misses coarse comparison time')
    # Field recovery is independent of the material integration rule and has no
    # dynamics or quadrature evaluation; CPU avoids allocating another GPU rule.
    field_config=dict(ca,device='cpu')
    if 'implementation' in ca:
        field_config['implementation']=dict(ca['implementation'],operator='original',linearization_cache=False)
    model,_=load_model(run,field_config);budget=ca['acceptance'];records=[]
    direction=np.array(model.parent.params.fiber_direction,dtype=float,copy=True)
    direction/=np.linalg.norm(direction)
    for item in ha:
        left=item['state'];right=lookup[round(left.time,10)]['state']
        fa=probe_frame(model,left,ca['probe_shape']);fb=probe_frame(model,right,ca['probe_shape'])
        if not np.array_equal(fa['X'],fb['X']):raise ValueError('different probe positions')
        weights=regions(fa['X']);row=dict(time=left.time,regions={})
        for region,w in weights.items():
            data={}
            for key,unit_absolute in [('x',budget['displacement_atol_m']),('velocity',budget['physical_velocity_atol_m_s']),('PK1',budget['PK1_atol_Pa'])]:
                aa,bb=fa[key],fb[key]
                if key=='x':aa=aa-fa['X'];bb=bb-fb['X']
                data[key]=metric(aa,bb,unit_absolute,budget['field_rtol'],w)
            aa=np.einsum('i,...ij,j->...',direction,fa['PK1'],direction)
            bb=np.einsum('i,...ij,j->...',direction,fb['PK1'],direction)
            data['fiber_PK1']=metric(aa,bb,budget['PK1_atol_Pa'],budget['field_rtol'],w)
            row['regions'][region]=data
        records.append(row)
    reaction=[];ra=ha[-1]['rows'];rb=hb[-1]['rows']
    for row in ra:
        start=row['time']-row['dt'];end=row['time']
        ref=impulse_average(rb,start,end)
        reaction.append(dict(start=start,end=end,**metric(row['reaction_N'],ref,budget['reaction_atol_N'],budget['field_rtol'])))
    result=dict(coarse=coarse,fine=fine,field_records=records,reaction_intervals=reaction,
        passed=all(m['passed'] for r in records for region in r['regions'].values() for m in region.values())
            and all(r['passed'] for r in reaction),
        scope='finite-grid physical comparison; no asymptotic convergence or spatial certification')
    write(run/'comparisons'/f'{coarse}--{fine}.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--coarse',required=True);p.add_argument('--fine',required=True);a=p.parse_args()
    with serial_lock(a.run):
        result=compare_cases(a.run,a.coarse,a.fine)
        print('comparison passed:',result['passed'],flush=True)
