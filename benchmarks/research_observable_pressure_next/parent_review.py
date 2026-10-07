"""Compare saved full trajectories at common physical points and intervals."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import APP,APP_SHA,read,write,serial_lock
from .run import load_model
from .finalize import hist
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from benchmarks.research_sequential_next.compare import metric,regions

def review(run):
    run=Path(run);folder=run/'cases/final-full';m,_=load_model(run,read(folder/'execution-protocol.json'));cache=CachedProbes(m);old=hist(APP/'cases/final-full');new=hist(folder);lookup={round(x['state'].time,10):x for x in new};weights=regions(cache.X);direction=m.parent.params.fiber_direction
    records=[];maxima={};unchanged=0.
    for item in old:
        t=item['state'].time;current=lookup[round(t,10)];a,b=cache.frame(current['state']),cache.frame(item['state']);row={}
        for area,w in weights.items():
            row[area]={}
            for key,field,atol in [('displacement_m','x',5e-5),('velocity_m_s','velocity',1e-4),('PK1_Pa','PK1',.02),('fiber_Pa','fiber',.02)]:
                if field=='fiber':av,bv=(np.einsum('i,...ij,j->...',direction,z['PK1'],direction) for z in (a,b))
                elif field=='x':av,bv=a['x']-a['X'],b['x']-b['X']
                else:av,bv=a[field],b[field]
                r=metric(av,bv,atol,.05,w);row[area][key]=r;tag=area+'/'+key
                if tag not in maxima or r['absolute']>maxima[tag]['absolute']:maxima[tag]=dict(time_s=t,**r)
                if t<=1.075+1e-12:unchanged=max(unchanged,r['absolute'])
        records.append(dict(time_s=t,regions=row))
    reactions=[];newrows=new[-1]['rows']
    for row in old[-1]['rows']:
        end=row['time'];begin=end-row['dt'];pieces=[r for r in newrows if r['time']>begin+1e-12 and r['time']<=end+1e-12]
        if abs(sum(r['dt'] for r in pieces)-row['dt'])>1e-11:raise ValueError('common physical interval is incomplete')
        raw=sum(r['dt']*r['reaction_N'] for r in pieces)/row['dt'];endpoint=sum(r['dt']*r['endpoint_reaction_N'] for r in pieces)
        comparison=metric(raw,row['reaction_N'],1e-4,.05)
        reactions.append(dict(begin_s=begin,end_s=end,actual_subintervals=len(pieces),new_average_N=raw,parent_average_N=row['reaction_N'],new_impulse_Ns=raw*row['dt'],parent_impulse_Ns=row['reaction_N']*row['dt'],new_endpoint_impulse_Ns=endpoint,parent_endpoint_impulse_Ns=row['endpoint_reaction_N']*row['dt'],comparison=comparison))
    if unchanged>1e-8:raise ValueError('unchanged prefix has changed physical fields')
    report=dict(status='passed_scoped_review',parent_release_sha256=APP_SHA,common_nodes=len(records),common_intervals=len(reactions),unchanged_prefix_max_absolute=unchanged,maxima=maxima,
        fields=records,reactions=reactions,parent_is_not_fine_reference=True,interpretation='Post-1.075 differences are reported, not used as proof of improvement; actual local fine references and full-cycle initial states are separate gates. No phase alignment or smoothing.',
        minimum_J=dict(parent=min(r['min_detF'] for r in old[-1]['rows']),current=min(r['min_detF'] for r in newrows)),raw_reaction_peaks_N=dict(parent=max(abs(r['reaction_N']) for r in old[-1]['rows']),current_native_intervals=max(abs(r['reaction_N']) for r in newrows),current_common_intervals=max(abs(r['new_average_N']) for r in reactions)),field_budget_exceedance_count=sum(not v['passed'] for r in records for area in r['regions'].values() for v in area.values()),reaction_budget_exceedance_count=sum(not r['comparison']['passed'] for r in reactions))
    write(run/'S6/parent-scene-review.json',report);print('PARENT_COMMON_REVIEW',unchanged,report['field_budget_exceedance_count'],report['reaction_budget_exceedance_count'],report['minimum_J'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):review(a.run)
