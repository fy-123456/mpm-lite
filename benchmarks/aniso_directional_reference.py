"""Original-state spatial diagnosis and factorial directional refinement."""
import argparse
import json
from pathlib import Path
import numpy as np

from engine.aniso_phase1.directional_reference import directional_geometry
from engine.aniso_phase1.convergence_reference import (
    geometry,knots,union_knots,tensor_rule,load_fields,directions,rms)
from engine.aniso_phase1.consistent_transfer import material_response
from engine.aniso_phase1.history_increment import material_tangent,ReferenceBasis,HistoryField,frozen
from .aniso_history_increment import guard,DATA_ROOT
from .aniso_convergence_reference import save,fingerprint
from .aniso_refinement import OLD,field_axes
from .aniso_spatial_energy import run as run_path,cached_compare
from .aniso_reference_scale import OUT as PREVIOUS,candidate_fields,scaled_differences

OUT=Path('docs/results/directional-reference/v1')
CASES=dict(base=(80,20,20),axial=(88,20,20),cross=(80,22,22),both=(88,22,22))
TIME_TARGET=.05
SPACE_TARGET=.1


def initial():return load_fields(OLD/'initial-switch-state.npz')


def label(case):return '161' if case=='base' else 'x'.join(map(str,CASES[case]))


def catalog(out,case):
    result={}
    for folder in ([PREVIOUS,out] if case=='base' else [out]):
        for p in (folder/'original').glob(f'g{label(case)}-dt*.json'):
            m=json.loads(p.read_text());c=m['config']
            if c['initial']!=fingerprint(initial()) or not all(m['checks'].values()):
                raise ValueError(f'initial/check mismatch: {p}')
            if c['duration']==.003 and p.with_suffix('.npz').exists():result[c['dt']]=(p,m)
    return result


def fields(item):return load_fields(item[0].with_suffix('.npz'))


def run(out,case,dt,device):
    import warp as wp
    from utils.resource_guard import prepare_warp_cache
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)
    source=directional_geometry(CASES[case]) if case!='base' else None
    return run_path(out,'original',161 if case=='base' else label(case),dt,device,
                    initial=initial(),source=source)


def location(out,a,b,name,order=3,edges=None):
    """Localize reference changes; component shares are not mesh causality."""
    signature=dict(a=fingerprint(a),b=fingerprint(b),order=order)
    if edges is not None:signature['edges']=[np.asarray(e).tolist() for e in edges]
    path=out/f'location-{name}.json'
    if path.exists():
        m=json.loads(path.read_text())
        if m['signature']==signature:return m
    if edges is None:edges=[np.linspace(.25,.75,33),np.linspace(.4375,.5625,9),np.linspace(.4375,.5625,9)]
    axes=union_knots(field_axes(a,b),edges);X,w=tensor_rule(axes,order)
    totals={k:np.zeros((3,3) if k in ('F','P') else (3,)) for k in ('x','F','P','v')}
    bins={k:[np.zeros(len(e)-1) for e in edges] for k in totals}
    volumes=[np.zeros(len(e)-1) for e in edges]
    region={key:{k:0. for k in totals} for key in ('clamp','switch','bulk')};volume={key:0. for key in region}
    for start in range(0,len(X),32768):
        guard();p=X[start:start+32768];weights=w[start:start+32768];A=directions(p)
        values=[]
        for f in (a,b):
            x,F=f['position'].evaluate(p);v=f['velocity'].evaluate(p)[0]
            values.append(dict(x=x,F=F,P=material_response(F,A,geometry(17).params)[1],v=v))
        masks=dict(clamp=p[:,0]<.3125,switch=(p[:,0]>=.4375)&(p[:,0]<=.5625))
        masks['bulk']=~(masks['clamp']|masks['switch'])
        ids=[np.clip(np.searchsorted(e,p[:,d],side='right')-1,0,len(e)-2) for d,e in enumerate(edges)]
        for d in range(3):volumes[d]+=np.bincount(ids[d],weights=weights,minlength=len(volumes[d]))
        for key,mask in masks.items():volume[key]+=float(weights[mask].sum())
        for k in totals:
            diff=values[0][k]-values[1][k];square=diff*diff
            totals[k]+=np.tensordot(weights,square,axes=(0,0))
            norm=square.reshape(len(p),-1).sum(axis=1)
            for d in range(3):bins[k][d]+=np.bincount(ids[d],weights=weights*norm,minlength=len(volumes[d]))
            for key,mask in masks.items():region[key][k]+=float(weights[mask]@norm[mask])
    result=dict(signature=signature,points=len(X),rms={k:float(np.sqrt(v.sum()/w.sum())) for k,v in totals.items()},
        component_share={k:(v/max(v.sum(),1e-300)).tolist() for k,v in totals.items()},
        derivative_column_share=(totals['F'].sum(axis=0)/max(totals['F'].sum(),1e-300)).tolist(),
        regions={key:dict(volume_fraction=volume[key]/w.sum(),squared_share={k:v/max(totals[k].sum(),1e-300) for k,v in row.items()}) for key,row in region.items()},
        profiles={k:[dict(coordinate=((e[:-1]+e[1:])/2).tolist(),rms=np.sqrt(v/volumes[d]).tolist(),
                    squared_share=(v/max(totals[k].sum(),1e-300)).tolist()) for d,(e,v) in enumerate(zip(edges,bins[k]))] for k in totals})
    save(path,result);return result


def locate(out):
    dt=1.220703125e-7
    a=load_fields(PREVIOUS/'original/g145-dt0.0000001220703.npz')
    b=fields(catalog(out,'base')[dt])
    r=location(out,a,b,'145-161')
    print('INITIAL LOCATION',r['derivative_column_share'],r['regions'],flush=True)
    plot_profiles(out,'previous-location',{'145 to 161':r['profiles']})
    return r


def initial_interfaces(out):
    """Read-only directional jump diagnostic on the unchanged old history."""
    from itertools import product
    f=initial()['position'];axes=knots(geometry(17));z,w=np.polynomial.legendre.leggauss(7)
    result=dict(initial=fingerprint(initial()),normal_directions={})
    for normal in range(3):
        tangential=[d for d in range(3) if d!=normal];points=[];weights=[]
        for d in tangential:
            a=axes[d];h=np.diff(a)
            points.append(((a[:-1,None]+a[1:,None])/2+h[:,None]*z/2).ravel())
            weights.append((h[:,None]*w/2).ravel())
        xy=np.array(list(product(*points)));area=np.outer(*weights).ravel()
        data={}
        for eps in (1e-7,1e-8):
            rows=[]
            for plane in axes[normal][1:-1]:
                X=np.empty((len(xy),3));X[:,normal]=plane;X[:,tangential]=xy
                minus=X.copy();plus=X.copy();minus[:,normal]-=eps;plus[:,normal]+=eps
                xm,Fm=f.evaluate(minus);xp,Fp=f.evaluate(plus)
                Pm=material_response(Fm,directions(minus),geometry(17).params)[1]
                Pp=material_response(Fp,directions(plus),geometry(17).params)[1]
                rows.append(dict(plane=float(plane),x=rms(xp-xm,area),F=rms(Fp-Fm,area),
                                 traction=rms((Pp-Pm)[:,:,normal],area)))
            data[str(eps)]=dict(planes=rows,rms={k:float(np.sqrt(np.mean([row[k]**2 for row in rows]))) for k in ('x','F','traction')})
        result['normal_directions']['XYZ'[normal]]=data
    save(out/'initial-interfaces.json',result)
    print('INITIAL INTERFACES',{d:v['1e-08']['rms'] for d,v in result['normal_directions'].items()},flush=True)
    return result


def initial_representation(out,cases=None,sources=None):
    history=initial();result=dict(initial=fingerprint(history),cases={})
    for case,counts in (CASES if cases is None else cases).items():
        source=directional_geometry(counts) if sources is None else sources[case]
        basis=ReferenceBasis(source)
        X,w=tensor_rule(knots(source),2);row={}
        for name,f in history.items():
            reconstructed=HistoryField(((basis,frozen(f.evaluate(source.X)[0])),))
            maxima=np.zeros(2)
            for start in range(0,len(X),32768):
                a=f.evaluate(X[start:start+32768]);b=reconstructed.evaluate(X[start:start+32768])
                maxima=np.maximum(maxima,[np.max(abs(x-y)) for x,y in zip(a,b)])
            row[name]=dict(value_max=float(maxima[0]),gradient_max=float(maxima[1]))
        row['passed']=all(v['value_max']<1e-12 and v['gradient_max']<1e-10 for v in row.values())
        result['cases'][case]=row;print('INITIAL REPRESENTATION',case,row,flush=True)
    result['passed']=all(v['passed'] for v in result['cases'].values())
    save(out/'initial-representation.json',result)
    if not result['passed']:raise RuntimeError('initial state is not exactly representable on these nested-history grids')
    return result


def time_check(out,case):
    items=catalog(out,case);candidates=candidate_fields(PREVIOUS);rows=[]
    for dt in sorted(items,reverse=True):
        if 2*dt not in items:continue
        comparison=scaled_differences(out,fields(items[2*dt]),fields(items[dt]),candidates)
        passed=all(m['ratios'][k]<TIME_TARGET for m in comparison.values() for k in ('F','P','v'))
        rows.append(dict(dt_fine=dt,comparison=comparison,passed=passed))
        print('TIME',case,dt,{g:m['ratios'] for g,m in comparison.items()},passed,flush=True)
    result=dict(case=case,target=TIME_TARGET,pairs=rows)
    save(out/f'time-{case}.json',result);return result


def joint_check(out,dt):
    a,b=[fields(catalog(out,case)[dt]) for case in ('base','both')]
    result=dict(dt=dt,comparison=scaled_differences(out,a,b,candidate_fields(PREVIOUS)))
    save(out/f'joint-dt{dt:.13f}.json',result)
    print('JOINT',dt,{g:m['ratios'] for g,m in result['comparison'].items()},flush=True)
    return result


def evaluated(f,X):
    x,F=f['position'].evaluate(X)
    return dict(x=x,F=F,v=f['velocity'].evaluate(X)[0],P=material_response(F,directions(X),geometry(17).params)[1])


def factorial_differences(values):
    pairs=dict(axial_coarse=('axial','base'),cross_coarse=('cross','base'),
               axial_fine=('both','cross'),cross_fine=('both','axial'),joint=('both','base'))
    result={name:{k:values[a][k]-values[b][k] for k in ('x','F','P','v')} for name,(a,b) in pairs.items()}
    result['interaction']={k:values['both'][k]-values['axial'][k]-values['cross'][k]+values['base'][k]
                           for k in ('x','F','P','v')}
    return result


def factorial(out,states,dt,order=3):
    candidates=candidate_fields(PREVIOUS)
    edges=[np.linspace(.25,.75,33),np.linspace(.4375,.5625,9),np.linspace(.4375,.5625,9)]
    axes=union_knots(field_axes(*states.values(),*[f for pair in candidates.values() for f in pair]),edges)
    params=geometry(17).params
    signature=dict(protocol='directional-factorial-v1',fields={name:fingerprint(f) for name,f in states.items()},
        candidates={g:[fingerprint(f) for f in pair] for g,pair in candidates.items()},
        dt=dt,order=order,material=[params.mu,params.lam,params.k_f],direction='smooth-Y-pi-over-two')
    path=out/'factorial.json'
    if path.exists():
        result=json.loads(path.read_text())
        if result['signature']==signature:return result
    X,w=tensor_rule(axes,order);keys=('x','F','P','v');names=('axial_coarse','cross_coarse','axial_fine','cross_fine','joint','interaction')
    totals={name:{k:0. for k in keys} for name in names};components={name:np.zeros((3,3)) for name in names}
    region={name:{zone:{k:0. for k in keys} for zone in ('clamp','switch','bulk')} for name in names}
    profile={name:{k:[np.zeros(len(e)-1) for e in edges] for k in keys} for name in names}
    volumes=[np.zeros(len(e)-1) for e in edges];zone_volume={zone:0. for zone in ('clamp','switch','bulk')}
    gaps={g:{k:0. for k in keys} for g in candidates};scale={k:0. for k in keys};dots={k:0. for k in keys}
    for start in range(0,len(X),32768):
        guard();p=X[start:start+32768];weights=w[start:start+32768]
        values={name:evaluated(f,p) for name,f in states.items()};diffs=factorial_differences(values)
        masks=dict(clamp=p[:,0]<.3125,switch=(p[:,0]>=.4375)&(p[:,0]<=.5625));masks['bulk']=~(masks['clamp']|masks['switch'])
        ids=[np.clip(np.searchsorted(e,p[:,d],side='right')-1,0,len(e)-2) for d,e in enumerate(edges)]
        for d in range(3):volumes[d]+=np.bincount(ids[d],weights=weights,minlength=len(volumes[d]))
        for zone,mask in masks.items():zone_volume[zone]+=float(weights[mask].sum())
        for name,diffs_field in diffs.items():
            components[name]+=np.tensordot(weights,diffs_field['F']**2,axes=(0,0))
            for k,diff in diffs_field.items():
                sq=(diff.reshape(len(p),-1)**2).sum(axis=1);totals[name][k]+=float(weights@sq)
                for zone,mask in masks.items():region[name][zone][k]+=float(weights[mask]@sq[mask])
                for d in range(3):profile[name][k][d]+=np.bincount(ids[d],weights=weights*sq,minlength=len(volumes[d]))
        for k in keys:
            v=values['both'][k]-(p if k=='x' else np.eye(3) if k=='F' else 0)
            scale[k]+=float(weights@(v.reshape(len(p),-1)**2).sum(axis=1))
            dots[k]+=float(weights@(diffs['axial_coarse'][k]*diffs['cross_coarse'][k]).reshape(len(p),-1).sum(axis=1))
        for g,pair in candidates.items():
            a,b=[evaluated(f,p) for f in pair]
            for k in keys:gaps[g][k]+=float(weights@((a[k]-b[k]).reshape(len(p),-1)**2).sum(axis=1))
        if start//32768%32==0:print('FACTORIAL',start+len(p),'/',len(X),flush=True)
    if any(v<=0 for row in gaps.values() for v in row.values()):raise ValueError('nonzero scheme gaps required for comparison ratios')
    result=dict(signature=signature,points=len(X),scheme_gap={g:{k:np.sqrt(v/w.sum()) for k,v in row.items()} for g,row in gaps.items()},effects={})
    for name,row in totals.items():
        result['effects'][name]=dict(rms={k:np.sqrt(v/w.sum()) for k,v in row.items()},
            relative={k:np.sqrt(v/max(scale[k],1e-300)) for k,v in row.items()},
            ratios={g:{k:np.sqrt(v/gaps[g][k]) for k,v in row.items()} for g in gaps},
            gradient_column_share=(components[name].sum(axis=0)/max(components[name].sum(),1e-300)).tolist(),
            regions={z:dict(volume_fraction=zone_volume[z]/w.sum(),squared_share={k:v/max(row[k],1e-300) for k,v in rr.items()}) for z,rr in region[name].items()},
            profiles={k:[dict(coordinate=((e[:-1]+e[1:])/2).tolist(),rms=np.sqrt(v/volumes[d]).tolist()) for d,(e,v) in enumerate(zip(edges,bins))] for k,bins in profile[name].items()})
    result['interaction_to_joint']={k:np.sqrt(totals['interaction'][k]/max(totals['joint'][k],1e-300)) for k in keys}
    result['axial_cross_correlation']={k:(dots[k]/np.sqrt(totals['axial_coarse'][k]*totals['cross_coarse'][k])
        if totals['axial_coarse'][k]*totals['cross_coarse'][k]>0 else None) for k in keys}
    save(path,result);return result


def effect_time(out,fine,coarse,dt):
    signature=dict(fine={k:fingerprint(f) for k,f in fine.items()},coarse={k:fingerprint(f) for k,f in coarse.items()},dt=dt)
    path=out/'effect-time.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['signature']==signature:return old
    X,w=tensor_rule(field_axes(*fine.values(),*coarse.values()),3)
    names=('axial_coarse','cross_coarse','axial_fine','cross_fine','joint','interaction');keys=('x','F','P','v')
    changes={n:{k:0. for k in keys} for n in names};sizes={n:{k:0. for k in keys} for n in names}
    coarse_sizes={n:{k:0. for k in keys} for n in names}
    for start in range(0,len(X),32768):
        guard();p=X[start:start+32768];weights=w[start:start+32768]
        fine_values=factorial_differences({k:evaluated(f,p) for k,f in fine.items()})
        coarse_values=factorial_differences({k:evaluated(f,p) for k,f in coarse.items()})
        for n in names:
            for k in keys:
                a,b=fine_values[n][k],coarse_values[n][k]
                changes[n][k]+=float(weights@((a-b).reshape(len(p),-1)**2).sum(axis=1))
                sizes[n][k]+=float(weights@(a.reshape(len(p),-1)**2).sum(axis=1))
                coarse_sizes[n][k]+=float(weights@(b.reshape(len(p),-1)**2).sum(axis=1))
        if start//32768%64==0:print('EFFECT TIME',start+len(p),'/',len(X),flush=True)
    result=dict(signature=signature,effects={n:dict(relative={k:np.sqrt(changes[n][k]/max(sizes[n][k],1e-300)) for k in keys},
        fine_rms={k:np.sqrt(sizes[n][k]/w.sum()) for k in keys},coarse_rms={k:np.sqrt(coarse_sizes[n][k]/w.sum()) for k in keys}) for n in names})
    result['passed']={n:all(row['relative'][k]<.05 for k in ('F','P','v')) for n,row in result['effects'].items()}
    save(path,result);return result


def analyze(out):
    times={case:time_check(out,case) for case in CASES};items={case:catalog(out,case) for case in CASES}
    common=set.intersection(*(set(v) for v in items.values()))
    if not common:raise RuntimeError('no common time step across the four spaces')
    dt=min(common);states={case:fields(v[dt]) for case,v in items.items()}
    passed={case:any(p['dt_fine']==dt and p['passed'] for p in t['pairs']) for case,t in times.items()}
    result=dict(dt=dt,time=times,time_passed_at_common_dt=passed,storage_start=guard())
    save(out/'results-progress.json',result)
    # Independent, read-only field evaluations; each keeps its own reduction
    # order and writes a separate cache artifact.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        spatial=pool.submit(factorial,out,states,dt)
        temporal=(pool.submit(effect_time,out,states,{case:fields(v[2*dt]) for case,v in items.items()},dt)
                  if all(2*dt in v for v in items.values()) else None)
        result['factorial']=spatial.result();save(out/'results-progress.json',result)
        if temporal is not None:result['effect_time']=temporal.result()
    result['space_passed']=all(v<SPACE_TARGET for row in result['factorial']['effects']['joint']['ratios'].values() for k,v in row.items() if k in ('F','P','v'))
    result['reference_ready']=all(passed.values()) and result['space_passed']
    records=[m for case,v in items.items() for p,m in v.values() if p.is_relative_to(out)]
    result['runs']=len(records);result['steps']=sum(len(m['rows']) for m in records)
    result['numerical_checks_passed']=all(all(m['checks'].values()) for m in records)
    result['storage_end']=guard();save(out/'results.json',result);return result


def endpoint(out,case,final,counts=None,source=None):
    path=out/f'endpoint-{case}.json';signature=dict(case=case,final=fingerprint(final))
    if counts is not None:signature['counts']=list(counts)
    if source is not None:signature['axes']=[a.tolist() for a in knots(source)]
    if path.exists():
        cached=json.loads(path.read_text())
        if cached['signature']==signature:return cached['metrics']
    source=directional_geometry(CASES[case] if counts is None else counts) if source is None else source
    basis=ReferenceBasis(source)
    axes=union_knots(knots(geometry(17)),knots(source))
    perturbation=np.random.default_rng(481).normal(size=source.X.shape)*.01;perturbation[source.fixed]=0
    values=[]
    for order in (5,7):
        X,w=tensor_rule(axes,order);energy=0.;force=np.zeros_like(source.X);action=force.copy()
        for start in range(0,len(X),32768):
            guard();p=X[start:start+32768];weights=w[start:start+32768];A=directions(p)
            mapping=basis.sample(p,weights);F=final['position'].evaluate(p)[1]
            psi,P=material_response(F,A,source.params)
            dF=np.stack([D@perturbation for D in mapping.D],axis=2)
            dP=material_tangent(F,A,dF,source.params);energy+=float(weights@psi)
            for d,D in enumerate(mapping.D):
                force+=D.T@(weights[:,None]*P[:,:,d]);action+=D.T@(weights[:,None]*dP[:,:,d])
        values.append((energy,force[source.free],action[source.free]))
        print('ENDPOINT',case,order,'complete',flush=True)
    metrics={k:float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)) for k,a,b in zip(('energy','force','tangent_action'),*values)}
    metrics['passed']=max(metrics.values())<1e-6
    save(path,dict(signature=signature,metrics=metrics));return metrics


def difference_norms(out,a,b,order):
    import hashlib
    signature=dict(a=fingerprint(a),b=fingerprint(b),order=order)
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    path=out/f'norms-{key}.json'
    if path.exists():return json.loads(path.read_text())['rms']
    X,w=tensor_rule(field_axes(a,b),order);totals={k:0. for k in ('x','F','P','v')}
    for start in range(0,len(X),32768):
        guard();p=X[start:start+32768];weights=w[start:start+32768]
        av,bv=evaluated(a,p),evaluated(b,p)
        for k in totals:totals[k]+=float(weights@((av[k]-bv[k]).reshape(len(p),-1)**2).sum(axis=1))
        if start//32768%128==0:print('NORMS',order,start+len(p),'/',len(X),flush=True)
    result={k:float(np.sqrt(v/w.sum())) for k,v in totals.items()}
    save(path,dict(signature=signature,rms=result));return result


def audit(out):
    result=json.loads((out/'results.json').read_text());dt=result['dt']
    items={case:catalog(out,case) for case in CASES};states={case:fields(v[dt]) for case,v in items.items()}
    initial_report=json.loads((out/'initial-representation.json').read_text())
    if initial_report['initial']!=fingerprint(initial()):raise ValueError('initial representation audit fingerprint mismatch')
    report=dict(endpoints={},initial_representation=initial_report)
    for case in CASES:
        report['endpoints'][case]=endpoint(out,case,states[case]);save(out/'audit-progress.json',report)
    # These read-only checks have disjoint cache keys and keep the same
    # per-norm accumulation order when evaluated concurrently.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        space_norm=pool.submit(difference_norms,out,states['both'],states['base'],5)
        time_norms={case:pool.submit(difference_norms,out,fields(items[case][2*dt]),states[case],5)
                    for case in CASES}
        independent_space=space_norm.result()
        independent_time={case:job.result() for case,job in time_norms.items()}
    independent=independent_space
    original=result['factorial']['effects']['joint']['rms']
    report['space_probes']={k:abs(original[k]/independent[k]-1) for k in original}
    # Every new grid gets its own independent temporal norm check.
    report['time_probes_by_case']={}
    for case,independent in independent_time.items():
        time=next(p for p in result['time'][case]['pairs'] if p['dt_fine']==dt)
        original=time['comparison']['17']['difference']['all']
        report['time_probes_by_case'][case]={k:abs(original[k+'_rms']/independent[k]-1) for k in independent}
    report['time_probes']=report['time_probes_by_case']['both']
    report['scheme_gap_probes']={}
    for g,pair in candidate_fields(PREVIOUS).items():
        independent=cached_compare(out,*pair,field_axes(*pair),7)['all']
        report['scheme_gap_probes'][g]={k:abs(result['factorial']['scheme_gap'][g][k]/independent[k+'_rms']-1) for k in ('x','F','P','v')}
    report['passed']=(initial_report['passed'] and all(v['passed'] for v in report['endpoints'].values())
        and max(report['space_probes'].values())<1e-6
        and all(max(v.values())<1e-6 for v in report['time_probes_by_case'].values())
        and all(max(v.values())<1e-6 for v in report['scheme_gap_probes'].values()))
    report['storage_end']=guard();save(out/'audit.json',report)
    if not report['passed']:raise RuntimeError('directional quadrature audit failed')
    return report


def plot_profiles(out,name,reports):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7),sharey='row',layout='constrained')
    for row,k in enumerate(('F','P')):
        for d in range(3):
            ax=axes[row,d]
            for label,profiles in reports.items():
                p=profiles[k][d];ax.plot(p['coordinate'],p['rms'],label=label)
            ax.set_xlabel('material '+'XYZ'[d]);ax.set_ylabel(k+' change RMS');ax.legend()
            if d==0:
                ax.axvspan(.25,.3125,color='gray',alpha=.1);ax.axvspan(.4375,.5625,color='orange',alpha=.1)
    for row in axes:row[0].set_ylim(bottom=0)
    fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((out/'results.json').read_text());f=r['factorial'];dt=r['dt']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    vals=[]
    for case in CASES:
        t=next(p for p in r['time'][case]['pairs'] if p['dt_fine']==dt)
        vals.append(max(m['ratios'][k] for m in t['comparison'].values() for k in ('F','P','v')))
    axes[0,0].bar(list(CASES),vals);axes[0,0].axhline(TIME_TARGET,color='r',linestyle='--')
    axes[0,0].set_ylabel('max T_F, T_P, T_v');axes[0,0].set_title(f'Common fine dt = {dt*1e6:.5f} microseconds')
    for j,g in enumerate(('17','33')):
        axes[0,1].bar(np.arange(3)+(j-.5)*.35,[f['effects']['joint']['ratios'][g][k] for k in ('F','P','v')],width=.35,label=f'candidate {g}')
    axes[0,1].set_xticks(range(3),['F','P','v']);axes[0,1].axhline(SPACE_TARGET,color='r',linestyle='--')
    axes[0,1].set_yscale('log');axes[0,1].set_ylabel('Joint refinement R');axes[0,1].legend()
    names=['axial_coarse','cross_coarse','axial_fine','cross_fine']
    for j,k in enumerate(('F','P','v')):
        axes[1,0].bar(np.arange(4)+(j-1)*.25,[f['effects'][n]['rms'][k]/f['effects']['joint']['rms'][k] for n in names],width=.25,label=k)
    axes[1,0].set_xticks(range(4),['x at coarse yz','yz at coarse x','x at fine yz','yz at fine x'],rotation=20)
    axes[1,0].set_ylabel('Directional change / joint change');axes[1,0].legend()
    axes[1,1].bar(['F','P','v'],[f['interaction_to_joint'][k] for k in ('F','P','v')])
    axes[1,1].set_ylabel('Interaction norm / joint change norm')
    fig.savefig(out/'convergence.png',dpi=180);fig.savefig(out/'convergence.pdf');plt.close(fig)
    labels=dict(axial_coarse='axial refinement',cross_coarse='cross-section refinement',joint='joint refinement')
    for name,reports in [('profiles',{labels[n]:f['effects'][n]['profiles'] for n in labels}),
                         ('previous-location',{'145 to 161':json.loads((out/'location-145-161.json').read_text())['profiles']})]:
        plot_profiles(out,name,reports)


def progress(out):
    import re,time
    from datetime import datetime,timezone
    while True:
        status=guard()
        text=['# 方向加密实验实时进度\n\n',f'更新时间（UTC）：{datetime.now(timezone.utc).isoformat()}\n\n',
              f'系统盘剩余 {status["system_free_bytes"]/2**30:.2f} GiB，数据盘剩余 {status["data_free_bytes"]/2**30:.2f} GiB。\n\n',
              '本页每五分钟自动更新。步数完成不等于时间或空间精度验收通过。\n\n',
              '| 组别 | dt (μs) | 已完成步数 | 总步数 | 终态文件 |\n|---|---:|---:|---:|---|\n']
        for path in sorted(out.glob('*.log')):
            match=re.match(r'^(base|axial|cross|both)-([0-9.]+)\.log$',path.name)
            if not match:continue
            rows=re.findall(r': (\d+)/(\d+), residual=',path.read_text())
            case,dt=match.group(1),float(match.group(2))
            step,total=rows[-1] if rows else ('0',str(round(.003/dt)))
            meta=out/'original'/f'g{label(case)}-dt{dt:.13f}.json'
            text.append(f'| {case} | {dt*1e6:.7f} | {step} | {total} | {"已保存" if meta.exists() else "运行中"} |\n')
        if (out/'results.json').exists():
            r=json.loads((out/'results.json').read_text())
            text.append(f'\n阶段结果的共同时间步：{r["dt"]*1e6:.9f} μs；时间检查：{r["time_passed_at_common_dt"]}；空间门槛：{r["space_passed"]}。\n')
        if (out/'audit.json').exists():
            r=json.loads((out/'audit.json').read_text());text.append(f'\n独立积分复核通过：{r["passed"]}。\n')
        (out/'PROGRESS.md').write_text(''.join(text))
        if (out/'audit.json').exists() or (out/'stop-progress').exists():break
        for _ in range(10):
            time.sleep(30)
            if (out/'audit.json').exists() or (out/'stop-progress').exists():break


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('locate','interfaces','initial','run','time','joint','analyze','audit','plot','progress'),required=True)
    p.add_argument('--case',choices=tuple(CASES),default='base')
    p.add_argument('--dt',type=float,default=1.220703125e-7)
    p.add_argument('--device',default='cuda:0');p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();guard();a.out.mkdir(parents=True,exist_ok=True)
    if a.stage=='locate':locate(a.out)
    elif a.stage=='interfaces':initial_interfaces(a.out)
    elif a.stage=='initial':initial_representation(a.out)
    elif a.stage=='time':time_check(a.out,a.case)
    elif a.stage=='joint':joint_check(a.out,a.dt)
    elif a.stage=='analyze':analyze(a.out)
    elif a.stage=='audit':audit(a.out)
    elif a.stage=='plot':plot(a.out)
    elif a.stage=='progress':progress(a.out)
    else:run(a.out,a.case,a.dt,a.device)


if __name__=='__main__':main()
