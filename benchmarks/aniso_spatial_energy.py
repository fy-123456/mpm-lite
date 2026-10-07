"""Finer spatial references and separate energy/velocity/traction controls."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import uuid
import numpy as np

from engine.aniso_phase1.convergence_reference import (
    geometry,knots,union_knots,tensor_rule,sites,load_fields,save_fields,directions)
from engine.aniso_phase1.initial_controls import (
    match_elastic,smooth_velocity,initial_dead_traction,release_factor,energy,velocity_moments)
from .aniso_convergence_reference import save,fingerprint
from .aniso_refinement import OLD,compare,field_axes
from .aniso_history_increment import guard

OUT=Path('docs/results/spatial-energy/v1')
CONTROL=Path('docs/results/reference-control/v1')
DT=.0000009765625


def cached_compare(out,a,b,axes,order=3):
    params=geometry(17).params
    signature=dict(protocol='spatial-energy-metrics-v1',a=fingerprint(a),b=fingerprint(b),
                   axes=[np.asarray(axis).tolist() for axis in axes],order=order,
                   material=[params.mu,params.lam,params.k_f],direction='smooth-Y-pi-over-two')
    key=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()
    folder=out/'metrics';folder.mkdir(exist_ok=True)
    path=folder/(key+'.json')
    if path.exists():return json.loads(path.read_text())['metrics']
    result=compare(a,b,tensor_rule(axes,order))
    temporary=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(dict(signature=signature,metrics=result),indent=2,allow_nan=False)+'\n')
    temporary.replace(path)
    return result


def initial_for(out,case):
    if case=='original': return load_fields(OLD/'initial-switch-state.npz')
    if case=='smooth': return load_fields(CONTROL/'smooth-initial.npz')
    return load_fields(out/('velocity-initial.npz' if case=='velocity' else 'matched-initial.npz'))


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    original=initial_for(out,'original');smooth=initial_for(out,'smooth')
    X,w=tensor_rule(knots(geometry(17)),7)
    position,elastic=match_elastic(original['position'],smooth['position'],X,w,geometry(17).params)
    velocity,kinetic=smooth_velocity(original['velocity'],X,w)
    matched=dict(position=position,velocity=original['velocity'])
    changed=dict(position=position,velocity=velocity)
    save_fields(out/'matched-initial.npz',**matched)
    save_fields(out/'velocity-initial.npz',**changed)
    X9,w9=tensor_rule(knots(geometry(17)),9)
    U0=energy(original['position'],X9,w9,geometry(17).params)
    Um=energy(position,X9,w9,geometry(17).params)
    initial=dict(original=original,smooth=smooth,matched=matched,velocity=changed)
    result=dict(elastic_match=elastic,kinetic_match=kinetic,
        independent_order9_energy_relative=abs(Um/U0-1),initial={},storage=guard())
    for name,fields in initial.items():
        result['initial'][name]=dict(fingerprint=fingerprint(fields),
            against_original=compare(fields,original,(X9,w9)),
            elastic=energy(fields['position'],X9,w9,geometry(17).params),
            kinetic=.5*float(np.sum(w9[:,None]*fields['velocity'].evaluate(X9)[0]**2)))
    if result['independent_order9_energy_relative']>1e-9:
        raise RuntimeError('independent energy matching check failed')
    save(out/'initial-controls.json',result)
    return result


def run(out,case,grid,dt,device='cuda:0',duration=.003,initial=None,initial_velocity_order=None,source=None,transfer_mode='nodal',progress_callback=None,replay=False):
    from engine.aniso_phase1.resident_reference import ResidentReference
    initial=initial_for(out,case) if initial is None else initial
    folder=out/case;folder.mkdir(parents=True,exist_ok=True)
    name=f'g{grid}-dt{dt:.13f}'
    config=dict(protocol='spatial-energy-v1',backend='resident-float64-deterministic-assembly',
                initial=fingerprint(initial),grid=grid,case=case,dt=dt,duration=duration,
                order=5,tolerance=1e-9,device=device,release_duration=.00025 if case=='boundary' else 0.)
    if transfer_mode != 'nodal':config['transfer_mode']=transfer_mode
    velocity_order=(7 if case=='velocity' else None) if initial_velocity_order is None else initial_velocity_order
    if velocity_order is not None:config['initial_velocity_order']=velocity_order
    if source is not None:config['reference_axes']=[a.tolist() for a in knots(source)]
    meta_path=folder/(name+'.json');field_path=folder/(name+'.npz')
    if meta_path.exists() and field_path.exists():
        meta=json.loads(meta_path.read_text())
        if (meta['config']==config and all(meta['checks'].values())
                and (not replay or (folder/(name+'-replay.npz')).exists())):
            print(name+': cached',flush=True);return meta,load_fields(field_path)
    guard();start=time.perf_counter();source=geometry(grid) if source is None else source
    axes=union_knots(knots(geometry(17)),knots(source))
    p=sites(*tensor_rule(axes,2));q=sites(*tensor_rule(axes,5))
    moments=velocity_moments(source,initial['velocity'],velocity_order) if velocity_order is not None else None
    model=ResidentReference(source,initial,p,q,device,velocity_moments=moments,transfer_mode=transfer_mode)
    traction=initial_dead_traction(source,initial['position']) if case=='boundary' else np.zeros_like(source.X)
    nsteps=round(duration/dt)
    if abs(nsteps*dt-duration)>1e-14:raise ValueError('dt must divide release duration')
    replay_X=tensor_rule(knots(geometry(17)),2)[0] if replay else None
    frames=[initial['position'].evaluate(replay_X)[0]] if replay else []
    frame_times=[0.];frame_energy=[model.energy+model.kinetic]
    rows=[];previous=model.energy+model.kinetic
    initial_mechanical=previous
    for step in range(nsteps):
        guard()
        external=release_factor((step+1)*dt)*traction
        info=model.step(dt,external=external)
        info['dissipation_checked']=info['mechanical']<=previous+info['external_work']+1e-12
        previous=info['mechanical'];rows.append(dict(step=step+1,**info))
        if replay and ((step+1)%max(1,nsteps//8)==0 or step+1==nsteps):
            frames.append(model.fields()['position'].evaluate(replay_X)[0])
            frame_times.append((step+1)*dt);frame_energy.append(previous)
        if progress_callback and ((step+1)%16==0 or step+1==nsteps):
            progress_callback(step+1,nsteps,info)
        if (step+1)%128==0 or step+1==nsteps:
            print(f'{case} g{grid} dt={dt:g}: {step+1}/{nsteps}, residual={info["scaled_residual_inf"]:.3g}',flush=True)
    fields=model.fields()
    model.device_coefficients.assign(model.coefficients)
    model.q.gradient(model.device_coefficients);model.p.gradient(model.device_coefficients)
    errors=dict(material_F=float(np.max(abs(fields['position'].evaluate(q.X)[1]-model.q.F.numpy()))),
                particle_F=float(np.max(abs(fields['position'].evaluate(p.X)[1]-model.p.F.numpy()))))
    vp=fields['velocity'].evaluate(p.X)[0]
    kinetic=.5*float(np.sum(p.weight[:,None]*vp*vp))
    errors['kinetic_readback']=abs(kinetic-model.kinetic)
    checks=dict(solve=max(r['scaled_residual_inf'] for r in rows)<=1.01e-9,
        admissible=min(r['min_det'] for r in rows)>0,clamp=max(r['clamp_speed'] for r in rows)<1e-12,
        history=max(errors['material_F'],errors['particle_F'])<1e-10,
        mass_readback=errors['kinetic_readback']<1e-15,
        budget=max(abs(r['energy_budget_residual']) for r in rows)<1e-12,
        dissipation=all(r['dissipation_checked'] for r in rows),
        transfer=max(r['transfer_roundtrip_relative'] for r in rows)<1e-10)
    result=dict(config=config,checks=checks,rows=rows,history_error=errors,
        initial_mechanical=initial_mechanical,material_points=len(q.X),mass_points=len(p.X),
        elapsed_seconds=time.perf_counter()-start,storage=guard(),rank=model.base.rank_info,
        external_work=sum(r['external_work'] for r in rows))
    if not all(checks.values()):raise RuntimeError(f'acceptance failed: {checks}')
    save_fields(field_path,**fields);save(meta_path,result)
    if replay:
        np.savez_compressed(folder/(name+'-replay.npz'),reference=replay_X,positions=np.asarray(frames),
                            times=frame_times,mechanical=frame_energy)
    model.close()
    return result,fields


def backend_check(out):
    folder=out/'original'
    path=folder/f'g97-dt{DT:.13f}.npz'
    a=load_fields(path)
    b=load_fields(CONTROL/'original/g97-reference-dt0.000000976563.npz')
    report=compare(a,b,tensor_rule(knots(geometry(97)),3))
    report['passed']=max(report['relative'].values())<1e-7
    save(out/'backend-check.json',report)
    if not report['passed']:raise RuntimeError('full trajectory backend check failed')
    return report


def find_runs(out,case):
    result={}
    signature=fingerprint(initial_for(out,case))
    for p in (out/case).glob('g*-dt*.json'):
        m=json.loads(p.read_text())
        if (all(m['checks'].values()) and m['config']['duration']==.003
                and m['config']['initial']==signature and p.with_suffix('.npz').exists()):
            result[(m['config']['grid'],m['config']['dt'])]=(p,m)
    return result


def initial_audit(out):
    fields=initial_for(out,'velocity');result={}
    for grid in (81,97):
        source=geometry(grid)
        values=[velocity_moments(source,fields['velocity'],order) for order in (2,7,9)]
        load5=initial_dead_traction(source,fields['position'],5)
        load7=initial_dead_traction(source,fields['position'],7)
        result[str(grid)]=dict(kinetic_relative=abs(values[1][0]/values[2][0]-1),
            velocity_rhs_relative=float(np.linalg.norm(values[1][1]-values[2][1])/np.linalg.norm(values[2][1])),
            old_order2_kinetic_relative=abs(values[0][0]/values[2][0]-1),
            old_order2_velocity_rhs_relative=float(np.linalg.norm(values[0][1]-values[2][1])/np.linalg.norm(values[2][1])),
            traction_relative=float(np.linalg.norm(load5-load7)/np.linalg.norm(load7)))
        print('INITIAL AUDIT',grid,result[str(grid)],flush=True)
    result['passed']=all(max(row[k] for k in ('kinetic_relative','velocity_rhs_relative','traction_relative'))<1e-9 for row in result.values())
    save(out/'initial-quadrature.json',result)
    if not result['passed']:raise RuntimeError('initial quadrature audit requires attention')
    return result


def audit(out):
    from .aniso_refinement import endpoint_audit
    result=dict(initial=initial_audit(out),endpoints={})
    for case in ('original','matched','velocity','boundary'):
        items=find_runs(out,case)
        grid=max(g for g,dt in items);dt=min(dt for g,dt in items if g==grid)
        final=load_fields(items[(grid,dt)][0].with_suffix('.npz'))
        result['endpoints'][case]=endpoint_audit(out,grid,case,final,initial_for(out,case))
        print('ENDPOINT AUDIT',case,result['endpoints'][case],flush=True)
        save(out/'quadrature-checks.json',result)
    items=find_runs(out,'original');ga,gb=sorted({g for g,dt in items})[-2:]
    common=set(dt for g,dt in items if g==ga)&set(dt for g,dt in items if g==gb)
    dt=min(common);fields=[load_fields(items[(g,dt)][0].with_suffix('.npz')) for g in (ga,gb)]
    comparisons=[cached_compare(out,*fields,field_axes(*fields),order) for order in (3,5)]
    result['probe_relative']={k:abs(comparisons[0]['all'][k+'_rms']/comparisons[1]['all'][k+'_rms']-1) for k in ('x','F','P','v')}
    result['passed']=(all(row['passed_1e-6'] for row in result['endpoints'].values())
                      and max(result['probe_relative'].values())<1e-6 and result['initial']['passed'])
    save(out/'quadrature-checks.json',result)
    if not result['passed']:raise RuntimeError('quadrature audit failed')
    return result


def analyze(out):
    result=dict(initial=json.loads((out/'initial-controls.json').read_text()),time={},space={},
                storage=guard(),comparisons={},reference_to_scheme_ratios={},ratio_components={})
    cases=('original','smooth','matched','velocity','boundary')
    runs={case:find_runs(out,case) for case in cases}
    # Previous cases provide verified same-protocol endpoints without rerunning.
    for case in ('original','smooth'):
        expected=fingerprint(initial_for(out,case))
        for grid in (81,97):
            for dt in (DT,2*DT):
                path=CONTROL/case/f'g{grid}-reference-dt{dt:.12f}.json'
                if path.exists():
                    meta=json.loads(path.read_text())
                    if meta['config']['initial']!=expected or not all(meta['checks'].values()):
                        raise ValueError('previous reference initial state or acceptance mismatch')
                    runs[case].setdefault((grid,dt),(path,meta))
    for case,items in runs.items():
        result['time'][case]={}
        for grid in sorted({g for g,dt in items}):
            pairs=[]
            for g,dt in sorted(items,reverse=True):
                if g!=grid or (g,2*dt) not in items:continue
                a=load_fields(items[(g,2*dt)][0].with_suffix('.npz'));b=load_fields(items[(g,dt)][0].with_suffix('.npz'))
                metrics=cached_compare(out,a,b,union_knots(knots(geometry(17)),knots(geometry(g))))
                pairs.append(dict(dt=dt,metrics=metrics,passed=all(metrics['relative'][k]<.01 for k in ('v','F','P'))))
                print('TIME',case,grid,dt,metrics['relative'],flush=True)
            result['time'][case][str(grid)]=pairs
        grids=sorted({g for g,dt in items})
        result['space'][case]={}
        for ga,gb in zip(grids,grids[1:]):
            common=set(dt for g,dt in items if g==ga)&set(dt for g,dt in items if g==gb)
            if not common:continue
            dt=min(common)
            a=load_fields(items[(ga,dt)][0].with_suffix('.npz'));b=load_fields(items[(gb,dt)][0].with_suffix('.npz'))
            metrics=cached_compare(out,a,b,union_knots(knots(geometry(ga)),knots(geometry(gb))))
            result['space'][case][f'{ga}-{gb}']=dict(dt=dt,metrics=metrics)
            print('SPACE',case,ga,gb,metrics['relative'],flush=True)
        save(out/'results.json',result)
    for other in ('smooth','matched','velocity','boundary'):
        baseline='smooth' if other=='matched' else ('matched' if other in ('velocity','boundary') else 'original')
        common=set(runs[baseline])&set(runs[other])
        if common:
            grid=max(g for g,dt in common);dt=min(dt for g,dt in common if g==grid)
            a=load_fields(runs[other][(grid,dt)][0].with_suffix('.npz'))
            b=load_fields(runs[baseline][(grid,dt)][0].with_suffix('.npz'))
            result['comparisons'][other]=dict(baseline=baseline,grid=grid,dt=dt,
                metrics=cached_compare(out,a,b,union_knots(knots(geometry(17)),knots(geometry(grid)))))
    common=set(runs['original'])&set(runs['matched'])
    if common:
        grid=max(g for g,dt in common);dt=min(dt for g,dt in common if g==grid)
        a=load_fields(runs['matched'][(grid,dt)][0].with_suffix('.npz'))
        b=load_fields(runs['original'][(grid,dt)][0].with_suffix('.npz'))
        result['comparisons']['matched-original']=dict(baseline='original',grid=grid,dt=dt,
            metrics=cached_compare(out,a,b,union_knots(knots(geometry(17)),knots(geometry(grid)))))
    items=runs['original'];grids=sorted({g for g,dt in items})
    if len(grids)>1:
        ga,gb=grids[-2:]
        common=set(dt for g,dt in items if g==ga)&set(dt for g,dt in items if g==gb)
        dt=min(common)
        refs=[load_fields(items[(g,dt)][0].with_suffix('.npz')) for g in (ga,gb)]
        prior=Path('docs/results/refinement/v1');candidate=json.loads((prior/'time-results.json').read_text())
        if candidate['initial']!=fingerprint(initial_for(out,'original')):
            raise ValueError('candidate and reference must start from the same physical state')
        result['reference_ratio_definition']=dict(reference_grids=[ga,gb],reference_dt=dt,
            threshold=.1,candidate_source=str(prior/'time-results.json'))
        for grid in (17,33):
            fields=[load_fields(prior/candidate['paths'][f'g{grid}-{c}']['final_file']) for c in ('shifted','enriched')]
            axes=field_axes(*refs,*fields)
            numerator=cached_compare(out,refs[0],refs[1],axes);denominator=cached_compare(out,fields[0],fields[1],axes)
            result['reference_to_scheme_ratios'][str(grid)]={k:numerator['all'][k+'_rms']/max(denominator['all'][k+'_rms'],1e-30) for k in ('x','F','P','v')}
            result['ratio_components'][str(grid)]=dict(reference_difference=numerator,
                scheme_difference=denominator,candidate_dt=candidate['paths'][f'g{grid}-enriched']['final_dt'])
        result['reference_ratio_passed']=all(v<.1 for row in result['reference_to_scheme_ratios'].values() for v in row.values())
    # Keep counts consistent with the completed files captured at analysis start.
    records=[m for items in runs.values() for p,m in items.values() if p.is_relative_to(out)]
    result['runs']=len(records);result['steps']=sum(len(m['rows']) for m in records)
    result['numerical_checks_passed']=all(all(m['checks'].values()) for m in records)
    rows=[r for m in records for r in m['rows']]
    result['run_audit']=dict(min_det=min(r['min_det'] for r in rows),
        max_residual=max(r['scaled_residual_inf'] for r in rows),
        max_budget_residual=max(abs(r['energy_budget_residual']) for r in rows),
        max_clamp=max(r['clamp_speed'] for r in rows),
        max_history_error=max(max(m['history_error'].values()) for m in records))
    result['storage_end']=guard();save(out/'results.json',result)
    return result


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=json.loads((out/'results.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    names=[];time_values=[]
    for case,grids in result['time'].items():
        for grid,pairs in grids.items():
            if not pairs:continue
            last=min(pairs,key=lambda p:p['dt'])
            names.append(f'{case}\n{grid}')
            time_values.append(max(last['metrics']['relative'][k] for k in ('v','F','P'))*100)
    axes[0,0].bar(range(len(names)),time_values)
    axes[0,0].set_xticks(range(len(names)),names,rotation=55,ha='right',fontsize=8)
    axes[0,0].axhline(1,color='r',linestyle='--',label='1% target')
    axes[0,0].set_ylabel('max adjacent v/F/P difference (%)');axes[0,0].legend()
    for k in ('F','P','v'):
        rows=result['space']['original'];labels=list(rows)
        axes[0,1].plot(labels,[rows[p]['metrics']['relative'][k]*100 for p in labels],'o-',label=k)
    axes[0,1].set_ylabel('original spatial reference change (%)');axes[0,1].legend()
    for grid,ratios in result['reference_to_scheme_ratios'].items():
        axes[1,0].plot(list(ratios),list(ratios.values()),'o-',label=f'candidate {grid}')
    axes[1,0].axhline(.1,color='r',linestyle='--',label='0.1 target')
    axes[1,0].set_yscale('log');axes[1,0].set_ylabel('reference change / scheme difference');axes[1,0].legend()
    cases=[c for c,rows in result['space'].items() if '81-97' in rows]
    for i,k in enumerate(('F','P','v')):
        axes[1,1].bar(np.arange(len(cases))+(i-1)*.25,
            [result['space'][c]['81-97']['metrics']['relative'][k]*100 for c in cases],width=.25,label=k)
    axes[1,1].set_xticks(range(len(cases)),cases,rotation=25)
    axes[1,1].set_ylabel('81 to 97 spatial change (%)');axes[1,1].legend()
    fig.savefig(out/'convergence.png',dpi=180);fig.savefig(out/'convergence.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('prepare','run','backend','analyze','initial-audit','audit','plot'),required=True)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--case',choices=('original','smooth','matched','velocity','boundary'),default='original')
    p.add_argument('--grid',type=int,default=97);p.add_argument('--dt',type=float,default=DT)
    p.add_argument('--device',default='cuda:0');p.add_argument('--duration',type=float,default=.003)
    a=p.parse_args();guard()
    if a.stage=='prepare':prepare(a.out)
    elif a.stage=='analyze':analyze(a.out)
    elif a.stage=='backend':backend_check(a.out)
    elif a.stage=='initial-audit':initial_audit(a.out)
    elif a.stage=='audit':audit(a.out)
    elif a.stage=='plot':plot(a.out)
    else:
        import warp as wp
        from utils.resource_guard import prepare_warp_cache
        from .aniso_history_increment import DATA_ROOT
        wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)
        run(a.out,a.case,a.grid,a.dt,a.device,a.duration)


if __name__=='__main__':main()
