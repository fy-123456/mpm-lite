"""Reference refinement measured against scheme gaps, with constrained controls."""
import argparse
import json
from pathlib import Path
import numpy as np

from engine.aniso_phase1.convergence_reference import (
    geometry,knots,tensor_rule,load_fields,save_fields,directions,sites,state_from_field,rms)
from engine.aniso_phase1.consistent_transfer import material_response
from engine.aniso_phase1.initial_controls import energy,velocity_moments
from engine.aniso_phase1.constrained_initial import momentum_energy_velocity,StressFit
from .aniso_spatial_energy import OUT as PREVIOUS,initial_for,cached_compare,run as run_path
from .aniso_refinement import OLD,field_axes,compare,endpoint_audit
from .aniso_convergence_reference import save,fingerprint
from .aniso_history_increment import guard,DATA_ROOT

OUT=Path('docs/results/reference-scale/v1')
TIME_TARGET=.05
SPACE_TARGET=.1


def initial(out,case):
    if case=='original':return load_fields(OLD/'initial-switch-state.npz')
    if case in ('matched','velocity'):return initial_for(PREVIOUS,case)
    return load_fields(out/f'{case}-initial.npz')


def prepare(out):
    out.mkdir(parents=True,exist_ok=True);guard()
    original=initial(out,'original');matched=initial(out,'matched');old_velocity=initial(out,'velocity')
    X,w=tensor_rule(knots(geometry(17)),7)
    velocity,vinfo=momentum_energy_velocity(original['velocity'],old_velocity['velocity'],X,w)
    momentum=dict(position=matched['position'],velocity=velocity)
    print('Momentum-energy control constructed',flush=True)
    fitter=StressFit(original['position'],matched['position'],X,w,geometry(17).params)
    position,sinfo=fitter.solve()
    stress=dict(position=position,velocity=original['velocity'])
    print('Stress fit',sinfo['iterations'],sinfo['initial_stress_relative'],sinfo['final_stress_relative'],flush=True)
    X9,w9=tensor_rule(knots(geometry(17)),9)
    v0=original['velocity'].evaluate(X9)[0];vs=velocity.evaluate(X9)[0]
    p0=w9 @ v0;ps=w9 @ vs;K0=.5*float(np.sum(w9[:,None]*v0*v0));Ks=.5*float(np.sum(w9[:,None]*vs*vs))
    x=matched['position'].evaluate(X9)[0]
    U0=energy(original['position'],X9,w9,geometry(17).params)
    Us=energy(position,X9,w9,geometry(17).params)
    scompare=compare(stress,original,(X9,w9));mcompare=compare(matched,original,(X9,w9))
    checks=dict(kinetic=abs(Ks/K0-1)<1e-9,momentum=np.linalg.norm(ps-p0)/np.linalg.norm(p0)<1e-9,
        elastic=abs(Us/U0-1)<1e-8,stress=scompare['all']['P_rms']<mcompare['all']['P_rms'],
        displacement=scompare['relative']['x']<=mcompare['relative']['x']+1e-8,
        deformation=scompare['relative']['F']<=mcompare['relative']['F']+1e-8)
    checks={key:bool(value) for key,value in checks.items()}
    result=dict(protocol='reference-scale-v1',velocity=vinfo,stress=sinfo,checks=checks,
        independent=dict(kinetic_relative=abs(Ks/K0-1),momentum_relative=float(np.linalg.norm(ps-p0)/np.linalg.norm(p0)),
            elastic_relative=abs(Us/U0-1),stress=scompare,matched=mcompare,
            original_angular_momentum=np.sum(w9[:,None]*np.cross(x,v0),axis=0).tolist(),
            new_angular_momentum=np.sum(w9[:,None]*np.cross(x,vs),axis=0).tolist()),
        fingerprints=dict(original=fingerprint(original),matched=fingerprint(matched),
                          momentum=fingerprint(momentum),stress=fingerprint(stress)),storage=guard())
    if not all(checks.values()):raise RuntimeError(f'initial control acceptance failed: {checks}')
    save_fields(out/'momentum-initial.npz',**momentum);save_fields(out/'stress-initial.npz',**stress)
    save(out/'initial-controls.json',result)
    return result


def catalog(out,case):
    result={};signature=fingerprint(initial(out,case))
    roots=[PREVIOUS,out] if case in ('original','matched','velocity') else [out]
    for root in roots:
        for p in (root/case).glob('g*-dt*.json'):
            m=json.loads(p.read_text());c=m['config']
            if c['initial']!=signature or not all(m['checks'].values()):
                raise ValueError(f'initial/acceptance mismatch: {p}')
            if c['duration']==.003 and p.with_suffix('.npz').exists():result[(c['grid'],c['dt'])]=(p,m)
    return result


def fields(item):return load_fields(item[0].with_suffix('.npz'))


def candidate_fields(out):
    root=Path('docs/results/refinement/v1');meta=json.loads((root/'time-results.json').read_text())
    if meta['initial']!=fingerprint(initial(out,'original')):raise ValueError('candidate initial state mismatch')
    return {str(g):[load_fields(root/meta['paths'][f'g{g}-{case}']['final_file']) for case in ('shifted','enriched')]
            for g in (17,33)}


def scaled_differences(out,a,b,candidates):
    result={}
    for grid,pair in candidates.items():
        axes=field_axes(a,b,*pair)
        delta=cached_compare(out,a,b,axes);gap=cached_compare(out,*pair,axes)
        ratios={k:delta['all'][k+'_rms']/max(gap['all'][k+'_rms'],1e-30) for k in ('x','F','P','v')}
        result[grid]=dict(ratios=ratios,difference=delta,scheme_gap=gap)
    return result


def references(out):
    items={key:item for key,item in catalog(out,'original').items() if key[0]>=129}
    candidates=candidate_fields(out)
    result=dict(time_target=TIME_TARGET,space_target=SPACE_TARGET,time={},space={},storage_start=guard())
    for grid in sorted({g for g,dt in items}):
        pairs=[]
        for g,dt in sorted(items,reverse=True):
            if g!=grid or (g,2*dt) not in items:continue
            comparison=scaled_differences(out,fields(items[(g,2*dt)]),fields(items[(g,dt)]),candidates)
            passed=all(row['ratios'][k]<TIME_TARGET for row in comparison.values() for k in ('F','P','v'))
            pairs.append(dict(dt_coarse=2*dt,dt_fine=dt,comparison=comparison,passed=passed))
            print('T',grid,dt,{g:r['ratios'] for g,r in comparison.items()},'passed',passed,flush=True)
            result['time'][str(grid)]=pairs;save(out/'references-progress.json',result)
    grids=sorted({g for g,dt in items})
    for ga,gb in zip(grids,grids[1:]):
        common=set(dt for g,dt in items if g==ga)&set(dt for g,dt in items if g==gb)
        if not common:continue
        dt=min(common)
        comparison=scaled_differences(out,fields(items[(ga,dt)]),fields(items[(gb,dt)]),candidates)
        passed=all(row['ratios'][k]<SPACE_TARGET for row in comparison.values() for k in ('F','P','v'))
        result['space'][f'{ga}-{gb}']=dict(dt=dt,comparison=comparison,passed=passed)
        print('R',ga,gb,dt,{g:r['ratios'] for g,r in comparison.items()},'passed',passed,flush=True)
        save(out/'references-progress.json',result)
    result['time_passed']=all(pairs and min(pairs,key=lambda p:p['dt_fine'])['passed'] for pairs in result['time'].values())
    last=result['space'].get('145-161')
    result['reference_ready']=bool(last and last['passed'] and result['time_passed'])
    result['storage_end']=guard();save(out/'references.json',result)
    return result


def controls(out):
    result=dict(initial=json.loads((out/'initial-controls.json').read_text()),time={},space={},effects={})
    runs={case:catalog(out,case) for case in ('matched','velocity','momentum','stress')}
    for case in ('momentum','stress'):
        items=runs[case];result['time'][case]={}
        for grid in sorted({g for g,dt in items}):
            pairs=[]
            for g,dt in sorted(items,reverse=True):
                if g!=grid or (g,2*dt) not in items:continue
                a,b=fields(items[(g,2*dt)]),fields(items[(g,dt)])
                report=cached_compare(out,a,b,field_axes(a,b))
                pairs.append(dict(dt_fine=dt,metrics=report,passed=all(report['relative'][k]<.01 for k in ('F','P','v'))))
                print('CONTROL TIME',case,g,report['relative'],flush=True)
            result['time'][case][str(grid)]=pairs
        common=set(dt for g,dt in items if g==81)&set(dt for g,dt in items if g==97)
        if common:
            dt=min(common);a,b=fields(items[(81,dt)]),fields(items[(97,dt)])
            result['space'][case]=dict(dt=dt,metrics=cached_compare(out,a,b,field_axes(a,b)))
        for baseline in (('matched','velocity') if case=='momentum' else ('matched',)):
            common=set(items)&set(runs[baseline])
            if common:
                grid=max(g for g,dt in common);dt=min(dt for g,dt in common if g==grid)
                a,b=fields(items[(grid,dt)]),fields(runs[baseline][(grid,dt)])
                result['effects'][case+'-'+baseline]=dict(grid=grid,dt=dt,metrics=cached_compare(out,a,b,field_axes(a,b)))
                if (grid,2*dt) in items and (grid,2*dt) in runs[baseline]:
                    coarse=[fields(items[(grid,2*dt)]),fields(runs[baseline][(grid,2*dt)])]
                    X,w=tensor_rule(field_axes(a,b,*coarse),3);A=directions(X);values=[]
                    for f in (a,b,*coarse):
                        x,F=f['position'].evaluate(X);v=f['velocity'].evaluate(X)[0]
                        values.append(dict(x=x,F=F,v=v,P=material_response(F,A,geometry(17).params)[1]))
                    result['effects'][case+'-'+baseline]['effect_time_relative']={
                        k:rms((values[2][k]-values[3][k])-(values[0][k]-values[1][k]),w)
                        /max(rms(values[0][k]-values[1][k],w),1e-30) for k in ('x','F','P','v')}
        save(out/'controls-progress.json',result)
    records=[m for case in ('original','momentum','stress') for p,m in catalog(out,case).values() if p.is_relative_to(out)]
    result['runs']=len(records);result['steps']=sum(len(m['rows']) for m in records)
    result['numerical_checks_passed']=all(all(m['checks'].values()) for m in records)
    rows=[row for m in records for row in m['rows']]
    result['numerics']=dict(minimum_det=min(row['min_det'] for row in rows),
        maximum_scaled_residual=max(row['scaled_residual_inf'] for row in rows),
        maximum_clamp_speed=max(row['clamp_speed'] for row in rows),
        maximum_budget_residual=max(abs(row['energy_budget_residual']) for row in rows),
        minimum_iterations=min(row['iterations'] for row in rows),
        maximum_iterations=max(row['iterations'] for row in rows),
        zero_iteration_steps=sum(row['iterations']==0 for row in rows),
        elapsed_run_seconds=sum(m['elapsed_seconds'] for m in records))
    result['time_passed']=all(p and min(p,key=lambda row:row['dt_fine'])['passed'] for gs in result['time'].values() for p in gs.values())
    result['storage_end']=guard();save(out/'controls.json',result)
    return result


def audit(out):
    from engine.aniso_phase1.refinement import TensorReferenceQ1
    result=dict(endpoints={},initial_velocity={},projection={})
    target=json.loads((out/'initial-controls.json').read_text())['velocity']
    for grid in (81,97):
        source=geometry(grid);v=initial(out,'momentum')['velocity']
        a,b=[velocity_moments(source,v,k) for k in (7,9)]
        result['initial_velocity'][str(grid)]=dict(kinetic=abs(a[0]/b[0]-1),rhs=float(np.linalg.norm(a[1]-b[1])/np.linalg.norm(b[1])))
        f=initial(out,'momentum');p=sites(*tensor_rule(knots(source),2))
        model=TensorReferenceQ1(source,state_from_field(f['position'],p,p))
        nodal=model.R @ model.mass_solve(model.R.T @ a[1])
        mass=np.asarray(model.particles.N.T @ p.weight).ravel();momentum=mass @ nodal
        result['projection'][str(grid)]=dict(kinetic_loss_relative=1-.5*float(np.sum(nodal*(model.M @ nodal)))/a[0],
            momentum_relative=float(np.linalg.norm(momentum-np.array(target['target_momentum']))/np.linalg.norm(target['target_momentum'])),
            momentum=momentum.tolist());model.close()
        print('AUDIT INITIAL VELOCITY',grid,result['initial_velocity'][str(grid)],flush=True)
        save(out/'audit-progress.json',result)
    for case in ('original','momentum','stress'):
        items=catalog(out,case);grid=max(g for g,dt in items);dt=min(dt for g,dt in items if g==grid)
        final=fields(items[(grid,dt)])
        result['endpoints'][case]=endpoint_audit(out,grid,case,final,initial(out,case))
        print('AUDIT',case,result['endpoints'][case],flush=True);save(out/'audit-progress.json',result)
    items=catalog(out,'original');ga,gb=sorted({g for g,dt in items})[-2:]
    common=set(dt for g,dt in items if g==ga)&set(dt for g,dt in items if g==gb);dt=min(common)
    a,b=fields(items[(ga,dt)]),fields(items[(gb,dt)])
    reports=[cached_compare(out,a,b,field_axes(a,b),k) for k in (3,5)]
    result['probes']={k:abs(reports[0]['all'][k+'_rms']/reports[1]['all'][k+'_rms']-1) for k in ('x','F','P','v')}
    print('AUDIT SPACE PROBES',result['probes'],flush=True)
    finest=min(dt for g,dt in items if g==gb)
    a,b=fields(items[(gb,2*finest)]),fields(items[(gb,finest)])
    reports=[cached_compare(out,a,b,field_axes(a,b),k) for k in (3,5)]
    result['time_probes']={k:abs(reports[0]['all'][k+'_rms']/reports[1]['all'][k+'_rms']-1) for k in ('x','F','P','v')}
    print('AUDIT TIME PROBES',result['time_probes'],flush=True)
    result['scheme_gap_probes']={}
    for key,pair in candidate_fields(out).items():
        common=cached_compare(out,*pair,field_axes(a,b,*pair),3)
        independent=cached_compare(out,*pair,field_axes(*pair),7)
        result['scheme_gap_probes'][key]={k:abs(common['all'][k+'_rms']/independent['all'][k+'_rms']-1) for k in ('x','F','P','v')}
    result['passed']=(all(v['passed_1e-6'] for v in result['endpoints'].values()) and max(result['probes'].values())<1e-6
                      and max(result['time_probes'].values())<1e-6
                      and all(max(v.values())<1e-6 for v in result['scheme_gap_probes'].values())
                      and all(max(v.values())<1e-9 for v in result['initial_velocity'].values()))
    save(out/'audit.json',result)
    if not result['passed']:raise RuntimeError('independent integration audit failed')
    return result


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    refs=json.loads((out/'references.json').read_text());control=json.loads((out/'controls.json').read_text())
    previous=json.loads((PREVIOUS/'results.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    rows=[(g,p) for g,pairs in refs['time'].items() for p in pairs]
    for j,candidate in enumerate(('17','33')):
        axes[0,0].bar(np.arange(len(rows))+(j-.5)*.35,
            [max(p['comparison'][candidate]['ratios'][k] for k in ('F','P','v')) for g,p in rows],
            width=.35,label=f'candidate {candidate}')
    axes[0,0].set_xticks(range(len(rows)),[f'{g}\n{p["dt_fine"]*1e6:.4f}' for g,p in rows],rotation=40)
    axes[0,0].axhline(TIME_TARGET,color='r',linestyle='--',label='0.05 target')
    axes[0,0].set_ylabel('max T_F, T_P, T_v');axes[0,0].set_xlabel('reference grid / fine dt (microseconds)');axes[0,0].legend()
    labels=list(refs['space'])
    for ax,k in ((axes[0,1],'F'),(axes[0,2],'P'),(axes[1,1],'v')):
        for candidate in ('17','33'):
            ax.plot(labels,[refs['space'][p]['comparison'][candidate]['ratios'][k]
                for p in labels],'o-',label=f'candidate {candidate}')
        ax.axhline(SPACE_TARGET,color='r',linestyle='--',label='0.1 target')
        ax.set_yscale('log');ax.set_ylabel(f'R_{k}');ax.set_xlabel('reference grid pair');ax.legend()
    init=control['initial']['independent']
    for j,name in enumerate(('matched','stress')):
        axes[1,0].bar(np.arange(3)+(j-.5)*.35,[init[name]['relative'][k]*100 for k in ('x','F','P')],
            width=.35,label=name)
    axes[1,0].set_xticks(range(3),['x','F','P']);axes[1,0].set_ylabel('initial difference from original (%)');axes[1,0].legend()
    cases=['matched','momentum','stress'];reports=[previous['space']['matched']['81-97']['metrics']]+[control['space'][c]['metrics'] for c in cases[1:]]
    for j,k in enumerate(('F','P','v')):
        axes[1,2].bar(np.arange(3)+(j-1)*.25,[r['relative'][k]*100 for r in reports],width=.25,label=k)
    axes[1,2].set_xticks(range(3),cases);axes[1,2].set_ylabel('81 to 97 spatial change (%)')
    axes[1,2].set_ylim(0,1.2*max(r['relative'][k]*100 for r in reports for k in ('F','P','v')))
    axes[1,2].legend(loc='upper center',ncol=3)
    fig.savefig(out/'convergence.png',dpi=180);fig.savefig(out/'convergence.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name in ('matched','stress'):
        profile=init[name]['profile']
        for ax,k in zip(axes,('F','P')):
            ax.plot(profile['x'],profile[k+'_rms'],label=name)
            ax.set_xlabel('material X coordinate');ax.set_ylabel(f'initial {k} difference RMS')
            ax.legend()
    fig.savefig(out/'initial-controls.png',dpi=180);fig.savefig(out/'initial-controls.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('prepare','run','references','controls','audit','plot'),required=True)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--case',choices=('original','momentum','stress'),default='original')
    p.add_argument('--grid',type=int,default=97);p.add_argument('--dt',type=float,default=.00000048828125)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();guard();a.out.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare':prepare(a.out)
    elif a.stage=='references':references(a.out)
    elif a.stage=='controls':controls(a.out)
    elif a.stage=='audit':audit(a.out)
    elif a.stage=='plot':plot(a.out)
    else:
        import warp as wp
        from utils.resource_guard import prepare_warp_cache
        wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)
        run_path(a.out,a.case,a.grid,a.dt,a.device,initial=initial(a.out,a.case),
                 initial_velocity_order=7 if a.case=='momentum' else None)


if __name__=='__main__':main()
