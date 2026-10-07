"""Continue fixed-space time refinement before testing finer references."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from engine.aniso_phase1.refinement import CachedFrozenQ1, TensorReferenceQ1
from engine.aniso_phase1.consistent_transfer import material_response
from engine.aniso_phase1.history_increment import material_tangent
from engine.aniso_phase1.convergence_reference import (
    geometry, knots, union_knots, tensor_rule, sites, state_from_field,
    load_fields, save_fields, compare_fields, directions, rms)
from .aniso_convergence_reference import shared_axes, save, fingerprint
from .aniso_history_increment import guard, history_error


OLD = Path('docs/results/convergence-reference/v1')


def run_path(out, grid, case, initial, particles, dt, tolerance=1e-9, order=5, tensor=False, device=None):
    name = f'g{grid}-{case}-dt{dt:.12f}'
    config = dict(grid=grid, case=case, dt=dt, duration=.003, order=order,
                  tolerance=tolerance, initial=fingerprint(initial), protocol='refinement-v1',
                  mass_points=len(particles.X))
    if tensor:
        config['backend'] = 'exact-tensor-mass'
        config['device'] = device or 'cpu'
    meta_path = out/(name+'.json')
    field_path = out/(name+'.npz')
    if meta_path.exists() and field_path.exists():
        data = json.loads(meta_path.read_text())
        if data['config'] == config and all(data['checks'].values()):
            print(name+': cached', flush=True)
            return data, load_fields(field_path)
    start = time.perf_counter()
    guard()
    source = geometry(grid)
    axes = knots(source, case in ('shifted','enriched'))
    q = sites(*tensor_rule(union_knots(knots(geometry(17)), axes, knots(source)), order))
    state = state_from_field(initial['position'], particles, q)
    model = (TensorReferenceQ1(source,state,device=device) if tensor else
             CachedFrozenQ1(source, state, axes, enrich=case == 'enriched'))
    vp = initial['velocity'].evaluate(particles.X)[0]
    steps = round(.003/dt)
    if abs(steps*dt-.003) > 1e-14:
        raise ValueError('dt must divide the fixed release duration')
    rows = []
    for step in range(steps):
        guard()
        state, vp, info = model.step(state, vp, dt, tolerance=tolerance)
        rows.append(dict(step=step+1, **info))
        if (step+1) % 64 == 0 or step+1 == steps:
            print(f'{name}: {step+1}/{steps}, residual={info["scaled_residual_inf"]:.3g}', flush=True)
    errors = history_error(state)
    checks = dict(solve=max(r['scaled_residual_inf'] for r in rows) <= 1.01*tolerance,
                  history=max(errors.values()) < 1e-8,
                  admissible=min(r['min_det'] for r in rows) > 0,
                  clamp=max(r['clamp_speed'] for r in rows) < 1e-9,
                  budget=max(abs(r['energy_budget_residual']) for r in rows) < 1e-12,
                  dissipation=all(b['mechanical'] <= a['mechanical']+1e-10 for a,b in zip(rows,rows[1:])))
    result = dict(config=config, rows=rows, checks=checks, history_error=errors,
                  rank=model.rank_info, elapsed_seconds=time.perf_counter()-start,
                  material_points=len(q.X), storage=guard())
    if not all(checks.values()):
        raise RuntimeError(f'numerical acceptance failed: {checks}')
    fields = dict(position=state.field, velocity=model.last_velocity)
    save_fields(field_path, **fields)
    save(meta_path, result)
    if tensor:
        model.close()
    return result, fields


def compare(a, b, probes):
    report = compare_fields(a['position'], a['velocity'], b['position'], b['velocity'],
                            *probes, geometry(17).params)
    scales = report['normalizers']
    report['relative'] = {k: report['all'][k+'_rms']/max(scales[s],1e-30)
                         for k,s in [('x','displacement_rms'),('F','F_minus_I_rms'),('P','P_rms'),('v','v_rms')]}
    return report


def time_study(out, grids=(17,33), cases=('control','shifted','enriched'), target=.01, max_halvings=5):
    out.mkdir(parents=True,exist_ok=True)
    initial = load_fields(OLD/'initial-switch-state.npz')
    particles = sites(*tensor_rule(shared_axes(),2))
    probes = tensor_rule(shared_axes(),3)
    summary_path = out/'time-results.json'
    result = dict(protocol='refinement-v1', target=target, normalization=dict(v='mass velocity',F='F-I',P='P'),
                  initial=fingerprint(initial),storage_start=guard(), paths={})
    for grid in grids:
        for case in cases:
            dt = .00003125
            previous = load_fields(OLD/f'g{grid}-{case}-dt{dt:.9f}.npz')
            row = dict(pairs=[], passed=False)
            result['paths'][f'g{grid}-{case}'] = row
            for level in range(max_halvings):
                dt /= 2
                meta, fields = run_path(out,grid,case,initial,particles,dt)
                metrics = compare(previous,fields,probes)
                row['pairs'].append(dict(dt_coarse=2*dt,dt_fine=dt,metrics=metrics))
                row['final_dt'] = dt
                row['final_file'] = f'g{grid}-{case}-dt{dt:.12f}.npz'
                row['passed'] = all(metrics['relative'][k] < target for k in ('v','F','P'))
                print(f'g{grid}-{case}: dt={dt:g}, relative={metrics["relative"]}, passed={row["passed"]}',flush=True)
                save(summary_path,result)
                previous = fields
                if row['passed']:
                    break
    result['all_passed'] = all(row['passed'] for row in result['paths'].values())
    result['storage_end'] = guard()
    save(summary_path,result)
    plot_time(out,result)
    return result


def solver_study(out):
    checkpoint=out/'solver-check'; checkpoint.mkdir(parents=True,exist_ok=True)
    initial=load_fields(OLD/'initial-switch-state.npz')
    particles=sites(*tensor_rule(shared_axes(),2))
    _,tight=run_path(checkpoint,33,'enriched',initial,particles,.00000390625,tolerance=1e-10)
    normal=load_fields(out/'g33-enriched-dt0.000003906250.npz')
    metrics=compare(normal,tight,tensor_rule(shared_axes(),3))
    save(checkpoint/'metrics.json',metrics)
    return metrics


def reference_study(out,grid,device=None):
    out.mkdir(parents=True,exist_ok=True)
    initial=load_fields(OLD/'initial-switch-state.npz')
    axes=union_knots(knots(geometry(17)),knots(geometry(grid)))
    particles=sites(*tensor_rule(axes,2))
    probes=tensor_rule(axes,3)
    result=dict(grid=grid,device=device,pairs=[],runs=[],storage_start=guard())
    previous=None
    for dt in (.000015625,.0000078125,.00000390625):
        meta,fields=run_path(out,grid,'reference',initial,particles,dt,tensor=True,device=device)
        result['runs'].append(dict(dt=dt,config=meta['config'],checks=meta['checks']))
        if previous is not None:
            metrics=compare(previous,fields,probes)
            result['pairs'].append(dict(dt_coarse=2*dt,dt_fine=dt,metrics=metrics))
            print(f'REFERENCE g{grid}: dt={dt:g} relative={metrics["relative"]}',flush=True)
        previous=fields
        save(out/f'reference-time-g{grid}.json',result)
    result['time_passed_1pct']=all(result['pairs'][-1]['metrics']['relative'][k]<.01 for k in ('v','F','P'))
    result['storage_end']=guard()
    save(out/f'reference-time-g{grid}.json',result)
    return result


def field_axes(*sets):
    groups = [[np.unique(b.X[:,d]) for d in range(3)]
              for fields in sets for b,_ in fields['position'].terms
              if getattr(b, 'kind', '') != 'global-polynomial-v1']
    if not groups:
        raise ValueError('analytic history has no cell interfaces; supply explicit integration axes')
    return union_knots(*groups)


def initial_trace_audit(initial):
    """Measure, without changing, the shared piecewise-Q1 initial history."""
    axes=knots(geometry(17)); z,w=np.polynomial.legendre.leggauss(5)
    points=[]; weights=[]
    for a in axes[1:]:
        h=np.diff(a)
        points.append(((a[:-1,None]+a[1:,None])/2+h[:,None]*z/2).ravel())
        weights.append((h[:,None]*w/2).ravel())
    yz=np.array(np.meshgrid(*points,indexing='ij')).reshape(2,-1).T
    area=np.outer(*weights).ravel()
    result={}
    for eps in (1e-7,1e-8):
        rows=[]
        for plane in axes[0][1:-1]:
            X=np.column_stack((np.full(len(yz),plane),yz))
            Xm=X.copy(); Xp=X.copy(); Xm[:,0]-=eps; Xp[:,0]+=eps
            xm,Fm=initial['position'].evaluate(Xm)
            xp,Fp=initial['position'].evaluate(Xp)
            A=directions(X)
            Pm=material_response(Fm,A,geometry(17).params)[1]
            Pp=material_response(Fp,A,geometry(17).params)[1]
            rows.append(dict(X=float(plane),x_trace_difference=rms(xp-xm,area),
                             F_trace_difference=rms(Fp-Fm,area),
                             traction_trace_difference=rms(Pp[:,:,0]-Pm[:,:,0],area)))
        result[str(eps)]=rows
    return result


def endpoint_audit(out,grid,case,final,initial=None):
    initial=load_fields(OLD/'initial-switch-state.npz') if initial is None else initial
    signature=dict(grid=grid,case=case,initial=fingerprint(initial),final=fingerprint(final))
    path=out/f'endpoint-g{grid}-{case}.json'
    if path.exists():
        cache=json.loads(path.read_text())
        if cache['signature']==signature: return cache['metrics']
    guard(); source=geometry(grid)
    ax=knots(source,case=='enriched')
    axes=union_knots(knots(geometry(17)),knots(source),ax)
    mass_axes=shared_axes() if case=='enriched' else axes
    particles=sites(*tensor_rule(mass_axes,2))
    q=sites(*tensor_rule(axes,5))
    state=state_from_field(initial['position'],particles,q)
    model=(CachedFrozenQ1(source,state,ax) if case=='enriched' else TensorReferenceQ1(source,state))
    direction=model.R @ (np.random.default_rng(481).normal(size=(model.R.shape[1],3))*.01)
    values=[]
    for order in (5,7):
        guard(); X,w=tensor_rule(axes,order); A=directions(X)
        mapping=model.map_points(X,w,A)
        F=final['position'].evaluate(X)[1]
        psi,P=material_response(F,A,source.params)
        force=sum(D.T @ (w[:,None]*P[:,:,d]) for d,D in enumerate(mapping.D))
        dP=material_tangent(F,A,model.gradient(mapping,direction),source.params)
        action=sum(D.T @ (w[:,None]*dP[:,:,d]) for d,D in enumerate(mapping.D))
        values.append((float(w @ psi),model.R.T @ force,model.R.T @ action))
    metrics={k:float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))
             for k,a,b in zip(('energy','force','tangent_action'),values[0],values[1])}
    metrics['passed_1e-6']=max(metrics.values())<1e-6
    if case!='enriched': model.close()
    save(path,dict(signature=signature,metrics=metrics))
    return metrics


def analyze(out,reference_grids=(65,81,97)):
    print('analysis: common probes and reference comparisons',flush=True)
    time_result=json.loads((out/'time-results.json').read_text())
    fields={name:load_fields(out/row['final_file']) for name,row in time_result['paths'].items()}
    references={g:load_fields(out/f'g{g}-reference-dt0.000003906250.npz') for g in reference_grids}
    ref=references[reference_grids[-1]]
    probes=tensor_rule(field_axes(ref,references[reference_grids[-2]],*fields.values()),3)
    result=dict(time=time_result,reference_time={},reference_space={},spatial={},scheme_differences={},
                reference_to_scheme_ratios={},probe_points=len(probes[0]),storage_start=guard())
    result['initial_trace']=initial_trace_audit(load_fields(OLD/'initial-switch-state.npz'))
    for g in reference_grids:
        result['reference_time'][str(g)]=json.loads((out/f'reference-time-g{g}.json').read_text())
    for a,b in zip(reference_grids,reference_grids[1:]):
        pair_probes=tensor_rule(field_axes(references[a],references[b]),3)
        result['reference_space'][f'{a}-{b}']=compare(references[a],references[b],pair_probes)
    delta=compare(references[reference_grids[-2]],ref,probes)
    result['reference_uncertainty']=delta
    for name,f in fields.items():
        print(f'analysis: {name} against finest reference',flush=True)
        result['spatial'][name]=compare(f,ref,probes)
    for grid in (17,33):
        gap=compare(fields[f'g{grid}-enriched'],fields[f'g{grid}-shifted'],probes)
        result['scheme_differences'][str(grid)]=gap
        result['reference_to_scheme_ratios'][str(grid)]={
            key:delta['all'][key+'_rms']/max(gap['all'][key+'_rms'],1e-30) for key in ('x','F','P','v')}
    legacy=load_fields(OLD/'g65-reference-dt0.000015625.npz')
    repeated=load_fields(out/'g65-reference-dt0.000015625000.npz')
    result['backend_trajectory_check']=compare(legacy,repeated,tensor_rule(shared_axes(),3))
    tight=load_fields(out/'solver-check/g33-enriched-dt0.000003906250.npz')
    result['solver_tolerance_check']=compare(fields['g33-enriched'],tight,tensor_rule(shared_axes(),3))
    save(out/'analysis-progress.json',result)
    print('analysis: endpoint quadrature audits',flush=True)
    result['quadrature_checks']={name:endpoint_audit(out,grid,case,f) for name,grid,case,f in
        [('g33-enriched',33,'enriched',fields['g33-enriched']),('g97-reference',97,'reference',ref)]}
    fine_probes=tensor_rule(field_axes(ref,fields['g33-enriched']),5)
    print('analysis: independent probe refinement',flush=True)
    probe_comparison=compare(fields['g33-enriched'],ref,fine_probes)
    result['probe_check']={k:abs(result['spatial']['g33-enriched']['all'][k+'_rms']/probe_comparison['all'][k+'_rms']-1)
                           for k in ('x','F','P','v')}
    run_files=sorted(out.glob('g*-dt*.json'))+sorted((out/'solver-check').glob('g*-dt*.json'))
    runs=[json.loads(path.read_text()) for path in run_files]
    rows=[row for run in runs for row in run['rows']]
    result['run_audit']=dict(trajectories=len(runs),steps=len(rows),
        min_det=min(row['min_det'] for row in rows),
        max_residual=max(row['scaled_residual_inf'] for row in rows),
        max_clamp=max(row['clamp_speed'] for row in rows),
        max_budget=max(abs(row['energy_budget_residual']) for row in rows),
        max_history=max(max(run['history_error'].values()) for run in runs),
        summed_run_seconds=sum(run['elapsed_seconds'] for run in runs))
    result['numerical_checks']=dict(
        trajectories=all(all(run['checks'].values()) for run in runs),
        backend=max(result['backend_trajectory_check']['relative'].values())<1e-6,
        solver_tolerance=max(result['solver_tolerance_check']['relative'].values())<.001,
        quadrature=all(row['passed_1e-6'] for row in result['quadrature_checks'].values()),
        probes=max(result['probe_check'].values())<1e-6)
    result['gates']=dict(
        candidate_time_1pct=time_result['all_passed'],
        reference_time_1pct=all(r['time_passed_1pct'] for r in result['reference_time'].values()),
        reference_below_one_tenth_scheme_gap=all(v<.1 for row in result['reference_to_scheme_ratios'].values() for v in row.values()),
        switch_error_concentration=all(result['spatial'][f'g{g}-enriched']['switch']['F_squared_error_share']>.5 for g in (17,33)))
    result['local_enrichment_justified']=all(result['gates'].values())
    result['storage_end']=guard()
    save(out/'results.json',result)
    plot(out,result)
    print(f'analysis complete: numerical={result["numerical_checks"]}, accuracy gates={result["gates"]}',flush=True)
    return result


def time_axes(axes,paths):
    for ax,key in zip(axes,('v','F','P')):
        for name,row in paths.items():
            ax.loglog([p['dt_coarse']*1000 for p in row['pairs']],
                      [p['metrics']['relative'][key]*100 for p in row['pairs']],'o-',label=name)
        ax.axhline(1,color='black',linestyle='--',linewidth=1)
        ax.set(title=f'Adjacent time-step difference: {key}',xlabel='Coarse dt (ms)',ylabel='Relative difference (%)')
        ax.legend(fontsize=6); ax.grid(alpha=.2)


def plotting():
    os.environ.setdefault('MPLCONFIGDIR','/tmp/mpm-lite-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_time(out,result):
    plt=plotting()
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    time_axes(axes,result['paths'])
    fig.tight_layout(); fig.savefig(out/'time-convergence.png',dpi=160); plt.close(fig)


def plot(out,result):
    plt=plotting()
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    time_axes(axes[0],result['time']['paths'])
    for ax,key in zip(axes[1,:2],('F','P')):
        for name,row in result['spatial'].items():
            ax.plot(row['profile']['x'],row['profile'][key+'_rms'],label=name)
        ax.axvspan(.4375,.5625,color='blue',alpha=.07)
        ax.axvspan(.25,.3125,color='red',alpha=.07)
        ax.set(title=f'{key} difference from grid 97 reference',xlabel='Material X coordinate',ylabel='Local RMS difference')
        ax.legend(fontsize=6)
    ax=axes[1,2]; keys=('x','F','P','v'); xx=np.arange(4)
    for i,grid in enumerate(('17','33')):
        ax.bar(xx+(i-.5)*.35,[result['reference_to_scheme_ratios'][grid][k] for k in keys],width=.35,label=f'grid {grid}')
    ax.axhline(.1,color='black',linestyle='--',label='Required < 0.1')
    ax.set_xticks(xx,keys); ax.set_yscale('log')
    ax.set(title='Reference change / enrichment field difference',ylabel='Ratio (log scale)'); ax.legend(fontsize=7)
    for ax in axes.ravel(): ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'summary.png',dpi=170); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('docs/results/refinement/v1'))
    parser.add_argument('--grids',type=int,nargs='+',choices=[17,33],default=[17,33])
    parser.add_argument('--cases',nargs='+',choices=['control','shifted','enriched'],default=['control','shifted','enriched'])
    parser.add_argument('--max-halvings',type=int,default=5)
    parser.add_argument('--stage',choices=['time','reference','solver','analyze'],default='time')
    parser.add_argument('--reference-grid',type=int,default=97)
    parser.add_argument('--device',default=None)
    args=parser.parse_args()
    if args.max_halvings<1: parser.error('--max-halvings must be positive')
    if args.device:
        from utils.resource_guard import prepare_warp_cache
        from .aniso_history_increment import DATA_ROOT
        import warp as wp
        wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)
    if args.stage=='time':
        time_study(args.output,args.grids,args.cases,max_halvings=args.max_halvings)
    elif args.stage=='reference':
        reference_study(args.output,args.reference_grid,args.device)
    elif args.stage=='solver':
        solver_study(args.output)
    else:
        result=analyze(args.output)
        if not all(result['numerical_checks'].values()):
            raise SystemExit(f'Numerical audits failed: {result["numerical_checks"]}')


if __name__ == '__main__':
    main()
