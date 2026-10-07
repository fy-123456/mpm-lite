"""D1-D8 correctness, stability and profiling on frozen v22 operators.

Run with one BLAS thread, e.g. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m benchmarks.research_d.run --out docs/results/parallel-v22/D/<run_id>
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import time
import unittest
import numpy as np
import warp as wp
from engine.aniso_phase1.research_d.identity import write_json,baseline_audit,own_sources,digest
from .protocol import GATES,CASES,freeze,problem,environment,storage_check


def difference(a,b,absolute=1e-8):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),absolute))


def timed(call,repeats=7,sync=None):
    rows=[]
    for _ in range(repeats):
        if sync:sync()
        start=time.perf_counter(); call()
        if sync:sync()
        rows.append(time.perf_counter()-start)
    return dict(raw_seconds=rows,median_seconds=float(np.median(rows)),mean_seconds=float(np.mean(rows)),
                p95_seconds=float(np.percentile(rows,95)),relative_range=float((max(rows)-min(rows))/np.median(rows)))


def static_cases(out):
    from engine.aniso_phase1.research_d.cpu import pcg
    from engine.aniso_phase1.research_d.gpu import TensorGPU,ResidentPCG
    from benchmarks.aniso_local_q3 import gradient,chunks
    records=[]
    for case in CASES:
        print('static',case['name'],flush=True); start=time.perf_counter(); p=problem(case)
        construct=time.perf_counter()-start; write_json(out/(case['name']+'-input.json'),p.identity)
        start=time.perf_counter(); gpu=TensorGPU(p); backend_build=time.perf_counter()-start
        if gpu.memory_bytes>1<<30: raise MemoryError('declared GPU workspace budget exceeded')
        resident=ResidentPCG(gpu,maxiter=2500,check_every=8)
        rng=np.random.default_rng(20260930); v=rng.normal(size=p.size); w=rng.normal(size=p.size)
        a=p.apply(v); b=gpu.host_apply(v)
        oracle=None
        if p.size<=3000:oracle=difference(a,p.assembled()@v,GATES['operator_absolute'])
        symmetry=abs(v@p.apply(w)-w@a)/max(np.linalg.norm(v)*np.linalg.norm(p.apply(w)),1e-12)
        eps_rows=[]
        for eps in (1e-3,1e-4,1e-5,1e-6):
            energy=lambda z:.5*z@p.apply(z)
            fd=(energy(v+eps*w)-energy(v-eps*w))/(2*eps)
            eps_rows.append(dict(eps=eps,relative=abs(fd-w@a)/max(abs(w@a),1e-8)))
        solves=[]; solutions={}; metrics={}
        for tol in GATES['linear_rtol']:
            for backend in ('cpu','gpu'):
                x,r=(pcg(p.apply,p.rhs,p.precondition,rtol=tol,maxiter=2500) if backend=='cpu'
                     else resident.solve(p.rhs,rtol=tol))
                key=f'{backend}-{tol:g}'; solutions[key]=x; metrics[key]=p.metrics(x)
                solves.append(dict(backend=backend,rtol=tol,**r.record(),physics=metrics[key]))
        cpu=solutions['cpu-1e-08']; device=solutions['gpu-1e-08']
        # Full active physical domain, exact declared quadrature, no selected probes.
        field_totals=np.zeros(2)
        for X,V in chunks(p.edges,order=p.degree+1,size=64):
            P0=gradient(X,p.edges,p.degree,p.field(cpu)).reshape(-1,9)@p.H.T
            P1=gradient(X,p.edges,p.degree,p.field(device)).reshape(-1,9)@p.H.T
            field_totals += [np.sum(V[:,None]*(P0-P1)**2),np.sum(V[:,None]*P0**2)]
        stress=float(np.sqrt(field_totals[0]/max(field_totals[1],1e-16)))
        reaction=abs(metrics['cpu-1e-08']['reaction_N']-metrics['gpu-1e-08']['reaction_N'])
        tolerance_sensitivity=difference(solutions['cpu-1e-06'],cpu)
        # A/A and stage profiling are separate from the formal paired benchmark.
        vd=wp.array(v,dtype=wp.float64,device='cuda:0'); yd=wp.empty_like(vd)
        stages=dict(cpu_operator=timed(lambda:p.apply(v)),cpu_preconditioner=timed(lambda:p.precondition(v)),
            gpu_operator_with_launch_sync=timed(lambda:gpu.apply(vd,yd),sync=wp.synchronize),
            gpu_preconditioner_with_launch_sync=timed(lambda:gpu.precondition(vd,yd),sync=wp.synchronize),
            h2d=timed(lambda:vd.assign(v),sync=wp.synchronize),d2h=timed(lambda:vd.numpy()),
            device_reduce_with_sync=timed(lambda:resident.dot(vd,vd,0),sync=wp.synchronize),
            empty_sync=timed(wp.synchronize))
        _,true_trace=resident.solve(p.rhs,rtol=1e-7,diagnostic_true_residuals=True)
        aa=dict(cpu=timed(lambda:pcg(p.apply,p.rhs,p.precondition,rtol=1e-7,maxiter=2500),repeats=3),
                gpu=timed(lambda:resident.solve(p.rhs,rtol=1e-7),repeats=3))
        passed=bool((oracle is None or oracle<=GATES['operator_relative']) and difference(a,b)<=GATES['operator_relative']
            and symmetry<=GATES['symmetry_relative'] and min(r['relative'] for r in eps_rows)<=GATES['energy_fd_relative']
            and all(r['converged'] and r['physics']['constraint_residual']<=GATES['constraint_absolute'] and
                    r['physics']['finite'] and r['physics']['work_identity_absolute_J']<=GATES['work_absolute_J'] for r in solves)
            and stress<=GATES['field_relative'] and reaction<=GATES['reaction_absolute_N']+
                GATES['field_relative']*abs(metrics['cpu-1e-08']['reaction_N'])
            and tolerance_sensitivity<=GATES['field_relative'])
        record=dict(case=case,identity_sha256=p.key,free_dofs=p.size,quadrature_points=int(np.prod(case['cells']))*(case['p']+1)**3,
            particle_count=None,particle_dependence='static tensor operator has no particle loop',
            construct_seconds=construct,gpu_backend_build_seconds=backend_build,
            oracle_relative=oracle,gpu_operator_relative=difference(a,b),symmetry_relative=float(symmetry),
            energy_derivative_scan=eps_rows,gpu_true_residual_trace=true_trace.record(),solves=solves,stress_relative=stress,reaction_absolute_N=reaction,
            tolerance_sensitivity=tolerance_sensitivity,stages=stages,aa=aa,passed=passed,
            memory=dict(backend_bytes=gpu.memory_bytes,workspace_bytes=resident.memory_bytes),environment=environment())
        write_json(out/(case['name']+'.json'),record); np.savez_compressed(out/(case['name']+'.npz'),**solutions)
        records.append(record); resident.close();gpu.close();gc.collect()
    write_json(out/'static-summary.json',dict(passed=all(r['passed'] for r in records),cases=records))
    return all(r['passed'] for r in records)


def precondition_cases(out):
    from engine.aniso_phase1.research_d.cpu import pcg
    from engine.aniso_phase1.research_d.precondition import Schwarz,tensor_blocks,BatchedSchwarz
    from engine.aniso_phase1.research_d.gpu import CSRGPU,ResidentPCG
    from engine.aniso_phase1.research_d.heterogeneous import FrozenHeterogeneous
    from engine.aniso_phase1.beam_reference import reference_hessian
    controls=[problem(CASES[0]),problem(CASES[4])]; p=controls[0]
    for ratio in (10.,100.):
        H=[reference_hessian(kf=200.*(ratio if i%2 else 1),direction=(1.,0.,0.) if i%2 else (2**-.5,2**-.5,0.))
           for i in range(len(p.edges[0])-1)]
        controls.append(FrozenHeterogeneous(p.edges,2,H,name=f'heterogeneous-ratio-{ratio:g}'))
    rows=[]
    for p in controls:
        start=time.perf_counter();A=p.assembled();assembly=time.perf_counter()-start
        base_x,base_r=pcg(p.apply,p.rhs,p.precondition,rtol=1e-7)
        candidates=[]
        for width in (2,3):
            blocks=tensor_blocks(p.free_shape,width,1)
            Z=np.zeros((p.size,3))
            for j in range(3):Z[j*p.size//3:(j+1)*p.size//3,j]=1.
            for kind in ('local','combined'):
                M=Schwarz(A,blocks,base=p.precondition if kind=='combined' else None,coarse=Z if kind=='combined' else None)
                x,r=pcg(p.apply,p.rhs,M,rtol=1e-7)
                candidate=dict(kind=kind,width=width,build_seconds=M.build_seconds,assembly_seconds=assembly,
                    total_seconds=assembly+M.build_seconds+r.seconds,memory_bytes=M.memory_bytes,
                    solve=r.record(),solution_relative=difference(x,base_x),physics=p.metrics(x))
                if kind=='local':
                    start=time.perf_counter();batch=BatchedSchwarz(M); upload=time.perf_counter()-start
                    v=np.random.default_rng(22).normal(size=p.size);vd=wp.array(v,dtype=wp.float64,device='cuda:0');outd=wp.empty_like(vd)
                    batch.apply(vd,outd);candidate['batch_relative']=difference(outd.numpy(),M(v))
                    candidate['batch_upload_seconds']=upload;candidate['batch_bytes']=batch.memory_bytes
                    candidate['batch_apply']=timed(lambda:batch.apply(vd,outd),sync=wp.synchronize)
                    gpu=CSRGPU(A);gpu.precondition=batch.apply
                    solve=ResidentPCG(gpu,maxiter=2500);gx,gr=solve.solve(p.rhs,rtol=1e-7)
                    candidate['gpu_solve']=gr.record();candidate['gpu_solution_relative']=difference(gx,base_x)
                    solve.close();gpu.close()
                candidate['passed']=bool(r.converged and candidate['solution_relative']<=GATES['field_relative']
                    and candidate.get('batch_relative',0)<=GATES['operator_relative']
                    and candidate.get('gpu_solve',{'converged':True})['converged']
                    and candidate.get('gpu_solution_relative',0)<=GATES['field_relative'])
                candidates.append(candidate)
        row=dict(name=p.name,identity=p.identity,baseline=base_r.record(),baseline_physics=p.metrics(base_x),candidates=candidates,
                 passed=base_r.converged and all(c['passed'] for c in candidates),
                 fallback='exact assembled CSR for heterogeneous GPU; not a heterogeneous tensor port')
        rows.append(row);write_json(out/(p.name+'-precondition.json'),row)
    write_json(out/'precondition-summary.json',dict(passed=all(r['passed'] for r in rows),cases=rows))
    return all(r['passed'] for r in rows)


def nonlinear_cases(out):
    from .nonlinear import implicit_checks,scene_run
    implicit={}; raws={}; scenes={}; arrays={}
    for device in ('cpu','cuda:0'):
        label=device.replace(':','-'); print('implicit',device,flush=True)
        rows,raw=implicit_checks(device);implicit[device]=rows;raws[device]=raw
        write_json(out/f'implicit-{label}.json',rows);np.savez_compressed(out/f'implicit-{label}.npz',**raw)
        for name,samples,angle,steps,dt,speed in [
                ('fixed',2,0.,60,.0005,.01),('fixed',2,90.,60,.0005,.01),
                ('tensile',1,45.,60,.0005,.01),('tensile',2,45.,60,.0005,.01),
                ('tensile',2,45.,80,.0025,.1)]:
            tag=f'{name}-ppc{samples}-angle{angle:g}'+('-full-amplitude' if dt>.001 else '')
            print('scene',device,tag,flush=True)
            r,a=scene_run(device,name,steps,samples,angle,dt=dt,speed=speed);scenes[device,tag]=r;arrays[device,tag]=a
            write_json(out/f'scene-{tag}-{label}.json',r);np.savez_compressed(out/f'scene-{tag}-{label}.npz',**a)
    comparisons=[]
    for tag in sorted(k[1] for k in scenes if k[0]=='cpu'):
        a,b=arrays['cpu',tag],arrays['cuda:0',tag]
        cr,gr=scenes['cpu',tag],scenes['cuda:0',tag]
        shape_ok=all(a[k].shape==b[k].shape for k in a)
        if shape_ok:
            displacement=difference(b['positions']-b['reference'],a['positions']-a['reference'])
            stress=difference(b['stresses'],a['stresses']);final_F=difference(b['final_F'],a['final_F'])
        else:displacement=stress=final_F=None
        reaction=None
        if shape_ok and tag.startswith('tensile'):
            cf=np.array([r['right_force'] for r in cr['rows']]);gf=np.array([r['right_force'] for r in gr['rows']])
            reaction=difference(gf,cf,GATES['reaction_absolute_N'])
        passed=bool(cr['passed'] and gr['passed'] and shape_ok and max(displacement,stress,final_F)<=GATES['field_relative']
                    and (reaction is None or reaction<=GATES['field_relative']))
        comparisons.append(dict(tag=tag,displacement_relative=displacement,stress_relative=stress,
                                 final_F_relative=final_F,reaction_relative=reaction,passed=passed))
    implicit_diff={k:difference(raws['cuda:0'][k],raws['cpu'][k]) for k in raws['cpu']}
    result=dict(passed=all(r['passed'] for rows in implicit.values() for r in rows)
        and all(r['passed'] for r in comparisons) and max(implicit_diff.values())<=GATES['field_relative'],
        implicit_device_differences=implicit_diff,scene_comparisons=comparisons,
        new_tensor_backend_dynamic_integration=False)
    write_json(out/'nonlinear-summary.json',result);return result['passed']


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--out',required=True)
    parser.add_argument('--phases',default='tests,static,precondition,nonlinear')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args();out=Path(args.out)
    if not args.resume:freeze(out)
    else:
        old=json.loads((out/'protocol.json').read_text())
        # Corrections get a distinct appended resume fingerprint, not overwritten protocol.
        write_json(out/('resume-'+str(time.time_ns())+'.json'),dict(prior_source=old['source_sha256'],source=own_sources(),environment=environment()))
    cache,storage=storage_check(os.environ.get('MPM_LITE_WARP_CACHE','/tmp/mpm-lite-research-d-cache'))
    wp.config.kernel_cache_dir=cache;wp.init();write_json(out/('resources-'+str(time.time_ns())+'.json'),dict(storage=storage,environment=environment()))
    phases=args.phases.split(',');results={}
    if 'tests' in phases:
        suite=unittest.defaultTestLoader.discover('tests/research_d',pattern='test_*.py',top_level_dir='.')
        with (out/'tests.log').open('x') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
        results['tests']=r.wasSuccessful();write_json(out/'tests.json',dict(tests=r.testsRun,passed=r.wasSuccessful(),skipped=len(r.skipped)))
    for name,func in [('static',static_cases),('precondition',precondition_cases),('nonlinear',nonlinear_cases)]:
        if name in phases:results[name]=func(out)
    write_json(out/('phase-summary-'+str(time.time_ns())+'.json'),dict(phases=results,passed=all(results.values()),environment=environment()))
    print(json.dumps(results),flush=True)
    raise SystemExit(0 if all(results.values()) else 2)

if __name__=='__main__':main()
