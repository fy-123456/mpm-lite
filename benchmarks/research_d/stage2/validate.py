"""Actual-space D3--D8 acceptance. Run only after protocol.py froze inputs."""
import json,os,time,resource,subprocess,platform
from pathlib import Path
import numpy as np
import warp as wp
from engine.aniso_phase1.research_d.frozen_inputs import load_frozen_inputs
from engine.aniso_phase1.research_d.stage2.gpu_operator import GPUOperator
from engine.aniso_phase1.research_d.stage2.contracts import sha,array_digest
from engine.aniso_phase1.research_d.stage2.solver import curvature,solve_linear,equilibrate
from engine.aniso_phase1.consistent_transfer import material_response
from benchmarks.research_d.stage2.bootstrap import PARENT,PIN

ROOT=Path.cwd();OUT=ROOT/'docs/results/parallel-v22-stage2/D/20260930-D-common-static'

def save(name,value):
    p=OUT/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def metric(a,b,rtol=1e-7,atol=1e-10,scale=1.):
    a,b=np.asarray(a),np.asarray(b);err=float(np.linalg.norm((a-b).ravel()));norm=float(np.linalg.norm(b.ravel()));limit=atol+rtol*max(norm,scale)
    return dict(error=err,reference_norm=norm,limit=limit,passed=err<=limit)

def probes(s,q):
    points=[s.test_vectors[f'points{k}'] for k in range(3)];x,F=s.evaluate(q,points)
    _,P=material_response(F.reshape(-1,3,3),np.broadcast_to(s.A,(F.size//9,3,3)),s.params)
    return dict(x=x,F=F,PK1=P)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    wp.config.kernel_cache_dir='/root/autodl-tmp/mpm-lite-research-d/stage2/warp-cache'
    s,_,_,baseline=load_frozen_inputs(ROOT/PARENT,ROOT,expected_sha256=PIN)
    save('baseline-before.json',baseline)
    protocol=json.loads((OUT/'protocol.json').read_text())
    with np.load(OUT/'vectors.npz') as z:v={k:z[k] for k in z.files}
    if any(array_digest(v[k])!=a for k,a in protocol['vectors'].items()):raise ValueError('frozen vectors changed')
    env=dict(python=platform.python_version(),numpy=np.__version__,warp=wp.__version__,pid=os.getpid(),threads=os.environ.get('OPENBLAS_NUM_THREADS'),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu','--format=csv'],text=True),system_free_bytes=os.statvfs('/').f_bavail*os.statvfs('/').f_frsize)
    save('environment.json',env)
    gpu=GPUOperator(s);print('GPU constructed',gpu.build_seconds,flush=True)
    states={k:s.restrict(a) for k,a in v.items() if k.startswith('state_')}
    dirs={k:s.restrict(v[k]) for k in ('direction_mixed','direction_local_only_direction')}
    comparisons={};derivatives={};integration={};timings=[]
    for name,q in states.items():
        print('state',name,flush=True);lin=gpu.prepare(q)
        for dn,d in dirs.items():
            key=name+'/'+dn;t=time.perf_counter();cpu=s.response(q,d,order=6);cpu_time=time.perf_counter()-t
            full=gpu.action(lin,d);g=dict(lin.response,full_tangent_action=full,tangent_action=s.restrict(full))
            comparisons[key]={k:metric(g[k],cpu[k]) for k in ('U','full_force','force','full_tangent_action')}
            np.savez(OUT/(name+'-'+dn+'-cpu.npz'),**cpu)
            timings.append(dict(state=name,direction=dn,cpu_seconds=cpu_time));print(key,cpu_time,flush=True)
            if name=='state_perturbation_3e-4_m':
                ref=s.response(q,d,order=7)
                # Parent material budgets are preserved verbatim, including no
                # new floor in the reference norm for acceptance.
                budgets=json.loads((ROOT/PARENT/'protocol.json').read_text())['material_reference']['budgets']
                integration[key]={}
                for k,bk in [('U','energy'),('full_force','force'),('full_tangent_action','tangent')]:
                    b=budgets[bk];integration[key][k]=metric(cpu[k],ref[k],b['rtol'],b['atol'],0.)
                np.savez(OUT/(name+'-'+dn+'-order7.npz'),**ref)
            actions=[]
            for h in protocol['fd_steps']:
                plus=gpu.evaluate(q+h*d);minus=gpu.evaluate(q-h*d)
                actions.append(dict(h=h,energy=metric((plus['U']-minus['U'])/(2*h),np.sum(cpu['force']*d),1e-4,1e-9,1e-3),
                     tangent=metric((plus['force']-minus['force'])/(2*h),cpu['tangent_action'],1e-3,1e-8,1.)))
            stable=sum(a['energy']['passed'] and a['tangent']['passed'] for a in actions)>=2
            derivatives[key]=dict(steps=actions,stable=stable)
        save('operator-comparison.json',comparisons);save('derivatives.json',derivatives);save('new-state-integration.json',integration);save('operator-timings.json',timings)
    # Layered real-space probes and Euclidean adjoints, including rigid and
    # constrained coefficients. All 144 local functions remain present.
    m=gpu.maps;points=[s.test_vectors[f'points{k}'] for k in range(3)];layout=m.layout(points);rng=np.random.default_rng(protocol['seed'])
    mappings={}
    for name,full in [('q0',s.expand(s.q0)),('mixed',s.direction_coefficients(dirs['direction_mixed'])),('local',s.direction_coefficients(dirs['direction_local_only_direction'])),('rigid',np.vstack((np.ones((s.n,3)),np.zeros((s.ndof-s.n,3)))))]:
        nodal=m.nodes(m.upload(full));cpu=s.nodes(full)
        mappings[name]=dict(nodes=metric(nodal.numpy().reshape(-1,3),cpu,1e-10,1e-10))
        dual=rng.normal(size=cpu.shape);lhs=np.sum(cpu*dual);rhs=np.sum(full*m.adjoint(m.upload(dual)).numpy().reshape(s.ndof,3))
        mappings[name]['nodal_adjoint']=metric(lhs,rhs,1e-10,1e-10)
    d=dirs['direction_mixed'];dx,dF=m.evaluate(d,points,True);fx=rng.normal(size=dx.shape);fF=rng.normal(size=dF.shape)
    pos=m.adjoint(m.sample(m.upload(fx),layout,transpose=True)).numpy().reshape(s.ndof,3)
    grad=m.gradient_adjoint(wp.array(fF.reshape(-1,3,3),dtype=wp.mat33d,device=gpu.device),layout,m.upload(np.ones(fF.size//9))).numpy().reshape(s.ndof,3)
    mappings['position_adjoint']=metric(np.sum(dx*fx),np.sum(d*s.restrict(pos)),1e-10,1e-10)
    mappings['gradient_adjoint']=metric(np.sum(dF*fF),np.sum(d*s.restrict(grad)),1e-10,1e-10)
    for name,d in dirs.items():
        a,b=m.evaluate(d,points,True);x,y=s.jvp(d,points);mappings[name+'-probes']=dict(position=metric(a,x),gradient=metric(b,y))
    save('mapping-audit.json',mappings)
    # Build the exact q0 Hessian by all 657 independent component directions.
    # Stored as a reusable search aid, never a replacement physical residual.
    lin=gpu.prepare(s.q0);n=s.q0.size;hessian=np.empty((n,n));begun=time.perf_counter()
    for j in range(n):
        d=np.zeros(n);d[j]=1.;hessian[:,j]=s.restrict(gpu.action(lin,d.reshape(s.q_shape))).ravel()
        if j%100==0:print('Hessian',j,n,time.perf_counter()-begun,flush=True)
        if time.perf_counter()-begun>protocol['resources']['max_solve_seconds']:raise TimeoutError('Hessian resource limit')
    build=time.perf_counter()-begun;c=curvature(hessian)
    np.savez(OUT/'search-matrix.npz',hessian=hessian,q0=s.q0)
    save('curvature.json',dict(archived=c,build_seconds=build,columns=n,state_sha256=array_digest(s.q0),physical_modification=False,
         certification='full q0 spectrum only; no SPD claim at other states'))
    records=[];b=-lin.response['force'].ravel()
    for name in protocol['preconditioners']:
        x,info=solve_linear(hessian,b,name=name,carrier_components=3*s.nfree_carrier,maxiter=protocol['resources']['max_linear'])
        # Independent exact device residual, not just the assembled matrix.
        true=float(np.linalg.norm(s.restrict(gpu.action(lin,x.reshape(s.q_shape))).ravel()-b))
        info.update(name=name,original_operator_residual=true,operator_matrix_build_seconds=build,total_including_matrix_build=build+info['total_seconds'])
        records.append(info)
    save('preconditioners.json',records)
    solves={};final={}
    for backend in ('cpu','gpu'):
        print('equilibrate',backend,flush=True)
        evaluate=(lambda q:s.response(q,order=6)) if backend=='cpu' else gpu.evaluate
        def checkpoint(q,trace,rejections):
            np.savez(OUT/(backend+'-checkpoint.npz'),q=q);save(backend+'-trajectory.json',dict(trace=trace,rejections=rejections))
        q,r,report=equilibrate(evaluate,s.q0,hessian,max_seconds=protocol['resources']['max_solve_seconds'],checkpoint=checkpoint)
        final[backend]=(q,r);solves[backend]=report
        np.savez(OUT/(backend+'-equilibrium.npz'),q=q,**r,**probes(s,q))
        save('static-solves.json',solves);print(backend,report['status'],report['trace'][-1],flush=True)
    qc,rc=final['cpu'];qg,rg=final['gpu'];pc,pg=probes(s,qc),probes(s,qg)
    equilibrium={k:metric(rg[k],rc[k],.005,1e-8,1e-3) for k in ('U','full_force')}
    equilibrium.update({k:metric(pg[k],pc[k],.005,1e-8,1e-3) for k in ('x','PK1')})
    equilibrium['q']=metric(qg,qc,.005,1e-10,1e-3)
    save('equilibrium-comparison.json',equilibrium)
    save('baseline-after.json',load_frozen_inputs(ROOT/PARENT,ROOT,expected_sha256=PIN)[3])
    save('resources.json',dict(host_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,gpu_static_map_bytes=m.static_bytes,material_points=gpu.count,build_seconds=gpu.build_seconds))
    checks=dict(operator=all(x['passed'] for row in comparisons.values() for x in row.values()),derivatives=all(x['stable'] for x in derivatives.values()),new_state_material=all(x['passed'] for row in integration.values() for x in row.values()),equilibrium=all(x['passed'] for x in equilibrium.values()) and all(x['converged'] for x in solves.values()))
    save('validation-status.json',dict(checks=checks,passed=all(checks.values()),dynamic_certified=False,continuum_spatial_certified=False,production_default_changed=False))
    print('DONE',checks,flush=True)

if __name__=='__main__':main()
