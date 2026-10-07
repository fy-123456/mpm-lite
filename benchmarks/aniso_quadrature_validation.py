"""Physical-domain quadrature reference and compression audit (static, no mass)."""
import argparse, gc, hashlib, json, platform, resource, subprocess, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import cg, LinearOperator
from engine.aniso_phase1.aligned_quadrature import (Rule, box_rule, face_rule, evaluate, assemble, LOW, HIGH, adaptive_rule)
from engine.aniso_phase1.trace_probe import BlendedMLS, ConstrainedStatic, CORNERS, face_points, grouped_maps
from engine.aniso_phase1.material_snapshot import MaterialSnapshot, center_support
from benchmarks.aniso_material_snapshot import make_snapshot
from benchmarks.aniso_boundary import reference
from benchmarks.aniso_rotation_dissipation import guard, table
from demos.aniso import DATA_ROOT
from utils.resource_guard import inspect_storage


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def fingerprint(*arrays):
    h = hashlib.sha256()
    for a in arrays:h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def setup(grid):
    s = make_snapshot(grid=grid);h = 1/(grid-1)
    snap = MaterialSnapshot(s.X, s.X, np.tile(np.eye(3), (len(s.X),1,1)), s.A, s.volume)
    nodes = np.unique(np.concatenate([k+CORNERS for k,_,_ in center_support(snap,h)]),axis=0)*h
    return snap, nodes, BlendedMLS(nodes,h)


def probes(nodes, Z, reference_u):
    x,y,z = (nodes-np.array([.25,.5,.5])).T
    fields = {}
    for name, components in dict(stretch=(x,0*x,0*x), shear=(y,0*x,0*x),
            bend_y=(-2*x*y,x*x,0*x), bend_z=(-2*x*z,0*x,x*x),
            twist=(0*x,-x*z,x*y)).items():
        p = np.column_stack(components).ravel();p = Z@(Z.T@p)
        fields[name] = p/np.linalg.norm(p)
    fields['reference_solution'] = reference_u/np.linalg.norm(reference_u)
    rng = np.random.default_rng(902)
    for i in range(3):
        p=Z@rng.normal(size=Z.shape[1]);fields[f'random_{i}']=p/np.linalg.norm(p)
    return fields


def run_grid(grid, out, names):
    guard();snap,nodes,blend = setup(grid);h=blend.h;n=len(nodes)
    tip,area = face_rule(h,order=4);L=np.asarray(area@evaluate(blend,tip)[0]).ravel()/area.sum()
    tip5,a5=face_rule(h,order=5);L5=np.asarray(a5@evaluate(blend,tip5)[0]).ravel()/a5.sum()
    f=np.zeros((n,3));f[:,1]=-1e-4*L;f=f.ravel()
    C=evaluate(blend,face_points(grid,.25,6,True))[0]
    checks=face_points(grid,.25,7,True);check=evaluate(blend,checks)[0]
    readers=evaluate(blend,snap.X)[0];q1,q1u=reference(grid,snap.X)
    matrices={};stats={};rules={};records=[]
    for name in names:
        guard();start=time.perf_counter();extra={}
        if name.startswith('adaptive'):
            from scipy.linalg import null_space
            Z=np.kron(null_space(C.toarray(),rcond=1e-10),np.eye(3))
            train=probes(nodes,Z,Z[:,0]);selected=[train[k] for k in ('stretch','shear','bend_y','bend_z','twist','random_0')]
            rule,extra=adaptive_rule(blend,int(name[8:]),selected,guard=guard)
        elif name.startswith('gauss'):
            rule=box_rule(h,int(name[5:]))
        elif name=='group4x8':
            x,V,_,_=grouped_maps(snap,nodes,h);rule=Rule(x,V,np.full(len(V),-1))
        elif name.startswith('particle'):
            s=make_snapshot(grid=grid,ppc_axis=int(name[8:]));rule=Rule(s.X,s.volume,np.full(len(s.X),-1))
        else:raise ValueError(name)
        rule_time=time.perf_counter()-start;start=time.perf_counter()
        K,support=assemble(blend,rule,guard=guard)
        matrices[name]=K;rules[name]=rule
        stats[name]=dict(samples=len(rule.weights),rule_build_seconds=rule_time,assembly_seconds=time.perf_counter()-start,
            volume=float(rule.weights.sum()),weight_min=float(rule.weights.min()),
            outside_points=int(np.count_nonzero(np.any((rule.points<LOW)|(rule.points>HIGH),axis=1))),
            rule_hash=fingerprint(rule.points,rule.weights),matrix_bytes=int(K.data.nbytes+K.indices.nbytes+K.indptr.nbytes),**support,**extra)
        print('ASSEMBLED',grid,name,json.dumps(stats[name]),flush=True)
    Kref=matrices['gauss4'];start=time.perf_counter();ref=ConstrainedStatic(Kref,C);uref,_=ref.solve(f)
    factor_time=time.perf_counter()-start;fields=probes(nodes,ref.Z,uref)
    convergence={}
    if 'gauss5' in matrices:
        D=matrices['gauss5']-Kref
        convergence['matrix_relative']=float(sparse.linalg.norm(D)/sparse.linalg.norm(Kref))
        convergence['probe_action_max']=max(float(np.linalg.norm(D@p)/np.linalg.norm(Kref@p)) for p in fields.values())
    C8=evaluate(blend,face_points(grid,.25,8,True))[0];ref8=ConstrainedStatic(Kref,C8);u8,_=ref8.solve(f)
    convergence.update(load_relative=float(np.linalg.norm(L-L5)/np.linalg.norm(L)),constraint_rank6=ref.rank,
        constraint_rank8=ref8.rank,clamp_density_solution_relative=float(np.linalg.norm(u8-uref)/np.linalg.norm(uref)))
    del ref8
    ref_tip=float(uref@f/-1e-4)
    for name,K in matrices.items():
        guard();start=time.perf_counter();A=ref.Z.T@(K@ref.Z);asym=float(np.linalg.norm(A-A.T)/np.linalg.norm(A));A=(A+A.T)/2
        low,vec=eigh(A,ref.A,subset_by_index=[0,0]);high=eigh(A,ref.A,subset_by_index=[len(A)-1,len(A)-1],eigvals_only=True)
        pmin=ref.Z@vec[:,0];pmin/=np.linalg.norm(pmin)
        record=dict(name=f'g{grid}-{name}',rule=name,grid=grid,nodes=n,particles=(len(rules[name].weights) if name.startswith('particle') else len(snap.X)),
            **stats[name],symmetry_relative=asym,rho_min=float(low[0]),rho_max=float(high[0]),
            analysis_seconds=time.perf_counter()-start,reference_factor_seconds=factor_time)
        probe_values={}
        for label,p in {**fields,'worst_mode':pmin}.items():
            kr=Kref@p;k=K@p;fr=ref.Z.T@kr
            probe_values[label]=dict(energy_relative=float((p@k)/(p@kr)-1),
                free_force_relative=float(np.linalg.norm(ref.Z.T@(k-kr))/max(np.linalg.norm(fr),1e-20)),
                action_relative=float(np.linalg.norm(k-kr)/max(np.linalg.norm(kr),1e-20)))
        # Preserve an actual failing mode, independently sampled in physical space.
        np.savez_compressed(out/(record['name']+'-mode.npz'),reference=snap.X,
            displacement=readers@pmin.reshape(-1,3),node_displacement=pmin.reshape(-1,3))
        if low[0]<=1e-8:
            record.update(status='failed_nonpositive_or_missing_mode',tip_displacement=None)
        else:
            start=time.perf_counter();from scipy.linalg import cho_factor,cho_solve
            factor=cho_factor(A);u=ref.Z@cho_solve(factor,ref.Z.T@f);record['solve_seconds']=time.perf_counter()-start
            R=K@u-f;tip_u=float(u@f/-1e-4);energy=float(.5*u@K@u)
            kr=Kref@u;record['own_solution_energy_relative']=float(u@K@u/(u@kr)-1)
            record['own_solution_reference_residual']=float(np.linalg.norm(ref.Z.T@(kr-f))/np.linalg.norm(ref.Z.T@f))
            record.update(status='screened',tip_displacement=tip_u,tip_relative_to_dense=tip_u/ref_tip-1,
                relative_to_q1=tip_u/q1['tip_displacement']-1,energy=energy,effective_stiffness=-1e-4/tip_u,
                force_for_mean_displacement=-1e-4*(-5e-4/tip_u),reaction=R.reshape(-1,3).sum(0).tolist(),
                clamp_max=float(np.max(np.linalg.norm(check@u.reshape(-1,3),axis=1))),
                free_residual=float(np.linalg.norm(ref.Z.T@R)/np.linalg.norm(ref.Z.T@f)),
                work_identity_relative=float(abs(2*energy-f@u)/abs(f@u)))
            # Same Jacobi preconditioner definition, same true residual threshold.
            iterations=[0];start=time.perf_counter()
            y,info=cg(A,ref.Z.T@f,rtol=1e-7,atol=0,maxiter=5000,
                M=LinearOperator(A.shape,matvec=lambda x:x/np.diag(A)),callback=lambda _:iterations.__setitem__(0,iterations[0]+1))
            record.update(pcg_iterations=iterations[0],pcg_info=int(info),pcg_seconds=time.perf_counter()-start,
                pcg_true_residual=float(np.linalg.norm(A@y-ref.Z.T@f)/np.linalg.norm(ref.Z.T@f)))
            np.savez_compressed(out/(record['name']+'.npz'),reference=snap.X,nodes=nodes,node_displacement=u.reshape(-1,3),
                displacement_mls=readers@u.reshape(-1,3),reference_displacement=q1u,
                dense_displacement=readers@uref.reshape(-1,3),clamp_points=checks,clamp_displacement=check@u.reshape(-1,3))
        save(out/(record['name']+'-probes.json'),probe_values);records.append(record)
        print('RESULT',json.dumps(record),flush=True)
        save(out/f'grid{grid}.json',dict(grid=grid,records=records,convergence=convergence,reference=q1))
    result=dict(grid=grid,records=records,convergence=convergence,reference=q1,
        configuration_hash=fingerprint(nodes,C.toarray(),f,snap.X,snap.volume),
        reference_min_eigenvalue=float(ref.eigen[0]),reference_max_eigenvalue=float(ref.eigen[-1]),
        reference_tip=ref_tip,reference_relative_to_q1=ref_tip/q1['tip_displacement']-1,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    save(out/f'grid{grid}.json',result);table(out/f'grid{grid}.csv',records)
    return result


def plot(out,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for result in results:
        rows=[r for r in result['records'] if r['tip_displacement'] is not None]
        for r in rows:
            label=f"g{r['grid']} {r['rule']}"
            axes[0].scatter(r['samples'],max(abs(r['tip_relative_to_dense']),1e-12),label=label)
            axes[1].scatter(r['assembly_seconds'],max(abs(r['tip_relative_to_dense']),1e-12))
            axes[2].plot([r['rho_min'],r['rho_max']],[label,label],'o-')
    axes[0].set(xlabel='Material samples',ylabel='Tip error vs same-space dense reference',yscale='log',xscale='log')
    axes[1].set(xlabel='CPU map + stiffness assembly (s)',ylabel='Tip error vs dense',yscale='log',xscale='log')
    axes[2].set(xlabel='Extreme stiffness ratio',xscale='log');axes[2].axvline(1,color='gray',ls='--')
    for ax in axes:ax.grid(alpha=.3)
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'summary.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grids',type=int,nargs='+',default=[17,33])
    p.add_argument('--rules',nargs='+',default=['gauss4','gauss5','gauss3','gauss2','group4x8','particle2','particle4'])
    p.add_argument('--out',type=Path,default=Path('docs/results/quadrature-validation/aligned-v1'));args=p.parse_args()
    if 'gauss4' not in args.rules:p.error('gauss4 reference required')
    if any(g not in (17,33) for g in args.grids):p.error('controlled grids are 17 and 33')
    guard();args.out.mkdir(parents=True,exist_ok=True)
    environment=dict(storage=vars(inspect_storage(DATA_ROOT)),python=platform.python_version(),static_device='cpu',
        head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],text=True),production_defaults_changed=False)
    source={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for folder in ('engine/aniso_phase1','benchmarks','tests') for x in Path(folder).glob('*.py')}
    environment['source_hashes']=source;save(args.out/'environment.json',environment);results=[]
    for grid in args.grids:
        results.append(run_grid(grid,args.out,args.rules));save(args.out/'results.json',results);plot(args.out,results);gc.collect()


if __name__=='__main__':main()
