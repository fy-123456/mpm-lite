import argparse,time,resource
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_absolute_state_next.state import MaterialStateModel
from engine.aniso_phase1.research_formal_pressure_next.geometry import Geometry
from engine.aniso_phase1.research_formal_pressure_next.material import Material
from engine.aniso_phase1.research_formal_pressure_next.solver import Solver
from engine.aniso_phase1.research_formal_pressure_next.state import save,load
from .geometry_check import write

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);a.add_argument('--drained',action='store_true');a.add_argument('--steps',type=int,default=4);a.add_argument('--alpha',type=float,default=1.);args=a.parse_args()
    if args.steps not in (1,4):raise ValueError('registered one or four steps only')
    case='dry' if args.alpha==0 else ('drained' if args.drained else 'closed')
    out=args.run/'S4'/case
    if (out/'summary.json').exists():raise ValueError('do not overwrite completed case')
    out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();m=MaterialStateModel();g=Geometry(m);op=Material(m,7);s=Solver(m,g,op,drained=args.drained,alpha=args.alpha);state=s.rest()
    op.deadline=start+600;op.limit=24
    unit=np.zeros_like(state.q);right=m.r.fixed[m.space.carrier_X[m.space.fixed_scalar_ids,0]>=.75];unit[right,0]=1
    X=np.stack(np.meshgrid(np.linspace(.125,.875,33),np.linspace(.375,.625,5),[.5],indexing='ij'),axis=-1).reshape(-1,3)
    xs=[X.copy()];Fs=[np.tile(np.eye(3),(len(X),1,1))];ps=[state.p.copy()];rows=[];times=[0.]
    try:
        for i in range(args.steps):
            target=(unit*(-.1*((i+1)*.0025)**2)).ravel()[s.fixed]
            state,row=s.step(state,.0025,target);x,F,v=m.fields(state,X)
            row['reaction_right_x']=float(row['reaction_fixed'][np.isin(m.r.fixed,right),0].sum())
            row['max_probe_displacement']=float(np.linalg.norm(x-X,axis=1).max())
            rows.append(row);xs.append(x);Fs.append(F);ps.append(state.p.copy());times.append(state.time)
            write(out/'metrics.json',rows);save(state,out/'checkpoint.npz',s.identity)
            np.savez_compressed(out/'frames.npz',X=X,x=xs,F=Fs,p=ps,time=times)
            print(case,'step',i+1,'calls',op.calls,'J',row['minJ'],'p',state.p,'error',row['energy_defect'],flush=True)
        if args.alpha==0:
            from engine.aniso_phase1.research_absolute_state_next.dynamics import DryMidpoint
            dry,metric=DryMidpoint(m,op.evaluate).step(m.rest(),.0025,target)
            gap_q=float(abs(state.q-dry.q).max());gap_v=float(abs(state.velocity-dry.velocity).max())
            assert gap_q<1e-9 and gap_v<1e-6 and abs(state.p).max()<1e-10
            write(out/'dry-reduction.json',dict(q_gap=gap_q,velocity_gap=gap_v,pressure_max=float(abs(state.p).max()),reference='unchanged absolute-state DryMidpoint same q7 operator'))
        restored=load(out/'checkpoint.npz',m,s.identity);assert restored.digest()==state.digest()
        stored=state.digest()
        # Identity and actual state transaction checks do not need another full step.
        try:load(out/'checkpoint.npz',m,{**s.identity,'alpha':99})
        except ValueError:pass
        else:raise AssertionError('foreign identity accepted')
        fields=m.fields(restored,X[:4]);np.savez_compressed(out/'replay-probes.npz',X=X[:4],x=fields[0],F=fields[1],v=fields[2])
        write(out/'identity.json',s.identity)
        write(out/'summary.json',dict(case=case,steps=args.steps,state_digest=stored,seconds=time.perf_counter()-start,
            material_calls=op.calls,material_seconds=op.seconds,jv_calls=s.jv_calls,minJ=min(t['minJ'] for t in rows),
            max_pressure=max(float(abs(t['pressure']).max()) for t in rows),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            formal_pressure_coupling=True,production=False,space_reference=False,geometry_points=g.points,material_points=4917248,
            rule=7,drained=args.drained,alpha=args.alpha,transport='constant reference hydraulic network',state_array_bytes=state.q.nbytes+state.velocity.nbytes+state.p.nbytes))
    except Exception as e:
        write(out/'failure.json',dict(type=type(e).__name__,reason=str(e),completed_steps=len(rows),seconds=time.perf_counter()-start,material_calls=op.calls))
        raise
