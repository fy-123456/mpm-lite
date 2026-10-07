"""Four very small nonlinear midpoint steps; full formal q7 material quadrature."""
import argparse,time,resource
from pathlib import Path
import numpy as np
from .run import write
from engine.aniso_phase1.research_absolute_state_next.state import MaterialStateModel
from engine.aniso_phase1.research_absolute_state_next.dynamics import DryMidpoint
from engine.aniso_phase1.research_b.tensor import TensorMaterialOperator,TensorRule

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();out=a.run/'S2/nonlinear4';out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();m=MaterialStateModel();r=m.r;s=m.space;op=TensorMaterialOperator(s,TensorRule.uniform(s.edges,7));calls=0;cache={}
    def evaluate(q):
        global calls
        key=q.tobytes()
        if key not in cache:
            raw=op.evaluate(r.expand(q));raw['force']=r.P.T@raw['force'];calls+=1
            # Cache has a hard cap; state keys include ALL coordinates and boundary.
            if len(cache)>=3:cache.pop(next(iter(cache)))
            cache[key]=raw
        return cache[key]
    solver=DryMidpoint(m,evaluate);state=m.rest();X=np.stack(np.meshgrid(np.linspace(.125,.875,33),np.linspace(.375,.625,5),[.5],indexing='ij'),axis=-1).reshape(-1,3)
    frames=[X.copy()];Fs=[np.tile(np.eye(3),(len(X),1,1))];vs=[np.zeros_like(X)];metrics=[]
    unit=np.zeros(state.q.shape);unit[r.fixed[s.carrier_X[s.fixed_scalar_ids,0]>=.75],0]=1
    for i in range(4):
        boundary=(unit*.1*((i+1)*.001)**2).ravel()[solver.fixed]
        state,row=solver.step(state,.001,boundary);x,F,v=m.fields(state,X)
        frames.append(x);Fs.append(F);vs.append(v);metrics.append(row)
        write(out/'metrics.json',metrics);print('nonlinear step',i+1,'calls',calls,'J',row['minJ'],'residual',row['true_free_force_residual'],flush=True)
    m.save(state,out/'final.npz');np.savez_compressed(out/'frames.npz',X=X,x=frames,F=Fs,velocity=vs,time=np.arange(5)*.001)
    write(out/'summary.json',dict(steps=4,model='original formal material space, corrected SH contract separate',integrator='nonlinear midpoint; original full q7 potential; rest tangent chord',
        material_evaluations=calls,points_per_call=4917248,seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        minJ=min(x['minJ'] for x in metrics),max_u=float(np.linalg.norm(frames[-1]-X,axis=1).max()),
        gate_M_original_material_small_window=True,formal_updated_SH_qualified=False,pressure_coupling=False,production_kernel=False))
