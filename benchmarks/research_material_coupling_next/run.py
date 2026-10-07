import argparse,json,time,resource
from pathlib import Path
import numpy as np
from .common import MaterialStateModel,unit_boundary,write,BASE
from engine.aniso_phase1.research_material_coupling_next.budget import Budget
from engine.aniso_phase1.research_material_coupling_next.operators import Geometry
from engine.aniso_phase1.research_material_coupling_next.controller import Controller
from engine.aniso_phase1.research_material_coupling_next.driver import advance_series
from engine.aniso_phase1.research_formal_pressure_next.state import save,load

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--steps',type=int,choices=(1,2,4),default=4)
    p.add_argument('--half-dt',action='store_true');p.add_argument('--mode',choices=('full','compressed'),default='full');p.add_argument('--drained',action='store_true');p.add_argument('--alpha',type=float,default=1.);a=p.parse_args()
    choice=json.loads((a.run/'S3/selection.json').read_text())['selected'];h=choice['dt']/(2 if a.half_dt else 1);coef=choice['amplitude_coefficient']
    if a.mode=='compressed' and not json.loads((a.run/'S1/material-screen.json').read_text())['accepted']:raise ValueError('q5 screen not qualified')
    label=('dry' if a.alpha==0 else ('drained' if a.drained else 'closed'))+'-'+a.mode+('-half-dt' if a.half_dt else '')
    out=a.run/'S4'/label
    if (out/'summary.json').exists():raise ValueError('completed output is immutable')
    out.mkdir(parents=True,exist_ok=True);budget=Budget();m=MaterialStateModel();g=Geometry(m,budget);primary=7 if a.mode=='full' else 5
    c=Controller(m,g,budget,primary=primary,reserve=7,options=dict(drained=a.drained,alpha=a.alpha),journal=lambda data:write(out/'attempts.json',data))
    solver=c._solver(primary);state=solver.rest();unit,right=unit_boundary(m);target=lambda t:(unit*(coef*t*t)).ravel()[solver.fixed]
    X=np.stack(np.meshgrid(np.linspace(.125,.875,33),np.linspace(.375,.625,5),[.5],indexing='ij'),axis=-1).reshape(-1,3)
    previous=[state];frames=[X.copy()];Fs=[np.tile(np.eye(3),(len(X),1,1))];ps=[state.p.copy()];times=[0.];metrics=[]
    def accept(state,row):
        x,F,v=m.fields(state,X);row['reaction_right_x']=float(row['reaction_fixed'][np.isin(m.r.fixed,right),0].sum());row['max_probe_displacement']=float(np.linalg.norm(x-X,axis=1).max())
        raw=c._material(state.rule).evaluate(state.q);row['mean_PK1_by_slab']=raw['weak_moments'][:,0]/(np.diff(m.space.edges[0])[:,None,None]*.25**2)
        accepted_residual=c._solver(state.rule).residual(previous[0],state.q,state.p,h)
        row['free_only_tolerance_diagnostic']=1e-8+1e-3*accepted_residual['free_scale']
        row['free_residual_norm']=float(np.linalg.norm(accepted_residual['mech'][solver.free]))
        previous[0]=state
        metrics.append(row);frames.append(x);Fs.append(F);ps.append(state.p.copy());times.append(state.time)
        write(out/'metrics.json',metrics);save(state,out/'checkpoint.npz',c.identity)
        np.savez_compressed(out/'frames.npz',X=X,x=frames,F=Fs,p=ps,time=times)
        if state.step==1:save(state,out/'first-checkpoint.npz',c.identity)
        print(label,'step',state.step,'material calls',budget.counts['material'],'J',row['minJ'],'R',row['reaction_right_x'],'p',state.p,'jv',row['jv_calls_step'],flush=True)
    try:
        state,rows=advance_series(c.step,state,h,a.steps,target,accept)
        dry_check=None
        if a.alpha==0:
            from engine.aniso_phase1.research_absolute_state_next.dynamics import DryMidpoint
            ref=DryMidpoint(m,c._material(primary).evaluate)
            dry,_=advance_series(ref.step,m.rest(),h,a.steps,target)
            dry_check=dict(q_gap=float(abs(dry.q-state.q).max()),velocity_gap=float(abs(dry.velocity-state.velocity).max()),time=dry.time,steps=dry.step)
            assert dry_check['q_gap']<1e-8 and dry_check['velocity_gap']<1e-5
        restored=load(out/'checkpoint.npz',m,c.identity);assert restored.digest()==state.digest()
        x,F,v=m.fields(restored,X[:4]);np.savez_compressed(out/'replay-probes.npz',X=X[:4],x=x,F=F,v=v)
        write(out/'identity.json',c.identity);write(out/'summary.json',dict(label=label,steps=a.steps,dt=h,coefficient=coef,state_digest=state.digest(),rule=state.rule,
            minJ=min(t['minJ'] for t in rows),max_pressure=max(float(abs(t['pressure']).max()) for t in rows),
            jv_calls=sum(t['jv_calls_step'] for t in rows),fallbacks=sum(t['fallback'] for t in rows),dry_check=dry_check,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,budget=budget.report(),production=False))
    except Exception as e:
        write(out/'failure.json',dict(type=type(e).__name__,reason=str(e),accepted_steps=len(metrics),budget=budget.report()));raise
