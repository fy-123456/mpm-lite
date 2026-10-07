"""Separate time error, grid error, conservation and physical observability."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import numpy as np
from .provenance import *
from .physics import baseline_model
from .coupling import protocol
from .grid_transfer import restriction
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_pressure_window_next.coupling_study import frame
from benchmarks.research_pressure_window_next.candidate_study import fields,good
from benchmarks.research_sequential_next.compare import metric,regions
from engine.aniso_phase1.research_post_release.fields import CachedProbes
from engine.aniso_phase1.research_observable_boundary_next.rt0 import CartesianTopology

CASES={'grid16-half':('coarse','observable-coarse-half'),'grid32-h':('fine','grid32-h'),'grid32-half':('fine','grid32-half')}

def folder(run,name):return (APP if name=='grid16-half' else Path(run))/'cases'/CASES[name][1]

def export(run):
    run=Path(run);verify(run);m,cfg=baseline_model(run,pressure=True);cache=CachedProbes(m);p=protocol()
    for name,(grid,case) in CASES.items():
        h=history(folder(run,name));c=SimpleNamespace(alpha=p['parameters']['alpha'],geometry=SimpleNamespace(topology=CartesianTopology(p['cuts'][grid])));out={k:[] for k in ('x','velocity','PK1','PK1_total','Cauchy_skeleton','Cauchy_total','cell_pressure_Pa','face_flux_m3_s')};obs=[]
        for i,item in enumerate(h):
            s=item['state'];f=frame(c,cache,s)
            for k in out:out[k].append(f[k])
            if i in np.linspace(0,len(h)-1,5,dtype=int):obs.append(dict(time_s=s.time,u_rms_m=float(np.linalg.norm(f['x']-f['X'])/np.sqrt(f['X'].size/3)),v_rms_m_s=float(np.linalg.norm(f['velocity'])/np.sqrt(f['X'].size/3))))
        np.savez_compressed(run/'S3'/f'{name}-probes.npz',X=f['X'],fiber=m.parent.params.fiber_direction,times=np.array([x['state'].time for x in h]),**{k:np.array(v) for k,v in out.items()})
        rows=h[-1]['rows'];first=h[0]['state'];last=h[-1]['state'];a=first.child_states['fluid'];b=last.child_states['fluid'];cap=p['parameters']['storage']*c.geometry.topology.V0
        E0=m.kinetic(first.velocity)+m.evaluate(first.q)['U']+.5*np.sum(cap*np.asarray(a['pressure_Pa'])**2);work=sum(abs(r['external_work_J'])+abs(r['source_work_J'])+abs(r['reservoir_work_J']) for r in rows);budget=1e-9+.01*max(E0,work,1e-8);balance=rows[-1]['total_energy_J']-E0+sum(r['darcy_dissipation_J']-r['external_work_J']-r['source_work_J']-r['reservoir_work_J'] for r in rows);mass=float(np.sum(np.asarray(b['content_m3'])-a['content_m3'])+b['cumulative_boundary_m3']-a['cumulative_boundary_m3']-np.sum(np.asarray(b['cumulative_source_m3'])-a['cumulative_source_m3']))
        observed=[i for i,x in enumerate(obs) if x['u_rms_m']>=1.5e-4 or x['v_rms_m_s']>=3e-4];observable=any(j-i>=2 for i in observed for j in observed)
        passed=all(r['min_detF']>.1 and r['true_scaled_residual']<=1 and r['darcy_dissipation_J']>=0 and max(abs(v) for v in r['pressure_work_defect_J'])<=1e-10 for r in rows) and abs(balance)<=budget and abs(mass)<=1e-10
        write(run/'S3'/f'{name}-physical.json',dict(status='passed_scoped' if passed and observable else 'limited',source=str(folder(run,name)),source_identity_sha256=sha(folder(run,name)/'identity.json'),source_final_state_sha256=sha(h[-1]['folder']/'state.json'),observations=obs,observable=observable,physical_passed=bool(passed),energy_balance_J=balance,energy_budget_J=budget,mass_defect_m3=mass,min_detF=min(r['min_detF'] for r in rows),numerical_probes_not_extra_frames=True));print('GRID_EXPORT',name,passed,observable,balance,mass,flush=True)

def arrays(run,name):
    with np.load(Path(run)/'S3'/f'{name}-probes.npz') as z:return {k:z[k] for k in z.files}

def compare(run):
    run=Path(run);p=protocol();top16=CartesianTopology(p['cuts']['coarse']);top32=CartesianTopology(p['cuts']['fine']);P,C,Z=restriction(top16,top32);out={}
    for label,left,right,stride in [('time','grid32-h','grid32-half',2),('grid','grid16-half','grid32-half',1)]:
        a=arrays(run,left);b=arrays(run,right);ha=history(folder(run,left));hb=history(folder(run,right));records=[]
        for i,x in enumerate(ha):
            j=i*stride;y=hb[j]
            if abs(x['state'].time-y['state'].time)>1e-15:raise ValueError('comparison time mismatch')
            fa=dict(X=a['X'],**{k:a[k][i] for k in ('x','velocity','PK1')});fb=dict(X=b['X'],**{k:b[k][j] for k in ('x','velocity','PK1')});fld=fields(fa,fb,a['fiber']);extra={}
            for region,w in regions(a['X']).items():
                for k in ('PK1_total','Cauchy_skeleton','Cauchy_total'):extra[region+'/'+k]=metric(a[k][i],b[k][j],.02,.05,w)
            pa=a['cell_pressure_Pa'][i];pb=b['cell_pressure_Pa'][j];extra['pressure']=metric(pa,P@pb if label=='grid' else pb,.001,.05,top16.V0 if label=='grid' else top32.V0)
            ca=x['state'].child_states['fluid'];cb=y['state'].child_states['fluid']
            for key in ('content_m3','cumulative_source_m3'):
                v=np.asarray(cb[key]);extra[key]=metric(ca[key],C@v if label=='grid' else v,1e-10,.05)
            extra['boundary_cumulative']=metric(ca['cumulative_boundary_m3'],cb['cumulative_boundary_m3'],1e-10,.05)
            if i:
                row=x['rows'][-1];fine=hb[-1]['rows'][stride*(i-1):stride*i];z=sum(np.asarray(v['flux_interval_m3_s'])*v['dt'] for v in fine)/row['dt'];rr=sum(v['reaction_N']*v['dt'] for v in fine)/row['dt'];zz=Z@z if label=='grid' else z;extra['face_flux_interval']=metric(ca['flux_interval_m3_s'],zz,1e-10,.05);extra['reaction_interval']=metric(row['reaction_N'],rr,1e-4,.05)
                if label=='grid':extra['boundary_flow_interval']=metric(float(top16.boundary_sign@np.asarray(ca['flux_interval_m3_s'])),float(top32.boundary_sign@z),1e-10,.05)
            records.append(dict(time_s=x['state'].time,fields=fld,extra=extra,passed=good(fld) and all(v['passed'] for v in extra.values())))
        maxima={k:max(r['extra'][k]['absolute']/r['extra'][k]['budget'] for r in records if k in r['extra']) for k in records[-1]['extra']}
        result=dict(status='passed_scoped' if all(r['passed'] for r in records) else 'limited',records=records,max_extra_budget_fractions=maxima,conservative_restriction=label=='grid',interval_flux_and_reaction=True);out[label]=result;write(run/'S3'/f'{label}-comparison.json',result)
    physical={name:read(run/'S3'/f'{name}-physical.json') for name in CASES};restart=read(run/'cases/grid32-h/summary.json')['actual_new_process_restart'];passed=all(x['status']=='passed_scoped' for x in [*out.values(),*physical.values()])
    write(run/'S3/grid-transfer-check.json',dict(status='passed_scoped',pressure_constant_error=float(np.max(abs(P@np.ones(32)-1))),content_total_error=float(np.max(abs(C.sum(axis=0)-1))),flux_divergence_error=float(np.max(abs(top16.B@Z-C@top32.B))),pressure_shape=list(P.shape),flux_shape=list(Z.shape)))
    write(run/'S3/coupling-scope-decision.json',dict(status='passed_scoped' if passed else 'limited',two_grid_short_window_passed=passed,complete_grids=2,time_passed=out['time']['status']=='passed_scoped',grid_passed=out['grid']['status']=='passed_scoped',actual_new_process_restart=restart,checkpoint_reload=True,actual_observable_coupling=all(x['observable'] for x in physical.values()),full_coupled_cycle=False,coupled_q5=False,production_C_E_integration=False,physically_calibrated=False,spatial_refinement_axes='x only'))
    print('GRID_COMPARISON',passed,'time',out['time']['status'],'grid',out['grid']['status'],out['grid']['max_extra_budget_fractions'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['export','compare']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
