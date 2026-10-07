"""Common-time raw pressure/solid comparison; postprocessing only."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import read,write,serial_lock
from benchmarks.research_sequential_next.compare import metric,regions
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from engine.aniso_phase1.research_observable_pressure_next.rt0 import BoundedTopology

def history(p):return GenerationStore(p,read(p/'identity.json')).history()

def raw_reactions(run,h,cells):
    from .coupling_study import setup
    from engine.aniso_phase1.research_cost_phase_next.pressure import CellGeometry
    from engine.aniso_phase1.research_sequential_next.integrator import ValidatedAVF
    m,cfg=setup(run);geometry=CellGeometry(m,cells);values=[]
    for left,right in zip(h[:4],h[1:5]):
        q0,q1=left['state'],right['state'];dt=q1.time-q0.time;W=(q1.q-q0.q)/dt
        avf=ValidatedAVF(m,cfg,q0);force=avf.path(q0.q,W,dt)['force'];g=geometry.discrete(q0.q,q1.q)
        pbar=.5*(np.array(q0.child_states['fluid']['pressure_Pa'])+q1.child_states['fluid']['pressure_Pa'])
        force=force-.8*np.einsum('k,kij->ij',pbar,g);impulse=2*m.M@(W-q0.velocity)+dt*force
        raw=float(np.sum(impulse*m.boundary.unit)/dt)
        if np.linalg.norm(impulse[m.free])>1e-8:raise ValueError('reconstructed momentum residual inconsistent')
        values.append(dict(time_s=q1.time,dt_s=dt,reaction_N=raw))
    return values


def review(run):
    run=Path(run);a,b=[read(run/f'S3/fixed-{n}.json') for n in (2,4)];boundary={2:0.,4:0.};records=[]
    for x,y in zip(a['rows'],b['rows']):
        if abs(x['time']-y['time'])>1e-14:raise ValueError('grid comparison time mismatch')
        for n,row in ((2,x),(4,y)):
            top=BoundedTopology([[0,1],[0,1],[0,1]],n);boundary[n]+=row['dt']*float(top.boundary_sign@np.array(row['flux_interval_m3_s']))
        records.append(dict(time_s=x['time'],boundary_volume=metric(boundary[2],boundary[4],1e-10,.05)))
    fixed=read(run/'S3/grid-comparison.json');fixed_passed=fixed['status']=='passed_scoped' and all(x['boundary_volume']['passed'] for x in records)
    out=dict(status='passed_scoped' if fixed_passed else 'grid_sensitive',fixed_common_time=records,pressure_spatial_accuracy=False)
    four=(run/'S3/four-cell-coupled.json').exists();coupled_passed=False
    if four:
        h2=history(run/'cases/pressure-2');h4=history(run/'cases/pressure-4');x,y=h2[4],h4[4]
        if abs(x['state'].time-y['state'].time)>1e-14:raise ValueError('different coupled physical endpoints')
        with np.load(x['folder']/'frame.npz') as xx,np.load(y['folder']/'frame.npz') as yy:
            if not np.array_equal(xx['X'],yy['X']):raise ValueError('different physical probes')
            fields={}
            for region,w in regions(xx['X']).items():
                fields[region]={}
                for key,at in [('x',5e-5),('velocity',1e-4),('solid_PK1',.02),('total_PK1',.02)]:
                    av,bv=xx[key],yy[key]
                    if key=='x':av=av-xx['X'];bv=bv-yy['X']
                    fields[region][key]=metric(av,bv,at,.05,w)
        fluid={}
        for key in ('content_m3','cumulative_source_m3','cumulative_boundary_m3'):
            fluid[key]=metric(float(np.sum(x['state'].child_states['fluid'][key])),float(np.sum(y['state'].child_states['fluid'][key])),1e-10,.05)
        pressure=metric(np.array(x['state'].child_states['fluid']['pressure_Pa']),np.array(y['state'].child_states['fluid']['pressure_Pa']).reshape(2,2).mean(axis=1),.001,.05)
        ra,rb=raw_reactions(run,h2,2),raw_reactions(run,h4,4)
        reactions=[dict(time_s=x['time_s'],**metric(x['reaction_N'],y['reaction_N'],1e-4,.05)) for x,y in zip(ra,rb)]
        coupled_passed=all(x['passed'] for x in reactions) and all(z['passed'] for v in fields.values() for z in v.values()) and all(z['passed'] for z in fluid.values()) and pressure['passed']
        out['coupled']=dict(common_time_s=x['state'].time,steps=4,fields=fields,fluid=fluid,pressure=pressure,passed=coupled_passed,
            raw_reaction_two=ra,raw_reaction_four=rb,reaction_comparison=reactions,reaction_scope='reconstructed original interval momentum including pressure volume discrete-gradient force; fixed grips have zero endpoint correction; no new integration')
    write(run/'S3/common-time-grid-review.json',out)
    decision=read(run/'S3/pressure-scope-decision.json');decision.update(four_cell_fixed=fixed_passed,four_cell_coupled=four and coupled_passed,
        four_cell_result=read(run/'S3/four-cell-coupled.json') if four else None,common_time_review='S3/common-time-grid-review.json',scope='aligned reference cells only; four coupled steps compared at identical times')
    write(run/'S3/pressure-scope-decision.json',decision);print('PRESSURE_SCOPE',fixed_passed,coupled_passed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):review(a.run)
