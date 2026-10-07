"""Read-only physical evidence from committed histories; never integrates."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import APP,read,write,sha,serial_lock
from benchmarks.research_sequential_next.checkpoint import GenerationStore

def history(folder):return GenerationStore(folder,read(Path(folder)/'identity.json')).history()

def pressure(run):
    run=Path(run);a=history(run/'cases/coupled-rt0-small2')[:5];b=history(APP/'cases/coupled-rt0-small2');errors={}
    if len(a)!=5 or len(b)!=5:raise ValueError('scaled/unscaled four-step evidence missing')
    for key in ('q','velocity','predictor'):
        errors[key]=max(float(np.max(abs(getattr(x['state'],key)-getattr(y['state'],key)))) for x,y in zip(a[1:],b[1:]))
    for key in ('pressure_Pa','flux_interval_m3_s','content_m3','cumulative_boundary_m3','cumulative_source_m3'):
        errors[key]=max(float(np.max(abs(np.asarray(x['state'].child_states['fluid'][key])-y['state'].child_states['fluid'][key]))) for x,y in zip(a,b))
    if max(errors.values())>1e-8:raise ValueError('scaling altered bounded original solution')
    write(run/'S4/original-equation-equivalence.json',dict(status='passed_scoped',steps=4,errors=errors,unscaled_case=str(APP/'cases/coupled-rt0-small2'),unscaled_identity_sha256=sha(APP/'cases/coupled-rt0-small2/identity.json'),new_integration_steps=0))
    print('PRESSURE_ORIGINAL_EQUIVALENCE',errors,flush=True)

def solid(run):
    run=Path(run);p=read(run/'S6/final-protocol.json');records={}
    for name in set((p['full_case'],p['default_case'])):
        folder=run/'cases'/name;h=history(folder);cfg=read(folder/'execution-protocol.json');rows=h[-1]['rows'];budget=cfg['acceptance']['energy_fraction']*cfg['acceptance']['energy_scale_J']
        def finite(value):
            if isinstance(value,dict):return all(finite(v) for v in value.values())
            if isinstance(value,list):return all(finite(v) for v in value)
            return bool(np.isfinite(value)) if isinstance(value,(float,int)) else True
        r=dict(steps=len(rows),all_finite=finite(rows),all_accepted=all(x['accepted'] for x in rows),max_true_residual_fraction=max(x['true_residual']/x['residual_tolerance'] for x in rows),
            boundary_max=max(max(x['displacement_constraint'],x['velocity_constraint']) for x in rows),min_detF=min(x['min_detF'] for x in rows),max_energy_balance_J=max(abs(x['energy_balance_J']) for x in rows),
            max_ledger_closure_J=max(abs(x['budget_defect_J']) for x in rows),path_budget_fraction=rows[-1]['cumulative_abs_path_quadrature_error_J']/budget,solve_work_budget_fraction=rows[-1]['cumulative_abs_solve_work_error_J']/budget)
        if not r['all_finite'] or not r['all_accepted'] or r['max_true_residual_fraction']>1 or r['boundary_max']>1e-8 or r['min_detF']<=.1 or max(r['path_budget_fraction'],r['solve_work_budget_fraction'])>1:raise ValueError('raw scene stability budget failed')
        records[name]=r
    write(run/'S6/raw-physical-check.json',dict(status='passed_scoped',cases=records,scope='all committed raw steps, inherited physical budgets, no extra integration'))
    print('RAW_SCENE_PHYSICS',records,flush=True)

def reference(run):
    from .provenance import COST_PARENT
    from engine.aniso_phase1.tensor_metrics import quadrature_axis,sampling
    from engine.aniso_phase1.tensor_reference import apply_axis
    from benchmarks.research_sequential_next.compare import metric
    run=Path(run);fields=[]
    for name,folder in [('R4',COST_PARENT/'P3/R4'),('R5',run/'S2/R5')]:
        with np.load(run/f'S2/material/{name}/state-fields.npz') as z:edges=tuple(z[f'edge{i}'].copy() for i in range(3));p=int(z['p'])
        with np.load(folder/'data.npz') as z:nodes=z['nodes'].copy()
        fields.append((edges,p,nodes))
    qs=[quadrature_axis(np.union1d(a,b),7) for a,b in zip(fields[0][0],fields[1][0])];totals={k:np.zeros(3) for k in ('global_domain','clamps','transition','interior')}
    for start in range(0,len(qs[0][0]),14):
        sl=slice(start,start+14);points=[qs[0][0][sl],qs[1][0],qs[2][0]];values=[]
        for edges,p,nodes in fields:
            u=nodes.reshape(*(p*(len(e)-1)+1 for e in edges),3)
            for i in range(3):u=apply_axis(sampling(edges[i],p,points[i]),u,i)
            values.append(u)
        w=qs[0][1][sl,None,None]*qs[1][1][None,:,None]*qs[2][1][None,None,:]
        data=np.stack((np.sum((values[0]-values[1])**2,axis=-1),np.sum(values[1]**2,axis=-1),np.ones_like(w)),axis=-1)*w[...,None]
        x=(points[0]-fields[0][0][0][0])/(fields[0][0][0][-1]-fields[0][0][0][0]);masks=dict(global_domain=np.ones(len(x),bool),clamps=(x<=.125)|(x>=.875),transition=((x>.125)&(x<=.25))|((x>=.75)&(x<.875)),interior=(x>.25)&(x<.75))
        for key,mask in masks.items():totals[key]+=data.sum(axis=(1,2))[mask].sum(axis=0)
    displacement={}
    for key,v in totals.items():
        error=float(np.sqrt(v[0]/v[2]));scale=float(np.sqrt(v[1]/v[2]));displacement[key]=dict(absolute_m=error,reference_norm_m=scale,passed=error<=5e-5+.05*scale)
    reaction=metric(read(COST_PARENT/'P3/R4/result.json')['solve']['reaction_N'],read(run/'S2/R5/result.json')['solve']['reaction_N'],1e-4,.05)
    if not all(v['passed'] for v in displacement.values()) or not reaction['passed']:raise ValueError('adjacent reference displacement/reaction failed')
    write(run/'S2/reference-displacement-reaction.json',dict(status='passed_scoped',displacement=displacement,reaction=reaction,common_cell_order=7,new_integration_steps=0))
    print('REFERENCE_DISPLACEMENT_REACTION',displacement['interior'],reaction,flush=True)

def pressure_configuration(run):
    from .coupling_study import coupled
    from engine.aniso_phase1.research_post_release.fields import CachedProbes
    run=Path(run);folder=run/'cases/coupled-rt0-small2';identity=read(folder/'identity.json');h=history(folder);saved=h[-1]['state']
    c,m,cfg=coupled(run,saved);oldcfg=read(folder/'execution-protocol.json')['config'];expected=__import__('copy').deepcopy(oldcfg);expected['implementation']['reuse_transpose_buffers']=False
    if cfg!=expected or type(m).__name__!='SegmentedModel' or cfg['implementation']['reuse_transpose_buffers'] or c.state.digest()!=saved.digest():raise ValueError('pressure config correction changes numerical input/state')
    cache=CachedProbes(m);newframe=c.frame(cache)
    with np.load(h[-1]['folder']/'frame.npz') as z:error=max(float(np.max(abs(newframe[k]-z[k]))) for k in z.files)
    if error>1e-8:raise ValueError('pressure config correction changes fields')
    fixture=Path(__file__).with_name('coupling_study.py');old_source=run/'S4/coupled-protocol-source/benchmarks/research_phase_reference_next/coupling_study.py'
    if sha(old_source)!=identity['fixture_source_sha256']:raise ValueError('old fixture snapshot unavailable')
    expected_source=old_source.read_text().replace("end=.05,space=choice['package'],mass_order=7,full_order=7)","end=.05,space=choice['package'],mass_order=7,full_order=7,reuse_transpose_buffers=False)")
    if fixture.read_text()!=expected_source:raise ValueError('fixture diff includes more than inactive config declaration')
    write(run/'S4/config-correction-check.json',dict(status='passed_scoped',old_fixture_sha256=sha(old_source),final_fixture_sha256=sha(fixture),
        sole_change='pressure fixture explicit reuse_transpose_buffers=False agrees with already-used SegmentedModel; neither TensorCellAVF nor ValidatedAVF reads this construction flag',
        final_model_type=type(m).__module__+'.'+type(m).__name__,complete_state_digest_unchanged=True,coupling_identity_equal=c.identity==identity['coupling'],field_max=error,extra_integrated_steps=0,old_case_identity_preserved=True))
    if c.identity!=identity['coupling']:raise ValueError('coupling identity changed')
    print('PRESSURE_CONFIG_CORRECTED',error,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['pressure','solid','reference','pressure_configuration']);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):globals()[a.phase](a.run)
