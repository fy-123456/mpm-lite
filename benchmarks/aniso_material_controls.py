"""Frozen/moving geometry x isolated/coupled velocity-dissipation controls.

Fixed-position coupled cases are constrained numerical counterfactuals, not
physical loading trajectories. Only moving/coupled cases are real trajectories.
"""
import hashlib,json,os,sys,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from demos.aniso import Config,Scene
from engine.aniso_phase1.unresolved_velocity import VelocityFilter
from engine.aniso_phase1.diagnostics import particle_kinetic
from benchmarks.aniso_material_history import ROOT,BASE,LEVELS,load,write
from benchmarks.aniso_dynamic_space import stress
OUT=BASE/'v14-controls';SOURCE=BASE/'v13/cases/baseline-fourth/audit-02000.npz'
GROUPS=[(kind,geometry,mode,order) for kind in ('isolated','coupled') for geometry in ('fixed','moving') for mode,order in [('none','post'),('null','post'),('weak','post'),('weak','pre'),('weak','symmetric')]]

def name(group,level):return '-'.join(group)+'-'+level

def worker(case,p):
    cfg=p['cases'][case];kind,geometry,mode,order=cfg['group'];dt=cfg['dt'];steps=round(p['duration']/dt)
    with np.load(SOURCE) as f:z={k:f[k].copy() for k in f.files}
    x0=z['particle_x_after'];v=z['particle_velocity_after'].copy();C=z['particle_C_after'].copy();m=z['particle_mass'];x=x0.copy();h=.125;vg=(z['particle_x_after']-z['particle_x_before'])/.000125
    scene=None;stats=[];frames=[];times=[];reactions=[];total_loss=0.;start=time.monotonic()
    if kind=='coupled':
        old=load(BASE/'v13/protocol.json')['configs']['baseline-fourth'];old.update(dt=dt,flip_ratio=.9**(dt/.001),velocity_dissipation='none')
        scene=Scene(Config(**old),'cpu');s=scene.solver
        for a,b in [('x','x'),('F','F'),('v','velocity'),('C','C')]:getattr(s,'ptc_'+a).assign(z['particle_'+b+'_after'])
        s.ptc_reference_x.assign(z['particle_reference_x']);s.sim_time=.25
    def damp(factor):
        nonlocal v,C,total_loss
        if mode=='none':return
        f=VelocityFilter(x,m,h,dt*factor,np.sqrt(10)/h,mode)
        v,C,d=f.apply(v,C);stats.append(d);total_loss+=d['dissipation_delta']
    def put():
        if scene is not None:s.ptc_v.assign(v);s.ptc_C.assign(C)
    def frame(t):
        frames.append(s.ptc_F.numpy().copy() if scene is not None else np.concatenate([v,C.reshape(len(C),9)],axis=1));times.append(t)
    frame(0.)
    for step in range(steps):
        if order in ('pre','symmetric'):damp(1. if order=='pre' else .5);put()
        if scene is not None:
            assert scene.step(),s.last_step_stats;x=s.ptc_x.numpy().copy();v=s.ptc_v.numpy().copy();C=s.ptc_C.numpy().copy()
            if geometry=='fixed':x=x0.copy();s.ptc_x.assign(x)
            reactions.append(scene.loading_rows[-1]['right_force'])
        else:x=x0+(step+1)*dt*vg if geometry=='moving' else x0.copy()
        if order in ('post','symmetric'):damp(1. if order=='post' else .5);put()
        if (step+1)%round(.01/dt)==0:frame((step+1)*dt)
    dest=OUT/case;dest.mkdir(exist_ok=False)
    np.savez_compressed(dest/'frames.npz',time=times,field=frames,x=x,v=v,C=C,reaction=reactions)
    r=dict(completed=True,steps=steps,wall_seconds=time.monotonic()-start,kinetic_J=sum(particle_kinetic(x,v,C,m,h)),filter_loss_J=total_loss,
        max_filter_identity=max((d['dissipation_identity_error'] for d in stats),default=0.),max_filter_momentum=max((d['momentum_change_norm'] for d in stats),default=0.))
    assert r['max_filter_identity']<1e-14 and r['max_filter_momentum']<1e-12
    write(dest/'summary.json',r)


def analyze(p):
    groups={};endfields={}
    for group in GROUPS:
        label='-'.join(group);fields={};times=None
        for level in LEVELS:
            case=name(group,level);assert load(OUT/case/'summary.json')['completed']
            with np.load(OUT/case/'frames.npz') as z:
                a=z['field'].copy();times=z['time'].copy();endfields[case]=a[-1]
            if group[0]=='coupled':
                cfg=load(BASE/'v13/protocol.json')['configs']['baseline-fourth'];a=stress(a.reshape(-1,3,3),cfg).reshape(a.shape)
            fields[level]=a
        norm=lambda a:float(np.sqrt(np.mean(a*a)))
        pairs=[]
        for a,b in zip(list(LEVELS)[:-1],list(LEVELS)[1:]):pairs.append(dict(pair=[a,b],absolute=norm(fields[a]-fields[b]),relative=norm(fields[a]-fields[b])/max(norm(fields[b]),1e-30)))
        groups[label]=dict(pairs=pairs,field='PK1 stress' if group[0]=='coupled' else 'concatenated v/C (different units, algebraic equality screen only)',final_pair=pairs[-1]['relative'])
    write(OUT/'summary.json',dict(completed=True,cases=len(p['cases']),steps=sum(load(OUT/case/'summary.json')['steps'] for case in p['cases']),groups=groups,
        fixed_geometry_scope='diagnostic frozen-position operator; positions externally reset, not a physical energy or trajectory accuracy certificate',observed_differences_are_not_true_solution_error=True))
    print({g:r['final_pair'] for g,r in groups.items()})


def main():
    if len(sys.argv)>1:
        p=load(OUT/'protocol.json');worker(sys.argv[1],p);return
    OUT.mkdir(exist_ok=False);cases={name(g,l):dict(group=g,dt=dt) for g in GROUPS for l,dt in LEVELS.items()}
    write(OUT/'protocol.json',dict(cases=cases,duration=.04,start_time=.25,source=str(SOURCE.relative_to(ROOT)),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),script_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),intent='separate geometry-dependent filter and operator splitting from physical local-motion damping; no tuning'))
    p=load(OUT/'protocol.json')
    def launch(case):
        with (OUT/(case+'.log')).open('x') as f:r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_material_controls',case],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        return dict(case=case,exit_code=r.returncode)
    with ThreadPoolExecutor(max_workers=8) as pool:r=list(pool.map(launch,cases))
    write(OUT/'batch.json',r);assert all(x['exit_code']==0 for x in r),[x for x in r if x['exit_code']];analyze(p)

if __name__=='__main__':main()
