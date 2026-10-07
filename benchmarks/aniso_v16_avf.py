"""Additional v16 time-centered comparison; phase-one algorithms stay frozen."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time,traceback,zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from benchmarks.aniso_v16_experiments import OUT,ROOT,load,write,sha,sources,frame
from benchmarks.aniso_carrier_joint import load_case,cube,spectrum
from engine.aniso_phase1.carrier_avf import CarrierAVFSolver
from engine.aniso_phase1.carrier_joint import gradient
AVF=OUT/'avf'


def avf_sources():
    result=sources()
    for name in ('benchmarks/aniso_v16_avf.py','tests/test_aniso_carrier_avf.py'):
        result[name]=sha(ROOT/name)
    return result


def freeze():
    AVF.mkdir(exist_ok=False);old=load(OUT/'protocol.json')
    assert all(sha(ROOT/n)==h for n,h in old['source_sha256'].items())
    with zipfile.ZipFile(OUT/'phase1-executed-source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in old['source_sha256']:z.write(ROOT/name,name)
        z.writestr('SOURCE_SHA256.json',json.dumps(old['source_sha256'],indent=2)+'\n')
    cases={}
    for t,label in ((1.1,'early_hold'),(1.6,'late_hold')):
        for moving in (False,True):
            for level,dt in (('coarse',.001),('fine',.0005),('finest',.00025),('fourth',.000125)):
                cases[f'{label}-'+('moving' if moving else 'frozen')+f'-avf-{level}']=dict(start=t,moving=moving,dt=dt,duration=.05)
    write(AVF/'protocol.json',dict(cases=cases,source_sha256=avf_sources(),tests=['tests.test_aniso_carrier_avf'],device='cpu',cuda='not_run',
        phase1_source_unchanged=True,scope='Same snapshots and same elastic/kinetic model; change only joint time update from backward Euler to Gauss-2 average vector field.',
        energy='path-averaged force and exact tangent of quadrature incremental potential; no damping or offsets',default_changed=False))


def worker(name,out):
    p=load(AVF/'protocol.json');assert avf_sources()==p['source_sha256'];spec=p['cases'][name]
    dest=out/'avf'/name;dest.mkdir(parents=True,exist_ok=False)
    s,e,m,h,source=load_case(spec['start']);so=CarrierAVFSolver(s,e,m,h,moving=spec['moving']);frames=[frame(s,e)];started=time.monotonic()
    write(dest/'config.json',dict(**spec,source=source));status=dict(completed=False,steps=0)
    try:
        count=round(spec['duration']/spec['dt']);every=round(.005/spec['dt'])
        with (dest/'steps.jsonl').open('x',buffering=1) as log:
            for k in range(1,count+1):
                old=so.state.clone();r=so.step(spec['dt']);log.write(json.dumps(r,allow_nan=False)+'\n');status['steps']=k
                assert r['history_commit_max']<1e-10
                assert abs(r['kinetic_force_work_defect_J'])<5e-13,r
                assert abs(r['potential_quadrature_error_J'])<5e-13,r
                assert r['newton_residual']/spec['dt']<1e-7
                if k%every==0:frames.append(frame(so.state,e))
        g=so.geometry;np.savez_compressed(dest/'audit-terminal.npz',**frame(so.state,e),x_before=old.x,Y_before=old.Y,v_before=old.v,C_before=old.C,
            nodes=g.nodes,N=g.N,E=g.E,Q=g.Q,J=g.J,q=g.metric,mass=m,V=e.V,A=e.A,ids=e.ids,P=e.P,weights=e.weights,
            **{f'B{k}':b for k,b in enumerate(e.B)})
        status.update(completed=True,terminal_time=so.state.time)
    except Exception:status['error']=traceback.format_exc();print(status['error'],flush=True)
    np.savez_compressed(dest/'frames.npz',**{k:np.array([f[k] for f in frames]) for k in frames[0]})
    status['seconds']=time.monotonic()-started;write(dest/'status.json',status)
    assert avf_sources()==p['source_sha256'];return status['completed']


def run(jobs):
    p=load(AVF/'protocol.json');assert load(AVF/'tests.json')['required_checks_passed'];assert not (AVF/'batch.json').exists()
    scratch=Path(load(OUT/'storage-protocol.json')['scratch'])
    def launch(name):
        with (AVF/(name+'.log')).open('x') as log:
            r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_v16_avf','worker','--case',name,'--output',str(scratch)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        src=scratch/'avf'/name;dest=AVF/'cases'/name
        if src.exists():
            shutil.copytree(src,dest);files={str(f.relative_to(src)):sha(f) for f in src.rglob('*') if f.is_file()}
            assert all(sha(dest/n)==v for n,v in files.items())
        else:files={}
        record=dict(case=name,exit_code=r.returncode,files_verified=len(files));print(record,flush=True);return record
    with ThreadPoolExecutor(max_workers=jobs) as pool:records=list(pool.map(launch,p['cases']))
    result=dict(completed=all(r['exit_code']==0 for r in records),records=records);write(AVF/'batch.json',result);return result['completed']


def crossing():
    assert not (AVF/'moving-support.json').exists();v=np.array([.1,.025,-.02]);s,e,m,h=cube(velocity=v)
    so=CarrierAVFSolver(s,e,m,h,moving=True,clamped=False);records=[];gates=[]
    for k in range(180):
        r=so.step(.005);t=so.state.time
        r.update(x_error=float(np.max(abs(so.state.x-s.x-t*v))),Y_error=float(np.max(abs(so.state.Y-s.Y-t*v))),
                 F_error=float(np.max(abs(gradient(e.B,so.state.Y)-np.eye(3)))),v_error=float(np.max(abs(so.state.v-v))),C_error=float(np.max(abs(so.state.C))))
        records.append(r)
        if k+1 in (1,2,60,120,180):gates.append(dict(step=k+1,gate=spectrum(e.tangent(so.state.Y),6)))
    maxima={key:max(r[key] for r in records) for key in ('x_error','Y_error','F_error','v_error','C_error','momentum_change_norm')}
    passed=max(maxima.values())<1e-10 and all(g['gate']['passed'] for g in gates)
    write(AVF/'moving-support.json',dict(passed=passed,steps=180,max_errors=maxima,records=records,actual_massless_gates=gates))
    assert passed;print(maxima,flush=True)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('freeze','tests','run','worker','crossing'));p.add_argument('--jobs',type=int,default=6);p.add_argument('--case');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.action=='freeze':freeze();return
    if a.action=='tests':
        from benchmarks.aniso_mainline import tests
        ok=tests(AVF,load(AVF/'protocol.json'))
    elif a.action=='worker':ok=worker(a.case,a.output)
    elif a.action=='crossing':crossing();ok=True
    else:ok=run(a.jobs)
    raise SystemExit(0 if ok else 2)

if __name__=='__main__':main()
