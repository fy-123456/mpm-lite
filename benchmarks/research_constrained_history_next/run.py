"""Short physical cycle plus independently replayable material retry packages."""
import argparse,json,time,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_constrained_history_next.model import BoundedBridge,advance,compare
from engine.aniso_phase1.research_constrained_history_next.checkpoint import save

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cycle(run):
    b=BoundedBridge();out=run/'S3/cycle24';out.mkdir(parents=True,exist_ok=True)
    frames={k:[] for k in ('x','F','qvel','time','pressure')};metrics=[]
    def frame():
        s=b.state
        for k,v in dict(x=s.particles.x,F=s.particles.F,qvel=s.qv,time=s.time,pressure=s.p).items():frames[k].append(np.copy(v))
    frame();tick=time.perf_counter()
    for i in range(24):
        m=b.step(.0025);metrics.append(m);frame()
        if i==3:save(b,out/'step4.npz')
        print('cycle',i+1,'J',m['min_particle_detF'],'rule',m['material_order'],flush=True)
    save(b,out/'final.npz');np.savez_compressed(out/'frames.npz',X=b.state.particles.X,**{k:np.array(v) for k,v in frames.items()})
    write(out/'metrics.json',metrics)
    write(out/'summary.json',dict(steps=24,seconds=time.perf_counter()-tick,final_time=b.state.time,
        fallback_count=b.state.fallback_count,total_history_points=metrics[-1]['total_history_points'],
        min_J=min(m['min_particle_detF'] for m in metrics),max_u=max(m['max_particle_displacement'] for m in metrics),
        max_mass_defect=max(m['actual_content_defect'] for m in metrics),darcy=b.state.dissipation,
        projection_work=b.state.projection_work,rule_work=b.state.rule_work,
        scope='small common space only; no formal144 dynamic adoption'))

def retry_package(run):
    out=run/'S3/retry';out.mkdir(parents=True,exist_ok=True)
    b=BoundedBridge(primary_order=2);save(b,out/'before.npz');a,r,v,fit=b.prepare_pair()
    a.save(out/'primary.npz');r.save(out/'reserve.npz')
    np.savez_compressed(out/'initial.npz',v=v,p=b.state.p,time=b.state.time,h=.0025)
    metadata={n:sha(out/n) for n in ('primary.npz','reserve.npz','initial.npz')}
    write(out/'manifest.json',metadata)
    m=b.step();save(b,out/'after.npz');write(out/'metrics.json',m)
    result=subprocess.run([sys.executable,'-B','-m','benchmarks.research_constrained_history_next.replay','--package',str(out)],capture_output=True,text=True,check=True)
    write(out/'subprocess.json',json.loads(result.stdout))
    with np.load(out/'replayed.npz') as d:
        for k,v in dict(d=b.state.last_increment,p=b.state.p).items():np.testing.assert_array_equal(d[k],v)
    # Re-arm only in a diagnostic branch after the reserve and low histories evolved.
    b.state.active_order=2;save(b,out/'evolved-before.npz');m=b.step();write(out/'evolved-retry.json',m)
    if not m['retried'] or m['rule_energy_jump']==0:raise AssertionError('evolved retry did not test rule work')
    result=subprocess.run([sys.executable,'-B','-m','benchmarks.research_constrained_history_next.replay','--checkpoint',str(out/'evolved-before.npz'),'--output',str(out/'evolved-replayed.npz')],capture_output=True,text=True,check=True)
    restored=json.loads(result.stdout);write(out/'evolved-subprocess.json',restored)
    if restored['digest']!=b.state.digest():raise AssertionError('fresh process evolved retry differs')

def scaling(run):
    rows=[]
    for ppc in (2,3,4,8):
        b=BoundedBridge(ppc=ppc);a,r,v,fit=b.prepare_pair();seen=set();total=0
        for op in (a,r):
            for val in vars(op).values():
                if isinstance(val,np.ndarray) and id(val) not in seen:total+=val.nbytes;seen.add(id(val))
        tick=time.perf_counter();compare(a,r,np.zeros_like(v));elapsed=time.perf_counter()-tick
        rows.append(dict(particles=len(b.state.particles.X),primary_points=len(a.points),reserve_points=len(r.points),
            geometry_points=len(a.gX),free_vectors=v.size,operator_top_level_array_bytes=total,
            history_array_bytes=sum(getattr(b.state,k).nbytes for k in ('qx','qF','qv','reserve_x','reserve_F')),
            prepare_seconds=fit['seconds'],pair_material_check_seconds=elapsed,inner_particle_reads=0))
    write(run/'S6/scaling.json',dict(rows=rows,scope='fixed budgets; owned top-level arrays, not process RSS; Prepare and Load still depend on particles'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    cycle(a.run);retry_package(a.run);scaling(a.run)
