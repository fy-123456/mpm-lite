"""Replay a complete rejected-and-retried coupled step from particle-free packages."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_constrained_history_next.model import SeparatedOperator,advance,compare

def run(path):
    m=json.loads((path/'manifest.json').read_text())
    for name,h in m.items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest()!=h:raise ValueError('changed retry package')
    a=SeparatedOperator.load_package(path/'primary.npz');r=SeparatedOperator.load_package(path/'reserve.npz')
    if not np.array_equal(a.M,r.M) or not np.array_equal(a.gD,r.gD):raise ValueError('retry geometry or mass mismatch')
    with np.load(path/'initial.npz') as s:v=s['v'];p=s['p'];h=float(s['h']);t=float(s['time'])
    zero=np.zeros_like(v);d,v1,p1,z,metrics=advance(a,zero,v,p,h,t)
    verdict=compare(a,r,d)
    if verdict['passed']:raise ValueError('diagnostic primary did not reject')
    d,v1,p1,z,metrics=advance(r,zero,v,p,h,t)
    np.savez_compressed(path/'replayed.npz',d=d,v=v1,p=p1,z=z)
    return dict(retried=True,primary_order=a.order,accepted_order=r.order,particle_arrays=0,
        pressure_geometry_points=len(r.gX),material_points=len(r.points),check=verdict,
        claim='full frozen coupled retry only, not particle Prepare/Load')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.checkpoint:
        from engine.aniso_phase1.research_constrained_history_next.checkpoint import load,save
        b=load(a.checkpoint);m=b.step();save(b,a.output)
        print(json.dumps(dict(retried=m['retried'],digest=b.state.digest(),rule_energy_jump=m['rule_energy_jump'])))
    else:print(json.dumps(run(a.package)))
