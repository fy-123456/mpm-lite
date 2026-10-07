import argparse,json,time,resource
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_absolute_state_next.state import MaterialStateModel
from engine.aniso_phase1.research_formal_pressure_next.geometry import Geometry

def write(p,x):
    def default(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,default=default,indent=2)+'\n')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',type=Path,required=True);args=a.parse_args();start=time.perf_counter()
    m=MaterialStateModel();g=Geometry(m);q=m.rest().q
    rng=np.random.default_rng(24);w=np.zeros_like(q);w[m.r.free]=rng.normal(size=(len(m.r.free),3))*1e-6
    # Include a nonzero prescribed grip increment, not just free motion.
    b=np.zeros_like(q);b[m.r.fixed[m.space.carrier_X[m.space.fixed_scalar_ids,0]>=.75],0]=-1e-5
    q1=w+b
    z=g.evaluate(q);non=g.evaluate(q1,w);higher=Geometry(m,10).evaluate(q1);chain=g.chain(q,q1)
    eps=1e-3;plus=g.evaluate(q1+eps*w);minus=g.evaluate(q1-eps*w)
    dV=np.einsum('cni,ni->c',non['G'],w)
    dG=(plus['G']-minus['G'])/(2*eps)
    err=lambda a,b:float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))
    report=dict(reference_volume=z['V'],analytic_volume=g.reference_volume,volume_error=err(z['V'],g.reference_volume),
        order_volume_relative=err(non['V'],higher['V']),order_gradient_relative=err(non['G'],higher['G']),
        derivative_relative=err((plus['V']-minus['V'])/(2*eps),dV),hessian_relative=err(dG,non['dG']),
        chain_defect=chain['V1']-chain['V0']-np.einsum('cni,ni->c',chain['G'],q1),
        omitted_boundary_defect=chain['V1']-chain['V0']-np.einsum('cni,ni->c',chain['G'],w),
        geometry_points=g.points,seconds=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(args.run/'S1/geometry.json',report);print(json.dumps(report,default=lambda x:x.tolist(),indent=2))
    assert report['volume_error']<1e-8 and report['order_gradient_relative']<1e-5
    assert report['derivative_relative']<1e-3 and report['hessian_relative']<1e-3
    assert max(abs(report['chain_defect']))<1e-9
