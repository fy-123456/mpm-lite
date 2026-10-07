"""Supplemental self checks, with full error curves saved rather than rounded."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .run import write, sha, ROOT
from .scenes import source_rule, operator, states, directions, errors
from engine.aniso_phase1.research_b.rules import compress
from engine.aniso_phase1.tensor_metrics import quadrature_axis


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rows=[]; curves=[]
    ds=directions();start=time.perf_counter()
    for scene in list(states('train'))+list(states('development')):
        full=operator(source_rule(scene['angle'],scene['mixture'],order=4))
        finer=operator(source_rule(scene['angle'],scene['mixture'],order=5))
        for label,d in ds.items():
            a,b=full.evaluate(scene['q'],d),finer.evaluate(scene['q'],d)
            a['weak_moments'],b['weak_moments']=full.weak_moments(scene['q']),finer.weak_moments(scene['q'])
            rows.append(dict(scene=scene['id'],direction=label,errors=errors(a,b,scale=.1)))
        if scene['id'].endswith('/hold'):
            candidate=operator(compress(full.rule,8,'representatives'))
            q,d=scene['q'],ds['random'];base=candidate.evaluate(q,d)
            exact=float(np.sum(base['force']*d));exact_norm=float(np.linalg.norm(base['tangent_action']))
            for eps in (1e-3,1e-4,1e-5,1e-6,1e-7):
                plus,minus=candidate.evaluate(q+eps*d),candidate.evaluate(q-eps*d)
                de=(plus['U']-minus['U'])/(2*eps)-exact
                dt=float(np.linalg.norm((plus['force']-minus['force'])/(2*eps)-base['tangent_action']))
                curves.append(dict(scene=scene['id'],eps=eps,energy_absolute=abs(de),
                    energy_relative=abs(de)/max(abs(exact),1e-8),tangent_absolute=dt,
                    tangent_relative=dt/max(exact_norm,1e-4)))
    edge_path=ROOT/'docs/results/lite-aniso-mainline/v22/multiscale/space/q4/round6.npz'
    tensor=[]
    with np.load(edge_path) as z:
        edges=[z[f'axis{k}'] for k in range(3)]
    for order in (3,4,5,6,7):
        moments=[];maximum=0.;minimum=float('inf')
        for e in edges:
            axis_moments=[]
            for lo,hi in zip(e[:-1],e[1:]):
                x,w=quadrature_axis([lo,hi],order);minimum=min(minimum,float(w.min()))
                axis_moments.append([float(w@(x**k)) for k in range(3)])
                maximum=max(maximum,max(abs(axis_moments[-1][k]-(hi**(k+1)-lo**(k+1))/(k+1)) for k in range(3)))
            moments.append(np.sum(axis_moments,axis=0))
        tensor.append(dict(order=order,volume=float(np.prod([m[0] for m in moments])),
            positive_axis_weight_min=minimum,max_cell_axis_moment_absolute_error=maximum,
            samples=int(np.prod([len(e)-1 for e in edges])*order**3),
            coverage='one tensor product rule per material cell; no overlapping patch weights'))
    result=dict(source_sha256=sha(Path(__file__)),reference_rows=rows,derivative_curves=curves,
        tensor_moment_audit=tensor,reference_passed=all(x['passed'] for r in rows for x in r['errors'].values()),
        seconds=time.perf_counter()-start)
    write(args.output/'supplemental-diagnostics.json',result)
    print('SMALL_REFERENCE_CERTIFIED',result['reference_passed'],flush=True)


if __name__=='__main__':main()
