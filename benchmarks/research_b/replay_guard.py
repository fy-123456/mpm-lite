"""Check that finite-state guards leave frozen candidate responses unchanged."""
import argparse
from pathlib import Path
import numpy as np
from .run import ROOT, sha, write
from .scenes import states, source_rule, operator, directions
from engine.aniso_phase1.research_b.rules import compress
from engine.aniso_phase1.research_b.tensor import V22Space, TensorRule, TensorMaterialOperator


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    p=args.output;rows=[]
    with np.load(p/'small-raw.npz') as stored:
        for state in list(states('train'))+list(states('development')):
            op=operator(compress(source_rule(state['angle'],state['mixture']),8,'representatives'))
            for label,d in directions().items():
                result=op.evaluate(state['q'],d);result['weak_moments']=op.weak_moments(state['q'])
                prefix=f'representatives-8-{state["id"].replace("/","-")}-{label}-compact-'
                for key in ('U','force','tangent_action','weak_moments'):
                    difference=float(np.max(np.abs(result[key]-stored[prefix+key])))
                    rows.append(dict(case=state['id'],direction=label,quantity=key,max_absolute=difference))
    space=V22Space(ROOT);op=TensorMaterialOperator(space,TensorRule.uniform(space.edges,4))
    d=np.random.default_rng(220930).normal(size=space.q.shape);d/=np.linalg.norm(d)
    result=op.evaluate(space.q,d)
    with np.load(p/'v22-raw.npz') as stored:
        for key in ('U','force','tangent_action','weak_moments'):
            difference=float(np.max(np.abs(result[key]-stored['archive_hold-random-4-'+key])))
            rows.append(dict(case='v22/archive_hold',direction='random',quantity=key,max_absolute=difference))
    write(p/'guard-revalidation'/'finite-state-replay.json',dict(source_sha256=sha(Path(__file__)),
        rows=rows,bitwise_unchanged=all(r['max_absolute']==0 for r in rows),
        passed=all(r['max_absolute']<1e-12 for r in rows)))
    print('FINITE_STATE_REPLAY',max(r['max_absolute'] for r in rows),flush=True)


if __name__=='__main__':main()
