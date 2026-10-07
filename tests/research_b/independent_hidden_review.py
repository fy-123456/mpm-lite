"""One-shot B7 independent hidden review; existing candidate code stays read-only.

Freeze writes the complete input definition before any material evaluation.
Run refuses to overwrite any prior result or use modified frozen sources.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from benchmarks.research_b.scenes import source_rule, operator, errors
from engine.aniso_phase1.research_b.rules import MaterialRule, compress, moment_audit
from engine.aniso_phase1.research_b.operator import MaterialSession

ROOT = Path(__file__).resolve().parents[2]
METRICS = ('U', 'force', 'weak_moments', 'tangent_action')
FAMILIES = [
    dict(name='hidden_axial17', angle=17., mixture=False, spatial_turn=False,
         q=[[.067,.019,0.],[0.,-.0134,0.],[0.,0.,-.01005],[.05,.045,0.],
            [0.,0.,0.],[0.,0.,0.],[-.045,0.,0.],[0.,0.,0.],[0.,0.,0.]]),
    dict(name='hidden_mixed73', angle=73., mixture=True, spatial_turn=False,
         q=[[.047,-.026,0.],[.013,-.0094,0.],[0.,0.,-.00705],[0.,-.135,0.],
            [0.,0.,0.],[0.,0.,0.],[.135,0.,0.],[0.,0.,0.],[0.,0.,.06]]),
    dict(name='hidden_turning112', angle=112., mixture=False, spatial_turn=True,
         q=[[-.018,-.065,.009],[.021,.011,0.],[0.,.015,.004],[0.,.19,0.],
            [0.,0.,.055],[0.,0.,0.],[-.19,0.,0.],[.06,0.,-.04],[0.,-.025,0.]])
]
STAGES = [('load', .62), ('hold', 1.), ('unload', .27), ('end_hold', 0.)]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    with open(path, 'x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False, ensure_ascii=False,
                  default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
        stream.write('\n')


def source_hashes():
    return {str(path.relative_to(ROOT)): sha(path) for pattern in
            ('engine/aniso_phase1/research_b/*.py', 'benchmarks/research_b/*.py')
            for path in sorted(ROOT.glob(pattern))}


def verify_frozen(output, kind):
    protocol = json.loads((output/f'hidden-{kind}-protocol.json').read_text())
    if sha(Path(__file__)) != protocol['review_entry_sha256']:
        raise RuntimeError('independent hidden entry changed after input freeze')
    if source_hashes() != protocol['source_sha256']:
        raise RuntimeError('candidate implementation changed after hidden input freeze')
    if sha(output/protocol['candidate_freeze_file']) != protocol['candidate_freeze_sha256']:
        raise RuntimeError('candidate freeze changed after hidden input freeze')
    return protocol


def freeze_common(output, name, freeze_file):
    freeze = json.loads((output/freeze_file).read_text())
    actual = source_hashes()
    differences = [name for name, digest in freeze['source_sha256'].items()
                   if (name.startswith('engine/aniso_phase1/research_b/') or
                       name.startswith('benchmarks/research_b/')) and actual.get(name) != digest]
    if differences:
        raise RuntimeError(('candidate fingerprint mismatch', differences))
    return dict(schema_version=1, reviewer='independent B7 review',
        created_utc=datetime.now(timezone.utc).isoformat(), hidden_kind=name,
        candidate_freeze_file=freeze_file, candidate_freeze_sha256=sha(output/freeze_file),
        review_entry_sha256=sha(Path(__file__)), source_sha256=actual,
        protocol_sha256=sha(output/'protocol.json'), gpu_used=False,
        threads=1, dtype='float64', thresholds=dict(U=.01, force=.01, weak_moments=.01, tangent_action=.02),
        reference_budget_fraction=.1, normalization='scenes.errors frozen dimensioned floors',
        historical_spatial_reference_used=False,
        exclusions=['C dynamics', 'new continuum accuracy claim', 'pointwise compressed stress reconstruction'],
        policy='one opening only; record any failed family, do not tune this candidate on hidden data')


def small_directions():
    rng = np.random.default_rng(902631)
    random = rng.normal(size=(9, 3))
    affine = np.zeros((9, 3)); affine[0,0]=.8; affine[1,1]=-.4; affine[0,2]=.3
    bending = np.zeros((9,3)); bending[4,2]=1.; bending[8,1]=-1.
    soft = np.zeros((9,3)); soft[2,2]=1.
    stress = np.zeros((9,3)); stress[0,1]=.6; stress[1,0]=1.; stress[0,0]=.3
    return {k: a/np.linalg.norm(a) for k,a in dict(random=random, affine=affine,
            quadratic_bending=bending, soft=soft, stress_sensitive=stress).items()}


def freeze_small(output, freeze_file):
    record = freeze_common(output, 'small', freeze_file)
    record.update(families=FAMILIES, stages=STAGES,
        source_orders=[4,5], candidate=dict(method='representatives',groups_per_partition=8),
        spatial_direction_formula='theta_deg=112+25*(X-.5)/.375+12*(Y-.5)/.125; phi_deg=17*(Z-.5)/.125; a=(cos(theta)*cos(phi),sin(theta)*cos(phi),sin(phi))',
        q_interpretation='q rows [x,y,z,x²/2,y²/2,z²/2,xy,xz,yz] centered at .5; columns displacement components; q multiplied by stage scale',
        directions=small_directions())
    write(output/'hidden-small-protocol.json', record)
    print('FROZEN independent small input', flush=True)


def hidden_rule(family, order):
    base = source_rule(family['angle'], family['mixture'], order=order)
    if not family['spatial_turn']:
        return base
    X = base.X
    theta = np.deg2rad(112+25*(X[:,0]-.5)/.375+12*(X[:,1]-.5)/.125)
    phi = np.deg2rad(17*(X[:,2]-.5)/.125)
    a = np.column_stack((np.cos(theta)*np.cos(phi), np.sin(theta)*np.cos(phi), np.sin(phi)))
    return MaterialRule.from_directions(X, base.weight, a, base.partition)


def matched(op, q, direction):
    result = op.evaluate(q, direction)
    result['weak_moments'] = op.weak_moments(q)
    return result


def maxima(rows, field):
    return {metric: max((dict(relative=row[field][metric]['relative'],
        absolute=row[field][metric]['absolute'], case=row['case'], direction=row['direction'])
        for row in rows), key=lambda r:r['relative']) for metric in METRICS}


def rejected(call):
    try:
        with np.errstate(all='ignore'):
            call()
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        return dict(rejected=True, exception=type(error).__name__, message=str(error))
    return dict(rejected=False)


def transaction_audit(full, compact, q, direction):
    q = q.copy(); saved = q.copy()
    session = MaterialSession(compact, 1e-5, 5e-5)
    session.trial(q)
    frozen_trial = rejected(lambda:session.propose(full,q,direction))
    invalid = np.zeros_like(q); invalid[0,0]=-2.
    invalid_trial = rejected(lambda:session.trial(invalid))
    invalid_commit = rejected(lambda:session.commit(accepted_step=True))
    version_failed = session.cache_version
    session.rollback()
    session.trial(q); session.commit(accepted_step=True)
    proposal = session.propose(full,q,direction)
    stale = rejected(lambda:session.commit_rule(proposal,q*.99,accepted_step=True))
    accepted = session.commit_rule(proposal,q,accepted_step=True)
    stable_state = bool(np.array_equal(q,saved))
    singular = np.zeros_like(q); singular[0,0]=-1.
    nan_state = np.zeros_like(q); nan_state[0,0]=float('nan')
    huge = np.zeros_like(q); huge[0,0]=1e200
    guards = dict(inverted=invalid_trial, singular=rejected(lambda:compact.evaluate(singular)),
        nonfinite=rejected(lambda:compact.evaluate(nan_state)),
        overflow_energy_only=rejected(lambda:compact.evaluate(huge)))
    passed = all(x['rejected'] for x in guards.values()) and frozen_trial['rejected'] and invalid_commit['rejected'] and stale['rejected'] and version_failed==0 and stable_state
    return dict(passed=bool(passed), guards=guards, frozen_inside_trial=frozen_trial,
        failed_trial_cannot_commit=invalid_commit, stale_rule_state=stale,
        caller_state_unchanged=stable_state, rule_change_accepted=accepted,
        frozen_single_budget_J=1e-5, frozen_cumulative_budget_J=5e-5,
        cumulative_absolute=session.cumulative_absolute, ledger=session.ledger)


def run_small(output):
    p = verify_frozen(output, 'small')
    if (output/'hidden-small-result.json').exists() or (output/'hidden-small-raw.npz').exists():
        raise RuntimeError('hidden results are append-only; this opening has already run')
    start=time.perf_counter(); rows=[]; raw={}; audits=[]; minimum=1.
    for family in p['families']:
        full_rule=hidden_rule(family,4); high_rule=hidden_rule(family,5)
        rule=compress(full_rule,8,'representatives')
        compact,full,high=operator(rule),operator(full_rule),operator(high_rule)
        audits.append(dict(family=family['name'], **moment_audit(full_rule,rule)))
        rule.save(output/(family['name']+'-rule.npz'))
        for stage,scale in p['stages']:
            q=np.asarray(family['q'])*scale; case=family['name']+'/'+stage
            raw[case+'/q']=q
            for label,arr in p['directions'].items():
                d=np.asarray(arr); raw['directions/'+label]=d
                candidate=matched(compact,q,d); reference=matched(full,q,d); check=matched(high,q,d)
                error=errors(candidate,reference); certification=errors(reference,check,scale=.1)
                minimum=min(minimum,candidate['min_detF'],reference['min_detF'],check['min_detF'])
                rows.append(dict(case=case,direction=label,errors=error,reference_self_check=certification,
                    passed=all(v['passed'] for v in error.values()),
                    reference_certified=all(v['passed'] for v in certification.values()),
                    min_detF=min(candidate['min_detF'],reference['min_detF'],check['min_detF'])))
                for tag,response in [('candidate',candidate),('order4',reference),('order5',check)]:
                    for metric in METRICS:
                        raw[case+'/'+label+'/'+tag+'/'+metric]=response[metric]
            print('HIDDEN SMALL',case,'complete',flush=True)
        if family==p['families'][0]:
            transactions=transaction_audit(full,compact,np.asarray(family['q']),np.asarray(p['directions']['random']))
    failures=[dict(case=r['case'],direction=r['direction'],metrics=[k for k,v in r['errors'].items() if not v['passed']])
              for r in rows if not r['passed']]
    cert=all(r['reference_certified'] for r in rows)
    passed=not failures and cert and transactions['passed']
    np.savez_compressed(output/'hidden-small-raw.npz',**raw)
    result=dict(passed=passed, candidate_passed=not failures,reference_certified=cert,
        independent_whole_families=len(p['families']),states=len(p['families'])*len(p['stages']),
        tangent_directions=len(p['directions']),max_errors=maxima(rows,'errors'),
        max_reference_errors=maxima(rows,'reference_self_check'),failures=failures,
        moment_audits=audits,transactions=transactions,min_detF=minimum,rows=rows,
        elapsed_seconds=time.perf_counter()-start,raw_sha256=sha(output/'hidden-small-raw.npz'),
        protocol_sha256=sha(output/'hidden-small-protocol.json'),
        conclusion='one-shot hidden acceptance; no post-opening tuning or retry')
    write(output/'hidden-small-result.json',result)
    print('HIDDEN SMALL RESULT',json.dumps({k:result[k] for k in ('passed','candidate_passed','reference_certified','failures','max_errors')}),flush=True)


def freeze_v22(output, freeze_file):
    from engine.aniso_phase1.research_b.tensor import V22Space
    record=freeze_common(output,'v22',freeze_file)
    frozen=json.loads((output/freeze_file).read_text())
    order=frozen['tensor_candidate_order']
    if order is None:
        raise RuntimeError('no frozen tensor candidate to review')
    s=V22Space(ROOT); q=s.q.copy()
    local=q*.83; local[-2,1]+=1.7e-5; local[-7,2]-=9e-6
    states=dict(hidden_load118=q*1.18,hidden_local83=local)
    rng=np.random.default_rng(902632)
    random=rng.normal(size=q.shape)
    local_d=np.zeros_like(q);local_d[-2,1]=1.;local_d[-7,2]=-.53
    # Scope fixed before input freeze: small hidden suite keeps all five
    # planned directions; expensive v22 adds two independent directions.
    ds={k:d/np.linalg.norm(d) for k,d in dict(random=random,stress_local=local_d).items()}
    record.update(candidate_order=order,reference_orders=[6,7],
        frozen_space_signature=s.signature,scalar_dofs=s.ndof,
        states={'hidden_load118':'1.18 * archived round6 displacement',
                'hidden_local83':'0.83 * archived round6 displacement; q[-2,1]+=1.7e-5; q[-7,2]-=9e-6'},
        local_direction='q[-2,1]=1; q[-7,2]=-.53; normalized',
        tangent_directions=list(ds),random_seed=902632,
        input_file='hidden-v22-inputs.npz',
        invalid_F_probe='affine u=-2X, hence F=-I; no reference spatial error results used')
    # Protocol is written before material response evaluation or input artifact.
    write(output/'hidden-v22-protocol.json',record)
    np.savez_compressed(output/'hidden-v22-inputs.npz',
        **{'state/'+k:v for k,v in states.items()},**{'direction/'+k:v for k,v in ds.items()})
    write(output/'hidden-v22-inputs-sha256.json',{'sha256':sha(output/'hidden-v22-inputs.npz')})
    print('FROZEN independent v22 inputs, candidate order',order,flush=True)


def run_v22(output):
    from engine.aniso_phase1.research_b.tensor import V22Space,TensorRule,TensorMaterialOperator
    p=verify_frozen(output,'v22')
    if (output/'hidden-v22-result.json').exists() or (output/'hidden-v22-raw.npz').exists():
        raise RuntimeError('hidden results are append-only; this opening has already run')
    expected=json.loads((output/'hidden-v22-inputs-sha256.json').read_text())['sha256']
    if sha(output/'hidden-v22-inputs.npz')!=expected:
        raise RuntimeError('v22 hidden inputs changed after freeze')
    s=V22Space(ROOT)
    if s.signature!=p['frozen_space_signature']:
        raise RuntimeError('v22 hidden space changed')
    with np.load(output/'hidden-v22-inputs.npz') as data:
        states={k[6:]:data[k].copy() for k in data if k.startswith('state/')}
        ds={k[10:]:data[k].copy() for k in data if k.startswith('direction/')}
    ops={o:TensorMaterialOperator(s,TensorRule.uniform(s.edges,o)) for o in [p['candidate_order'],6,7]}
    start=time.perf_counter(); rows=[]; raw={}; minimum=1.
    for name,q in states.items():
        for label,d in ds.items():
            values={}
            for order,op in ops.items():
                tick=time.perf_counter();value=op.evaluate(q,d);values[order]=value
                for metric in METRICS+('material_U','material_force','material_tangent_action','slab_energy','slab_tangent_work'):
                    raw[name+'/'+label+'/'+str(order)+'/'+metric]=value[metric]
                minimum=min(minimum,value['min_detF'])
                print('HIDDEN V22',name,label,'order',order,'seconds',time.perf_counter()-tick,flush=True)
            error=errors(values[p['candidate_order']],values[6]);certification=errors(values[6],values[7],scale=.1)
            rows.append(dict(case=name,direction=label,errors=error,reference_self_check=certification,
                passed=all(v['passed'] for v in error.values()),reference_certified=all(v['passed'] for v in certification.values())))
    invalid=np.zeros_like(s.q); invalid[:s.n]=-2*s.reference[:s.n]
    illegal=rejected(lambda:ops[p['candidate_order']].evaluate(invalid))
    failures=[dict(case=r['case'],direction=r['direction'],metrics=[k for k,v in r['errors'].items() if not v['passed']])
        for r in rows if not r['passed']]
    cert=all(r['reference_certified'] for r in rows)
    np.savez_compressed(output/'hidden-v22-raw.npz',**raw)
    result=dict(passed=not failures and cert and illegal['rejected'],candidate_passed=not failures,
        reference_certified=cert,states=len(states),tangent_directions=len(ds),
        max_errors=maxima(rows,'errors'),max_reference_errors=maxima(rows,'reference_self_check'),
        failures=failures,min_detF=minimum,invalid_F=illegal,rows=rows,
        elapsed_seconds=time.perf_counter()-start,raw_sha256=sha(output/'hidden-v22-raw.npz'),
        protocol_sha256=sha(output/'hidden-v22-protocol.json'),
        conclusion='new hidden amplitudes/local combination; fixed v22 space only, no continuum accuracy claim')
    write(output/'hidden-v22-result.json',result)
    print('HIDDEN V22 RESULT',json.dumps({k:result[k] for k in ('passed','candidate_passed','reference_certified','failures','max_errors')}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['freeze-small','run-small','freeze-v22','run-v22'],required=True)
    parser.add_argument('--freeze-file')
    args=parser.parse_args()
    if args.stage=='freeze-small':freeze_small(args.output,args.freeze_file or 'small-candidate-freeze.json')
    elif args.stage=='run-small':run_small(args.output)
    elif args.stage=='freeze-v22':freeze_v22(args.output,args.freeze_file or 'candidate-freeze.json')
    else:run_v22(args.output)


if __name__=='__main__':main()
