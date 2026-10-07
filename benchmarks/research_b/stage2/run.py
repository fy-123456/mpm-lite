"""Freeze -> development -> sealed selection -> one hidden opening -> delivery.

Run from the same isolated checkout that the parent loader verifies. Raw
responses, checkpoint metadata and failed trials are retained on the data disk.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
import numpy as np
from engine.aniso_phase1.research_d.frozen_inputs import load_frozen_inputs
from engine.aniso_phase1.research_b.stage2 import CommonMaterialOperator, FixedRule, MaterialSource
from engine.aniso_phase1.research_b.stage2.material import PARENT_SHA, SPACE_SHA, digest
from .states import build, SEED
from .metrics import metrics, save_response, load_response

PARENT=Path('docs/results/parallel-v22/integration/20260930T054100Z-common-inputs')
FIELDS=('F45','partition','turning','two_family_3d')


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
def read(path):return json.loads(Path(path).read_text())
def source_hashes(repo):
    return {str(p.relative_to(repo)):sha(p) for part in ('engine/aniso_phase1/research_b/stage2','benchmarks/research_b/stage2','tests/research_b/stage2') for p in sorted((repo/part).rglob('*.py'))}
def load(repo):return load_frozen_inputs(repo/PARENT,repo,expected_sha256=PARENT_SHA)


def freeze(repo,out):
    if (out/'protocol.json').exists():raise ValueError('protocol already frozen; choose a new run')
    s,_,_,audit=load(repo)
    rows,directions=build(s)
    out.mkdir(parents=True,exist_ok=True);(out/'raw').mkdir(exist_ok=True)
    arrays={r['name']:r['q'] for r in rows}
    arrays.update({'direction__'+k:v for k,v in directions.items()})
    np.savez_compressed(out/'states.npz',**arrays)
    public=[{k:v for k,v in r.items() if k!='q'} for r in rows]
    write(out/'state-families.json',dict(seed=SEED,states=public,direction_names=list(directions),
        normalization='max Frobenius gradient on fixed two-point kinematic probes; not a candidate fit',
        amplitude_tiers=[.045,.11,.21,.31],hidden_families=3,hidden_states=12,directions_per_state=5,
        replacement_policy='no automatic replacement; illegal states retained and fail their scope'))
    write(out/'baseline-check.json',dict(**audit,verified_at=now(),repository=str(repo),
        full_shape=[s.ndof,3],free_shape=list(s.q_shape),free_scalar_ids=s.free_scalar_ids.tolist(),
        source_snapshot='parent code-snapshot.zip plus byte-identical parent data and stage2 extension',
        parent_inputs={k:dict(path=str((repo/PARENT/k).resolve()),sha256=v) for k,v in read(repo/PARENT/'bundle.json')['files'].items()},
        initial_snapshot='static right grip displacement 0.005 m; time zero is not stress free'))
    protocol=dict(schema_version=1,created_at=now(),parent_bundle_sha256=PARENT_SHA,space_manifest_sha256=SPACE_SHA,
        extension_source_sha256=source_hashes(repo),states_sha256=sha(out/'states.npz'),state_families_sha256=sha(out/'state-families.json'),
        seed=SEED,fields={f:MaterialSource(f).manifest() for f in FIELDS},dtype='float64',device='cpu',
        units={'length':'m','energy':'J','force':'positive potential gradient, N','moduli':'Pa'},
        state_semantics={'static_free':[219,3],'dynamic_full_displacement':[369,3],'lift':[369,3],
                         'lift_velocity':[369,3],'direction_lift':'zero','reference_identity_added_once':True},
        boundaries='frozen grips, explicit full lift supplied by caller; state family q retains prescribed rows',
        invariant_mass=dict(density_kg_m3=1.,order=5,model='full reference continuum point inertia with carrier-local cross blocks'),
        orders=[3,4,5,6,7],escalation=8,preferred_candidate=4,
        selection='4 if every development metric passes; else 5; else certified full reference. No hidden tuning.',
        reference_budgets=dict(energy=dict(atol=1e-10,rtol=.002),force=dict(atol=2e-7,rtol=.002),
                               weak=dict(atol=2e-9,rtol=.002),tangent=dict(atol=2e-6,rtol=.005),work=dict(atol=2e-8,rtol=.005)),
        candidate_budgets=dict(energy=dict(atol=5e-10,rtol=.01),force=dict(atol=1e-6,rtol=.01),
                               weak=dict(atol=1e-8,rtol=.01),tangent=dict(atol=1e-5,rtol=.02),work=dict(atol=1e-7,rtol=.02)),
        numerical_policy='user requested practical numerical accuracy; new reference suggestions relaxed to 0.2%/0.5%; candidate 1%/2%; absolute near-zero floors fixed; parent tests unchanged',
        derivative_steps=[3e-5,1e-5,3e-6],min_detF=.15,
        reference_scope='finite registered states and directions; not a uniform error bound over all detF>0.15 states',
        resources=dict(workers=4,blas_threads_per_worker=1,estimated_peak_memory_GiB=8,wall_limit_seconds=7200,
                       checkpoint='one atomically written response and metadata per state/material/order',
                       retry='only missing tasks, reject protocol/source/input mismatch; retain failure results',
                       stop='disk below 5 GiB on root or below 2 GiB on data; invalid identity; deadline'),
        machine=dict(platform=platform.platform(),processor=platform.processor(),cpu_count=os.cpu_count(),
                     affinity=sorted(os.sched_getaffinity(0)),numpy=np.__version__),
        performance='shared machine: exploratory timings only unless exclusive resource window is evidenced',
        capabilities=dict(static_operator='pending',bounded_material_reference='pending',dynamic_cycle='dependency_pending',cuda='not_run',coupled_physics='not_run'))
    write(out/'protocol.json',protocol);write(out/'protocol-seal.json',dict(sha256=sha(out/'protocol.json'),sealed_at=now()))
    print(json.dumps({'phase':'frozen','states':len(rows),'fields':len(FIELDS),'protocol_sha256':sha(out/'protocol.json')}),flush=True)


def checked(repo,out):
    p=read(out/'protocol.json')
    if sha(out/'protocol.json')!=read(out/'protocol-seal.json')['sha256']:raise ValueError('protocol changed')
    if source_hashes(repo)!=p['extension_source_sha256']:raise ValueError('extension sources changed after seal')
    if sha(out/'states.npz')!=p['states_sha256'] or sha(out/'state-families.json')!=p['state_families_sha256']:raise ValueError('state inputs changed')
    return p


_WORKER={}
def worker_init(repo,out):
    repo=Path(repo);out=Path(out);p=checked(repo,out);s,_,_,_=load(repo)
    with np.load(out/'states.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    _WORKER.update(repo=repo,out=out,p=p,s=s,arrays=arrays,operators={})


def task(job):
    name,field,order=job;w=_WORKER;out=w['out'];s=w['s'];p=w['p']
    stem=f'{field}__{name}__q{order}';meta=out/'raw'/f'{stem}.json';array=out/'raw'/f'{stem}.npz'
    bound=dict(protocol_sha256=sha(out/'protocol.json'),name=name,field=field,order=order)
    if meta.exists():
        saved=read(meta)
        if any(saved.get(k)!=v for k,v in bound.items()):raise ValueError('checkpoint identity mismatch')
        if saved['status']=='ok' and sha(array)!=saved['array_sha256']:raise ValueError('checkpoint response changed')
        return saved
    started=time.perf_counter()
    try:
        key=(field,order)
        if key not in w['operators']:
            if len(w['operators']) >= 2: w['operators'].pop(next(iter(w['operators'])))
            w['operators'][key]=CommonMaterialOperator(s,FixedRule.uniform(s,order),MaterialSource(field),min_detF=p['min_detF'])
        op=w['operators'][key]
        dirs={k.removeprefix('direction__'):v for k,v in w['arrays'].items() if k.startswith('direction__')}
        r=op.evaluate_full(w['arrays'][name],directions=dirs)
        save_response(array,r)
        result=dict(**bound,status='ok',seconds=time.perf_counter()-started,array_sha256=sha(array),
                    min_detF=r['min_detF'],energy_J=r['energy_J'],material_energy_J=r['material_energy_J'],
                    point_count=r['point_count'],source_sha256=op.source.signature,rule_sha256=op.rule.signature,
                    cache_sha256=op.signature,construction_seconds=op.construction_seconds,timing_seconds=r['timing_seconds'],
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,finished_at=now())
    except (ValueError,FloatingPointError) as exc:
        result=dict(**bound,status='rejected',reason=str(exc),seconds=time.perf_counter()-started,finished_at=now())
    write(meta,result)
    return result


def run_jobs(repo,out,jobs,p):
    deadline=time.monotonic()+p['resources']['wall_limit_seconds']
    import shutil
    if shutil.disk_usage('/').free < 5*1024**3 or shutil.disk_usage(out).free < 2*1024**3:raise RuntimeError('disk headroom stop')
    missing=[]
    for job in jobs:
        name,field,order=job;meta=out/'raw'/f'{field}__{name}__q{order}.json'
        if meta.exists():
            m=read(meta)
            if m['protocol_sha256']!=sha(out/'protocol.json'):raise ValueError('old checkpoint protocol')
            if m['status']=='ok' and sha(meta.with_suffix('.npz'))!=m['array_sha256']:raise ValueError('checkpoint bytes changed')
        else:missing.append(job)
    print(json.dumps({'phase':'jobs','total':len(jobs),'missing':len(missing),'workers':p['resources']['workers']}),flush=True)
    with ProcessPoolExecutor(max_workers=p['resources']['workers'],initializer=worker_init,initargs=(str(repo),str(out))) as pool:
        futures={pool.submit(task,j):j for j in missing}
        for i,future in enumerate(as_completed(futures),1):
            r=future.result();print(json.dumps({'done':i,'of':len(missing),**{k:r.get(k) for k in ('name','field','order','status','seconds','min_detF')}}),flush=True)
            if time.monotonic()>deadline:
                for f in futures:f.cancel()
                raise RuntimeError('registered time limit exceeded; completed checkpoints preserved')


def pair(out,space,name,field,a,b,budget):
    paths=[out/'raw'/f'{field}__{name}__q{o}.json' for o in (a,b)]
    if not all(p.exists() and read(p)['status']=='ok' for p in paths):return dict(passed=False,status='illegal_or_missing',failed=['state response unavailable'])
    result=metrics(*(load_response(p.with_suffix('.npz')) for p in paths),space,budget)
    result.update(candidate_order=a,reference_order=b,name=name,field=field)
    return result


def development(repo,out):
    p=checked(repo,out);s,_,_,_=load(repo)
    states=[r for r in read(out/'state-families.json')['states'] if r['split']!='hidden']
    jobs=[(r['name'],f,o) for r in states for f in FIELDS for o in p['orders']]
    run_jobs(repo,out,jobs,p)
    refs={};scans={};selection={}
    for field in FIELDS:
        ref6={r['name']:pair(out,s,r['name'],field,6,7,p['reference_budgets']) for r in states}
        order=6;reference_passed=all(r['passed'] for r in ref6.values())
        record=dict(six_vs_seven=ref6)
        if not reference_passed:
            run_jobs(repo,out,[(r['name'],field,8) for r in states],p)
            ref7={r['name']:pair(out,s,r['name'],field,7,8,p['reference_budgets']) for r in states}
            order=7;reference_passed=all(r['passed'] for r in ref7.values());record['seven_vs_eight']=ref7
        record.update(order=order,passed=reference_passed);refs[field]=record
        scan={str(o):{r['name']:pair(out,s,r['name'],field,o,order,p['candidate_budgets']) for r in states} for o in (3,4,5)}
        selected=next((o for o in (4,5) if all(r['passed'] for r in scan[str(o)].values())),order)
        scans[field]=scan
        selection[field]=dict(candidate_order=selected,full_order=order,reference_passed=reference_passed,
                              source_sha256=MaterialSource(field).signature,
                              candidate_rule_sha256=FixedRule.uniform(s,selected).signature,
                              full_rule_sha256=FixedRule.uniform(s,order).signature,
                              point_reduction=1-(selected/order)**3,local_fallback_slabs=[],
                              fallback_policy='choose uniform higher order before physical step; local API independently tested')
    write(out/'full-reference-certification.json',refs);write(out/'rule-scan.json',scans)
    write(out/'candidate-seal.json',dict(sealed_at=now(),protocol_sha256=sha(out/'protocol.json'),
        extension_source_sha256=source_hashes(repo),fields=selection,hidden_opened=False,
        evidence_sha256={k:sha(out/k) for k in ('full-reference-certification.json','rule-scan.json')}))
    print(json.dumps({'phase':'candidate_sealed','fields':selection}),flush=True)


def hidden(repo,out):
    p=checked(repo,out);s,_,_,_=load(repo);seal=read(out/'candidate-seal.json')
    if seal['protocol_sha256']!=sha(out/'protocol.json') or seal['extension_source_sha256']!=source_hashes(repo):raise ValueError('candidate seal mismatch')
    opening=out/'hidden-opening.json'
    if opening.exists():
        if read(opening)['candidate_sha256']!=sha(out/'candidate-seal.json'):raise ValueError('hidden set already opened against another candidate')
    else:write(opening,dict(opened_at=now(),candidate_sha256=sha(out/'candidate-seal.json'),protocol_sha256=sha(out/'protocol.json'),policy='one evaluation; checkpoint resume only; no candidate changes'))
    states=[r for r in read(out/'state-families.json')['states'] if r['split']=='hidden']
    jobs=[(r['name'],f,o) for r in states for f,sel in seal['fields'].items() for o in sorted({sel['candidate_order'],sel['full_order'],sel['full_order']+1})]
    run_jobs(repo,out,jobs,p)
    records={}
    for field,sel in seal['fields'].items():
        data={}
        for state in states:
            name=state['name'];full=sel['full_order']
            ref=pair(out,s,name,field,full,full+1,p['reference_budgets'])
            cmp=pair(out,s,name,field,sel['candidate_order'],full,p['candidate_budgets'])
            data[name]=dict(reference=ref,candidate=cmp,passed=ref['passed'] and cmp['passed'])
        records[field]=dict(states=data,passed=sel['reference_passed'] and all(r['passed'] for r in data.values()),
                            state_count=len(data),direction_count=5,near_domain_states=[r['name'] for r in states if r['amplitude']>=.3])
    write(out/'hidden-acceptance.json',dict(protocol_sha256=sha(out/'protocol.json'),candidate_sha256=sha(out/'candidate-seal.json'),
        opened_at=read(opening)['opened_at'],completed_at=now(),fields=records,passed=all(r['passed'] for r in records.values()),
        failed_states={f:[n for n,r in v['states'].items() if not r['passed']] for f,v in records.items()}))
    print(json.dumps({'phase':'hidden_complete','passed':{f:v['passed'] for f,v in records.items()}}),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('freeze','development','hidden'));ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    repo=Path(__file__).resolve().parents[3];out=args.output.resolve()
    globals()[args.phase](repo,out)
if __name__=='__main__':main()
