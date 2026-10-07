"""Resume reference checks with stricter solves and per-attempt diagnostics.

Original thresholds and meshes are unchanged. Independent reference solves may
run concurrently because initial guesses do not define the equilibrium problem.
"""
import argparse,shutil
import numpy as np
from benchmarks.aniso_v22_common import *
from engine.aniso_phase1.tensor_reference import solve

def freeze():
    path=OUT/'reference-resume-protocol.json';assert not path.exists();folder=OUT/'reference';edges=[np.array(e) for e in load(folder/'level0-selection.json')['edges']]
    for k,e in enumerate(edges):
        active=np.flatnonzero((e[:-1]>=.25)&(e[1:]<=.75)) if k==0 else np.arange(len(e)-1);ids=np.union1d(active[:2],active[-2:]);edges[k]=np.union1d(e,(e[:-1]+e[1:])[ids]/2)
    write(folder/'level1-selection.json',dict(edges=[e.tolist() for e in edges],q4_nodes=int(np.prod([4*(len(e)-1)+1 for e in edges])),selection='Same second refinement specified in original protocol.'))
    write(path,dict(reason='Original level0 Q4 attempt failed its combined linear/work gate. Its pre-assert result was not persisted, so the failed subcondition cannot be reconstructed. Keep original log, retain identical gates, solve more tightly and persist every subsequent attempt before accepting it.',failed_log='/tmp/mpm-v22-reference.log',failed_log_sha256=sha(Path('/tmp/mpm-v22-reference.log')),rtols=[1e-13,1e-14],relative_residual_gate=1e-8,relative_work_gate=1e-8,source_sha256={n:sha(ROOT/n) for n in ['benchmarks/aniso_v22_reference_resume.py','engine/aniso_phase1/tensor_reference.py','engine/aniso_phase1/tensor_metrics.py']},parallelism='The three remaining fixed-mesh physical solves are independent; only warm-start choice differs from the original serial execution.'))

def job(name):
    protocol=load(OUT/'reference-resume-protocol.json')
    for n,d in protocol['source_sha256'].items():assert sha(ROOT/n)==d,n
    level=int(name[5]);p=int(name[-1]);folder=OUT/'reference';dest=folder/(name+'.npz');assert not dest.exists();edges=[np.array(e) for e in load(folder/f'level{level}-selection.json')['edges']]
    initial=read_field(folder/'level0-q3.npz') if p==3 else read_field(BASE/'v21/reference-extension/level1-q4.npz');attempts=folder/'attempts';attempts.mkdir(exist_ok=True)
    for index,rtol in enumerate(protocol['rtols']):
        print('START',name,'rtol',rtol,flush=True);u,r=solve(edges,p,hessian('F45'),initial=initial,rtol=rtol,callback=lambda n:print(name,'iteration',n,flush=True));r.update(requested_rtol=rtol,linear_gate_passed=bool(r['passed']),work_gate_passed=bool(r['work_identity_relative']<protocol['relative_work_gate']));record=attempts/f'{name}-attempt{index}';write(record.with_suffix('.json'),r);print('ATTEMPT',name,r,flush=True)
        np.savez_compressed(record.with_suffix('.npz'),u=u,degree=p,**{f'axis{k}':e for k,e in enumerate(edges)})
        if r['linear_gate_passed'] and r['work_gate_passed']:
            shutil.copy2(record.with_suffix('.npz'),dest);write(dest.with_suffix('.json'),r);print('ACCEPTED',name,flush=True);return
        initial=(edges,p,u)
    raise RuntimeError(f'{name}: prescribed tighter attempts exhausted; gates remain failed')

def finish():
    folder=OUT/'reference';cases={};pairs={};previous={p:read_field(BASE/f'v21/reference-extension/level1-q{p}.npz') for p in (3,4)}
    for level in (0,1):
        fields={}
        for p in (3,4):
            name=f'level{level}-q{p}';path=folder/(name+'.npz');assert path.exists();cases[name]=load(path.with_suffix('.json'));assert cases[name]['passed'] and cases[name]['work_identity_relative']<1e-8;fields[p]=read_field(path);metric=folder/(name+'-h.json')
            if metric.exists():pair=load(metric)
            else:pair=compare(previous[p],fields[p]);write(metric,pair)
            pairs[name+'-h']=pair;print('H CHECK',name,{k:v['stress_relative'] for k,v in pair['regions'].items()},flush=True)
        cross=compare(fields[3],fields[4]);pairs[f'level{level}-p']=cross;write(folder/f'level{level}-p.json',cross);print('P CHECK',level,{k:v['stress_relative'] for k,v in cross['regions'].items()},flush=True);previous=fields
    final=[v for k,v in pairs.items() if k.startswith('level1')];gates={k:dict(stress_passed=all(v['regions'][k]['stress_relative']<.02 for v in final),fiber_passed=all(v['regions'][k]['fiber_strain_relative']<.02 for v in final)) for k in ('global','grip','interior','deep_interior')}
    write(OUT/'reference-summary.json',dict(completed=True,cases=cases,pairs=pairs,regions=gates,latest_reference='reference/level1-q4.npz',all_stress_and_fiber_passed=all(v['stress_passed'] and v['fiber_passed'] for v in gates.values()),continuum_error_bound_proved=False,original_failed_attempt_preserved=True,continuation_protocol='reference-resume-protocol.json'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','job','finish']);p.add_argument('--name');a=p.parse_args();freeze() if a.action=='freeze' else job(a.name) if a.action=='job' else finish()
