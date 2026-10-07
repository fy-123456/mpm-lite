"""Focused grip/edge refinement, with same-order and cross-order checks."""
import numpy as np
from benchmarks.aniso_v22_common import *
from engine.aniso_phase1.tensor_reference import solve

def main():
    protocol=load(OUT/'protocol.json')
    for n,d in protocol['source_sha256'].items():assert sha(ROOT/n)==d,n
    folder=OUT/'reference';folder.mkdir(exist_ok=False)
    previous={p:read_field(BASE/f'v21/reference-extension/level1-q{p}.npz') for p in (3,4)}
    edges=[e.copy() for e in previous[4][0]];edges[0]=np.r_[.125,edges[0][(edges[0]>=.25)&(edges[0]<=.75)],.875];cases={};pairs={}
    for level in range(2):
        for k,e in enumerate(edges):
            active=np.flatnonzero((e[:-1]>=.25)&(e[1:]<=.75)) if k==0 else np.arange(len(e)-1)
            ids=np.union1d(active[:2],active[-2:]);edges[k]=np.union1d(e,(e[:-1]+e[1:])[ids]/2)
        nodes=int(np.prod([4*(len(e)-1)+1 for e in edges]));assert nodes<14000000
        write(folder/f'level{level}-selection.json',dict(edges=[e.tolist() for e in edges],q4_nodes=nodes,selection='Two nearest intervals on each side of free span and each transverse boundary; prior reference mesh otherwise retained.'))
        fields={}
        for p in (3,4):
            name=f'level{level}-q{p}';u,result=solve(edges,p,hessian('F45'),initial=previous[p],rtol=2e-12,callback=lambda n:print(name,'iteration',n,flush=True));assert result['passed'] and result['work_identity_relative']<1e-8
            path=folder/(name+'.npz');np.savez_compressed(path,u=u,degree=p,**{f'axis{k}':e for k,e in enumerate(edges)});write(path.with_suffix('.json'),result);fields[p]=([e.copy() for e in edges],p,u);cases[name]=result;print('SOLVED',name,result,flush=True)
            pair=compare(previous[p],fields[p]);pairs[name+'-h']=pair;write(folder/(name+'-h.json'),pair);print('H CHECK',name,{k:v['stress_relative'] for k,v in pair['regions'].items()},flush=True)
        cross=compare(fields[3],fields[4]);pairs[f'level{level}-p']=cross;write(folder/f'level{level}-p.json',cross);print('P CHECK',level,{k:v['stress_relative'] for k,v in cross['regions'].items()},flush=True);previous=fields
        write(OUT/'reference-summary.json',dict(completed=False,cases=cases,pairs=pairs))
    final=[v for k,v in pairs.items() if k.startswith('level1')];gates={k:dict(stress_passed=all(v['regions'][k]['stress_relative']<.02 for v in final),fiber_passed=all(v['regions'][k]['fiber_strain_relative']<.02 for v in final)) for k in ('global','grip','interior','deep_interior')}
    write(OUT/'reference-summary.json',dict(completed=True,cases=cases,pairs=pairs,regions=gates,latest_reference='reference/level1-q4.npz',all_stress_and_fiber_passed=all(v['stress_passed'] and v['fiber_passed'] for v in gates.values()),continuum_error_bound_proved=False))
if __name__=='__main__':main()
