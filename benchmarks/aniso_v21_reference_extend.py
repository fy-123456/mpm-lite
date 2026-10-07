"""Explicit v21 extension after cross-order testing exposed interior error."""
import numpy as np
from benchmarks.aniso_v21_common import *
from engine.aniso_phase1.tensor_reference import solve

def main():
    folder=OUT/'reference-extension';folder.mkdir(exist_ok=True);p=folder/'protocol.json'
    if not p.exists():write(p,dict(reason='Local Q3 h differences are small in the interior, but Q3/Q4 stress differs by 3.2% and fiber strain by 5.1%. Add a physical interior resolution floor plus stress-marked boundary refinement.',input_summary_sha256=sha(OUT/'reference-summary.json'),source_sha256={n:sha(ROOT/n) for n in ['benchmarks/aniso_v21_reference_extend.py','engine/aniso_phase1/tensor_reference.py','benchmarks/aniso_v21_common.py']},max_physical_interval=1/64,mark_fraction=.125,solver_rtol=2e-12,maximum_q4_nodes=8000000,maximum_new_levels=2,scope='Same hard grips/material/geometry. No singular corner exclusion and no weakened stress gate.'))
    protocol=load(p)
    for n,d in protocol['source_sha256'].items():assert sha(ROOT/n)==d,n
    a=read_field(OUT/'reference/local3-q4.npz');b=read_field(OUT/'reference/local4-q3.npz');pairs={};cases={};previous_q4=None
    for level in range(2):
        key=f'level{level}'
        choice=folder/(key+'-selection.json')
        if not choice.exists():
            d=compare(a,b,indicators=True);edges,marked=mark(b[0],d['axis_indicators']);edges=[np.union1d(e,[(x+y)/2 for x,y in zip(e[:-1],e[1:]) if y-x>1/64+1e-12 and (k>0 or (.25<=x and y<=.75))]) for k,e in enumerate(edges)];write(choice,dict(edges=[e.tolist() for e in edges],marks=marked,comparison=d,q4_nodes=int(np.prod([4*(len(e)-1)+1 for e in edges]))))
        choice=load(choice);edges=[np.array(e) for e in choice['edges']]
        if choice['q4_nodes']>protocol['maximum_q4_nodes']:
            write(folder/'resource-limit.json',dict(level=level,proposed_q4_nodes=choice['q4_nodes'],maximum_q4_nodes=protocol['maximum_q4_nodes'],explicit_unfinished_reference_gate=True));break
        fields=[]
        for degree in (3,4):
            name=f'{key}-q{degree}';path=folder/(name+'.npz')
            if path.exists():field=read_field(path);result=load(path.with_suffix('.json'))
            else:
                u,result=solve(edges,degree,hessian('F45'),initial=b if degree==3 else fields[0],rtol=protocol['solver_rtol'],callback=lambda n:print(name,'iteration',n,flush=True));assert result['passed'] and result['work_identity_relative']<1e-8;np.savez_compressed(path,u=u,degree=degree,**{f'axis{k}':e for k,e in enumerate(edges)});write(path.with_suffix('.json'),result);field=(edges,degree,u);print('completed',name,result,flush=True)
            fields.append(field);cases[name]=result
        cross=compare(*fields);cross['reaction_relative']=abs(cases[key+'-q3']['reaction_N']-cases[key+'-q4']['reaction_N'])/abs(cases[key+'-q4']['reaction_N']);pairs[key+'-cross']=cross;write(folder/(key+'-cross.json'),cross);print(key,'cross',cross['regions'],flush=True)
        if previous_q4 is not None:
            h=compare(previous_q4,fields[1]);pairs[key+'-h-q4']=h;write(folder/(key+'-h-q4.json'),h)
        a,b=fields;previous_q4=b
        write(OUT/'reference-extension-summary.json',dict(completed=False,cases=cases,pairs=pairs,latest_reference=f'reference-extension/{key}-q4.npz'))
    last=[v for k,v in pairs.items() if k.startswith('level1')];passed=bool(last) and all(v['stress_passed'] and v['fiber_strain_passed'] for v in last)
    write(OUT/'reference-extension-summary.json',dict(completed=True,cases=cases,pairs=pairs,latest_reference=f'reference-extension/{list(cases)[-1]}.npz',stress_and_fiber_reference_passed=passed,continuum_error_bound_proved=False,resource_limit_reached=(folder/'resource-limit.json').exists()))
if __name__=='__main__':main()
