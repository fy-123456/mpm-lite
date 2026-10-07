"""Frozen adaptive high-order reference refinement; no altered physical model."""
import argparse,time
import numpy as np
from benchmarks.aniso_v21_common import *
from engine.aniso_phase1.tensor_reference import solve

def sources_ok():
    for n,d in load(OUT/'protocol.json')['source_sha256'].items():assert sha(ROOT/n)==d,n

def run_case(name,edges,p,initial):
    folder=OUT/'reference';folder.mkdir(exist_ok=True);path=folder/(name+'.npz')
    if path.exists():return read_field(path),load(path.with_suffix('.json'))
    sources_ok();u,r=solve(edges,p,hessian('F45'),initial=initial,callback=lambda k:print(name,'iteration',k,flush=True));r['name']=name;assert r['passed'] and r['work_identity_relative']<1e-8
    np.savez_compressed(path,u=u,degree=p,**{f'axis{k}':e for k,e in enumerate(edges)});write(path.with_suffix('.json'),r);print('completed',name,r,flush=True);sources_ok();return (edges,p,u),r

def main():
    sources_ok();folder=OUT/'reference';folder.mkdir(exist_ok=True);old=read_field(BASE/'v11-reference-q3/cases/local2.npz',3);q2=read_field(BASE/'v11-reference/cases/F45-local2-q2.npz',2)
    if not (folder/'selection.json').exists():
        indicators=compare(q2,old,indicators=True);e1,marks=mark(old[0],indicators['axis_indicators']);write(folder/'selection.json',dict(input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [BASE/'v11-reference-q3/cases/local2.npz',BASE/'v11-reference/cases/F45-local2-q2.npz']},comparison=indicators,marks=marks,edges=[e.tolist() for e in e1]))
    selection=load(folder/'selection.json');e1=[np.array(e) for e in selection['edges']]
    # Reproduce the previous independent assembled solve with the new operator.
    replay,rr=run_case('replay-q3',old[0],3,old);err=float(np.max(abs(replay[2]-old[2])));assert err<1e-9
    r1,a=run_case('local3-q3',e1,3,old)
    first=compare(old,r1,indicators=True);write(folder/'old-to-local3.json',first);e2,marks=mark(e1,first['axis_indicators']);write(folder/'selection-next.json',dict(marks=marks,edges=[e.tolist() for e in e2],source='old-to-local3 Q3 stress difference, fixed before next solve'))
    # Higher polynomial order and further h refinement are independent checks.
    p4,b=run_case('local3-q4',e1,4,r1)
    r2,c=run_case('local4-q3',e2,3,r1)
    pairs={}
    for name,l,r in [('old_to_local3',old,r1),('q3_to_q4_local3',r1,p4),('local3_to_local4_q3',r1,r2),('local3_q4_to_local4_q3',p4,r2)]:
        v=first if name=='old_to_local3' else compare(l,r);pairs[name]=v;write(folder/(name+'.json'),v);print('comparison',name,v['regions'],flush=True)
    all_cases={'replay-q3':rr,'local3-q3':a,'local3-q4':b,'local4-q3':c};pairs['q3_to_q4_local3']['reaction_relative']=abs(a['reaction_N']-b['reaction_N'])/abs(b['reaction_N']);pairs['local3_to_local4_q3']['reaction_relative']=abs(a['reaction_N']-c['reaction_N'])/abs(c['reaction_N']);pairs['local3_q4_to_local4_q3']['reaction_relative']=abs(b['reaction_N']-c['reaction_N'])/abs(c['reaction_N']);required=[pairs[n] for n in ('q3_to_q4_local3','local3_to_local4_q3','local3_q4_to_local4_q3')]
    write(OUT/'reference-summary.json',dict(completed=True,cases=all_cases,pairs=pairs,replay_displacement_max=err,stress_reference_passed=all(x['stress_passed'] for x in required),fiber_reference_passed=all(x['fiber_strain_passed'] for x in required),reaction_reference_passed=all(x['reaction_relative']<.01 for x in required),continuum_error_bound_proved=False,training_reference='archived local2 Q3',held_out_references=['local3-q4','local4-q3']))
if __name__=='__main__':main()
