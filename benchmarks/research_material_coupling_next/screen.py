import argparse,time
from pathlib import Path
import numpy as np
from .common import MaterialStateModel,samples,write
from engine.aniso_phase1.research_material_coupling_next.budget import Budget
from engine.aniso_phase1.research_material_coupling_next.operators import Material,compare

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();budget=Budget();m=MaterialStateModel();states,w=samples(m)
    ops={n:Material(m,n,budget) for n in (2,5,7)};rows=[]
    for name,q in states.items():
        np.savez_compressed(a.run/'S1'/f'{name}.npz',q=q,direction=w)
        ref=ops[7].evaluate(q,w);low=ops[5].evaluate(q,w);report=compare(m,low,ref)
        item=dict(state=name,q5=report,reference_points=ref['material_calls'],candidate_points=low['material_calls'])
        if name=='local':item['q2_diagnostic']=compare(m,ops[2].evaluate(q),ref)
        rows.append(item);write(a.run/'S1/material-screen.json',dict(states=rows,budget=budget.report(),accepted=all(x['q5']['accepted'] for x in rows)))
        print(name,'q5 max error',report['max_error'],'J',ref['min_detF'],'q2',item.get('q2_diagnostic',{}).get('max_error'),flush=True)
    # One independent energy-force check in the mixed material-coordinate state.
    q=states['mixed'];eps=1e-3;f=ops[5].evaluate(q)['force'];plus=ops[5].evaluate(q+eps*w)['U'];minus=ops[5].evaluate(q-eps*w)['U']
    derivative=float(np.sum(f*w));fd=(plus-minus)/(2*eps);error=abs(fd-derivative)/max(abs(derivative),1e-10)
    write(a.run/'S1/energy-derivative.json',dict(relative=error,fd=fd,analytic=derivative,budget=budget.report()));assert error<.001
    assert all(x['q5']['accepted'] for x in rows),'q5 certification failed; keep full rule'
