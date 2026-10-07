"""Close new-direction quadrature coverage and check the solved endpoint."""
import json
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_d.frozen_inputs import load_frozen_inputs
from engine.aniso_phase1.history_increment import material_tangent
from engine.aniso_phase1.research_d.stage2.solver import curvature,solve_linear
from benchmarks.research_d.stage2.bootstrap import PARENT,PIN
from benchmarks.research_d.stage2.validate import OUT,save
from benchmarks.research_d.integration.material_reference import _pair

def main():
    root=Path.cwd();s,_,_,_=load_frozen_inputs(root/PARENT,root,expected_sha256=PIN)
    pp=json.loads((root/PARENT/'protocol.json').read_text())['material_reference']
    groups={'all':np.arange(s.ndof*3),**{k:np.asarray(v) for k,v in pp['dof_groups'].items()}}
    with np.load(OUT/'vectors.npz') as z:vals={k:z[k] for k in z.files}
    d=s.restrict(vals['direction_mixed']);result={}
    for name in ('state_archived_static','state_fixed_perturbation_1e-4_m'):
        q=s.restrict(vals[name]);ref=s.response(q,d,order=7)
        with np.load(OUT/(name+'-direction_mixed-cpu.npz')) as z:a={k:z[k] for k in z.files}
        a['force']=a['full_force'];a['tangent_action']=a['full_tangent_action']
        b=dict(ref,force=ref['full_force'],tangent_action=ref['full_tangent_action'])
        result[name]=dict(response=_pair(a,b,groups,pp['budgets']),tangent=_pair(a,b,groups,pp['budgets'],tangent=True))
        np.savez(OUT/(name+'-direction_mixed-order7.npz'),**ref)
        print('quadrature',name,result[name]['response']['passed'],result[name]['tangent']['passed'],flush=True)
    with np.load(OUT/'cpu-equilibrium.npz') as z:q=z['q'];a={k:z[k] for k in z.files}
    ref=s.response(q,order=7);a['force']=a['full_force'];b=dict(ref,force=ref['full_force'])
    result['derived_equilibrium_endpoint']=dict(response=_pair(a,b,groups,pp['budgets']))
    np.savez(OUT/'equilibrium-order7.npz',**ref)
    passed=all(r['passed'] for state in result.values() for r in state.values())
    save('complete-material-scope.json',dict(passed=passed,additional_checks=result,parent_local_directions_reused=True,
       exact_integration_claim=False,future_state_claim=False))
    controls={}
    for name,F in [('repeat',np.eye(3)),('compression',np.diag([.65,.8,1.])),('expansion',np.diag([4.,3.,2.]))]:
        direction=np.eye(9).reshape(9,3,3);H=material_tangent(np.broadcast_to(F,(9,3,3)),np.broadcast_to(s.A,(9,3,3)),direction,s.params).reshape(9,9).T
        _,info=solve_linear(H,np.ones(9));controls[name]=dict(curvature=curvature(H),solver=info['method'],residual=info['true_residual'])
    save('material-curvature-controls.json',controls)
    acc=json.loads((OUT/'correctness-acceptance.json').read_text());acc['checks']['complete_material_scope']=passed
    acc['passed']=all(acc['checks'].values());save('correctness-acceptance.json',acc)
    print('complete material scope',passed,controls,flush=True)

if __name__=='__main__':main()
