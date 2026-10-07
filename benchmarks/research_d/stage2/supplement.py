"""Complete grouped integration, mapping, final-source and failure checks."""
import json,os,time
from pathlib import Path
import numpy as np
import warp as wp
from engine.aniso_phase1.research_d.frozen_inputs import load_frozen_inputs
from engine.aniso_phase1.research_d.stage2.gpu_operator import GPUOperator
from engine.aniso_phase1.research_d.stage2.cpu_operator import contract_for
from engine.aniso_phase1.research_d.stage2.contracts import sha
from benchmarks.research_d.stage2.bootstrap import PARENT,PIN
from benchmarks.research_d.stage2.validate import OUT,save,metric
from benchmarks.research_d.integration.material_reference import _pair

def main():
    wp.config.kernel_cache_dir='/root/autodl-tmp/mpm-lite-research-d/stage2/warp-cache'
    root=Path.cwd();s,_,_,_=load_frozen_inputs(root/PARENT,root,expected_sha256=PIN)
    pp=json.loads((root/PARENT/'protocol.json').read_text())['material_reference']
    groups={'all':np.arange(s.ndof*3),**{k:np.asarray(v) for k,v in pp['dof_groups'].items()}}
    result={}
    for direction in ('direction_mixed','direction_local_only_direction'):
        base='state_perturbation_3e-4_m-'+direction
        def load(name):
            with np.load(OUT/(base+name)) as z:r={k:z[k] for k in z.files}
            r['force']=r['full_force'];r['tangent_action']=r['full_tangent_action'];return r
        a,b=load('-cpu.npz'),load('-order7.npz')
        result[direction]=dict(response=_pair(a,b,groups,pp['budgets']),tangent=_pair(a,b,groups,pp['budgets'],tangent=True))
    save('grouped-new-state-integration.json',result)
    with np.load(OUT/'vectors.npz') as z:vals={k:z[k] for k in z.files}
    op=GPUOperator(s);recheck={}
    for name in ('state_archived_static','state_fixed_perturbation_1e-4_m','state_perturbation_3e-4_m'):
        for dn in ('direction_mixed','direction_local_only_direction'):
            r=op.evaluate(s.restrict(vals[name]),s.restrict(vals[dn]))
            with np.load(OUT/(name+'-'+dn+'-cpu.npz')) as cpu:
                recheck[name+'/'+dn]={k:metric(r[k],cpu[k]) for k in ('U','full_force','full_tangent_action')}
    save('final-source-operator-recheck.json',recheck)
    sources={str(p.relative_to(root)):sha(p) for area in ('engine/aniso_phase1','benchmarks','tests') for p in (root/area/'research_d/stage2').rglob('*.py')}
    save('final-validation-source-sha256.json',sources)
    from dataclasses import asdict
    save('static-contract.json',asdict(contract_for(op,s.q0,extension_sources=sources,device='cuda:0')))
    # Actual invalid static displacement is rejected, with the caller's q intact.
    illegal=s.q0.copy();illegal[:s.nfree_carrier,0]=-2*s.carrier_X[s.free_scalar_ids[:s.nfree_carrier],0]
    copy=illegal.copy();rejected=False
    try:op.evaluate(illegal)
    except ValueError:rejected=True
    if not np.array_equal(copy,illegal):raise AssertionError('operator mutated invalid input')
    save('real-invalid-state.json',dict(rejected=rejected,input_unchanged=True))
    def allmetrics(v):
        if isinstance(v,dict):return v.get('passed',True) and all(allmetrics(x) for x in v.values() if isinstance(x,(dict,list)))
        if isinstance(v,list):return all(allmetrics(x) for x in v)
        return True
    base=json.loads((OUT/'validation-status.json').read_text());c=json.loads((OUT/'curvature.json').read_text())
    checks=dict(base['checks'],grouped_new_state=allmetrics(result),mapping=allmetrics(json.loads((OUT/'mapping-audit.json').read_text())),
        tangent_symmetry=c['archived']['symmetry']<=1e-10,final_source_operator=allmetrics(recheck),invalid_state=rejected)
    save('correctness-acceptance.json',dict(passed=all(checks.values()),checks=checks,
        bounded_states=list(k for k in vals if k.startswith('state_')),dynamic_certified=False,continuum_spatial_certified=False,
        certified_scope='static operator and equilibrium on the fixed parent instance',workspace_estimate_bytes=op.estimated_workspace_bytes))
    print(json.dumps(checks),flush=True)

if __name__=='__main__':main()
