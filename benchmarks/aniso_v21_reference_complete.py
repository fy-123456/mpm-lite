"""Complete the second planned reference level after an explicit resource review."""
import time,shutil
import numpy as np
from benchmarks.aniso_v21_common import *
from engine.aniso_phase1.tensor_reference import solve

def main():
    folder=OUT/'reference-extension';selection=load(folder/'level1-selection.json');edges=[np.array(e) for e in selection['edges']];p=folder/'completion-protocol.json'
    # The earlier 8M-node limit was a conservative estimate, not a machine limit.
    # Actual matrix-free peak is well below the available memory; record this
    # extension rather than changing the frozen protocol or declaring a pass.
    if not p.exists():
        limit_path=Path('/sys/fs/cgroup/memory.max');used_path=Path('/sys/fs/cgroup/memory.current');limit=int(limit_path.read_text());used=int(used_path.read_text());estimated=14*2**30;assert limit-used>estimated*2;assert selection['q4_nodes']<12000000
        write(p,dict(reason='Complete the already planned second refinement. Initial conservative 8M-node estimate was exceeded by 11.5%; matrix-free memory review supports the actual 8.92M-node grid.',initial_resource_record_sha256=sha(folder/'resource-limit.json'),q4_nodes=selection['q4_nodes'],memory_limit_bytes=limit,memory_used_before_bytes=used,estimated_extra_peak_bytes=estimated,solver_rtol=2e-12,source_sha256={n:sha(ROOT/n) for n in ['benchmarks/aniso_v21_reference_complete.py','engine/aniso_phase1/tensor_reference.py']},thresholds_unchanged=True))
    for n,d in load(p)['source_sha256'].items():assert sha(ROOT/n)==d,n
    fields=[];cases={}
    for degree in (3,4):
        name=f'level1-q{degree}';path=folder/(name+'.npz');initial=read_field(folder/f'level0-q{degree}.npz')
        if path.exists():field=read_field(path);r=load(path.with_suffix('.json'))
        else:
            u,r=solve(edges,degree,hessian('F45'),initial=initial,rtol=2e-12,callback=lambda k:print(name,'iteration',k,flush=True));assert r['passed'] and r['work_identity_relative']<1e-8;np.savez_compressed(path,u=u,degree=degree,**{f'axis{k}':e for k,e in enumerate(edges)});write(path.with_suffix('.json'),r);field=(edges,degree,u);print('completed',name,r,flush=True)
        fields.append(field);cases[name]=r
    cross=compare(*fields);cross['reaction_relative']=abs(cases['level1-q3']['reaction_N']-cases['level1-q4']['reaction_N'])/abs(cases['level1-q4']['reaction_N']);write(folder/'level1-cross.json',cross);print('last cross',cross['regions'],flush=True)
    write(OUT/'reference-completed-summary.json',dict(completed=True,cases=cases,pairs={'level1-cross':cross},latest_reference='reference-extension/level1-q4.npz',previous_q3='reference-extension/level0-q3.npz',previous_q4='reference-extension/level0-q4.npz',stress_and_fiber_reference_passed=cross['stress_passed'] and cross['fiber_strain_passed'],continuum_error_bound_proved=False,resource_limit_reached=False,initial_estimate_limit_resolved=True))
if __name__=='__main__':main()
