"""One registered k_f variation with the frozen 144 basis and empirical R4/R5."""
from pathlib import Path
import argparse,copy,time,gc
import numpy as np
from dataclasses import replace
from .provenance import APP,ROOT,read,write,sha,digest,register,verify,serial_lock
from .spatial_study import reload_level,qcheck
from .spaces import load_selected
from benchmarks.research_sequential_next.spatial import static_solve
from benchmarks.research_reference_next.field_audit import compare_nodal
from benchmarks.research_sequential_next.compare import metric,regions
from benchmarks.research_sequential_next.run import probe_frame
from engine.aniso_phase1.research_sequential.condensation import Condensation
from engine.aniso_phase1.research_sequential_next.segmented import SegmentedModel

def prepare(run):
    run=Path(run);d=read(run/'S2/reference-convergence.json')
    if d['qualified_level']<=4:
        write(run/'S2/generalization-check.json',dict(status='blocked_reference_limited',reason='no qualified new reference; no material solves or retraining'));return
    # Search complete historical JSON text for an explicit stiffness value; this is evidence discovery, not hidden-test certification.
    import re
    hits=[]
    for folder in (APP,):
        for p in folder.rglob('*.json'):
            if 'source' in p.parts or p.stat().st_size>4000000:continue
            if re.search(r'"k_f"\s*:\s*220(?:\.0)?\b',p.read_text()):hits.append(str(p))
    register(run,'S2/material-protocol.json',dict(k_f=220.,multiplier=1.1,mu=10.,lam=20.,fiber_angle_degrees=45.,peak_m=.005,
        cases=['formal','R4',f"R{d['qualified_level']}"],case_count=3,formal_basis_fixed=True,retraining=False,
        original_mass=True,original_Ks=True,initial_guess='same-space main .005 equilibrium',search_matrix='old material rest stiffness ONLY as quasi-Newton search metric; evaluate full new-material nonlinear residual',
        historical_value_matches=hits,hidden=False,history_limit='local parent structured metadata searched; incomplete historical input coverage, no unseen claim',soft_seconds=600,max_rss_GiB=16))

def solve(run,name):
    import warp as wp
    run=Path(run);verify(run);p=read(run/'S2/material-protocol.json')
    if name not in p['cases']:raise ValueError('unregistered material case')
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');start=time.perf_counter()
    if name=='formal':
        r,_=load_selected(read(run/'selected-space.json')['package'])
        with np.load(APP/'S1/candidates/global-snapshot6/static.npz') as z:initial=z['full'].copy()
    else:r,initial,_,_=reload_level(run,int(name[1:]))
    old=r;space=copy.copy(r.parent);space.params=replace(space.params,k_f=p['k_f']);space.A=space.params.A0
    space.signature=digest(dict(parent=old.parent.signature,material=dict(mu=space.params.mu,lam=space.params.lam,k_f=space.params.k_f,fiber=space.params.fiber_direction.tolist()),purpose='registered static generalization',Ks='original unchanged'))
    space.metadata=dict(space.metadata,material_variation_k_f=220.,original_Ks=True)
    r=Condensation(space,old.original_mass,old.original_stiffness)
    if not np.array_equal(r.P,old.P) or not np.array_equal(r.M,old.M) or not np.array_equal(space.Ks,old.parent.Ks):raise ValueError('variation changed physical mass/condensation/stabilization')
    m=SegmentedModel(r,order=7,device='cuda:0',hold=.005);state,stats=static_solve(m,initial,max_seconds=max(1,600-(time.perf_counter()-start)));frame=probe_frame(m,state,[33,7,7]);del m,old;gc.collect()
    checks=qcheck(r,state.q);folder=run/'S2/material'/name;folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'state-fields.npz',q=state.q,full=r.expand(state.q),nodes=space.nodes(r.expand(state.q)),edge0=space.edges[0],edge1=space.edges[1],edge2=space.edges[2],p=space.p,**frame)
    write(folder/'result.json',dict(status='passed_scoped',solve=stats,material=checks,model_space=space.signature,k_f=220.,old_rest_matrix_search_only=True,seconds=time.perf_counter()-start,data_sha256=sha(folder/'state-fields.npz')))
    print('MATERIAL_VARIATION',name,stats['nonlinear_residual_N'],flush=True)

def analyze(run):
    from engine.aniso_phase1.types import AnisotropicMaterialParams
    run=Path(run);p=read(run/'S2/material-protocol.json');names=p['cases'];data={name:np.load(run/'S2/material'/name/'state-fields.npz') for name in names};a,b,c=names
    def nodal(name):
        z=data[name];return (tuple(z[f'edge{i}'] for i in range(3)),int(z['p']),z['nodes'])
    params=AnisotropicMaterialParams(10.,20.,220.,np.array([1.,1.,0.]));adj=compare_nodal(nodal(b),nodal(c),params.A0,params);formal=compare_nodal(nodal(a),nodal(c),params.A0,params)
    fields={k:metric(data[a]['x']-data[a]['X'],data[c]['x']-data[c]['X'],5e-5,.05,w) for k,w in regions(data[c]['X']).items()}
    reaction=metric(read(run/f'S2/material/{a}/result.json')['solve']['reaction_N'],read(run/f'S2/material/{c}/result.json')['solve']['reaction_N'],1e-4,.05)
    passed=all(x['passed'] for reg in formal.values() for x in reg.values()) and all(x['passed'] for x in fields.values()) and reaction['passed']
    reliable=all(x['passed'] and x['absolute']<=.25*x['budget'] for reg in adj.values() for x in reg.values())
    write(run/'S2/generalization-check.json',dict(status='passed_scoped' if passed and reliable else 'material_or_reference_limited',formal_fields_passed=passed,reference_resolved_at_engineering_budget=reliable,
        errors=formal,reference_adjacent=adj,displacement=fields,reaction=reaction,hidden=False,scope='one k_f=220 static .005 F45 variation; fixed basis and original Ks; not uniform material/continuum certification'))
    production,_=load_selected(read(run/'selected-space.json')['package'])
    with np.load(APP/'S1/candidates/global-snapshot6/static.npz') as z:mainfull=z['full'].copy()
    with np.load(run/'S2/R5/data.npz') as z:mainnodes=z['nodes'].copy()
    reference_edges=nodal(c)[:2]
    main_error=compare_nodal((production.parent.edges,production.parent.p,production.parent.nodes(mainfull)),(*reference_edges,mainnodes),production.parent.A,production.parent.params)
    main_reaction=metric(read(APP/'S1/candidates/global-snapshot6/result.json')['solve']['reaction_N'],read(run/'S2/R5/result.json')['solve']['reaction_N'],1e-4,.05)
    write(run/'S2/current-vs-R5.json',dict(status='passed_scoped' if all(v['passed'] for reg in main_error.values() for v in reg.values()) and main_reaction['passed'] else 'limited',errors=main_error,reaction=main_reaction,formal_space_unchanged=True))
    print('GENERALIZATION',passed,reliable,formal['interior'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','solve','analyze']);p.add_argument('--name');p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):
        if a.phase=='solve':solve(a.run,a.name)
        else:globals()[a.phase](a.run)
