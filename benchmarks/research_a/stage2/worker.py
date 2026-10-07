"""Isolated workers: every execution imports the independently sealed sources."""
import argparse
import json
from pathlib import Path
import resource
from time import monotonic
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from .protocol import require_frozen, write_json, sha, save_array, ROOT, PARENT_REL, TRUSTED_PARENT, PARENT_SHA
from engine.aniso_phase1.research_a.stage2.cases import registered_cases, supports
from engine.aniso_phase1.research_a.stage2.problem import Problem
from engine.aniso_phase1.research_a.stage2.adaptive import enrich
from engine.aniso_phase1.research_a.stage2.reference import estimate, pilot_edges, solve, compare


def peak(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def construction(run):
    protocol=require_frozen(run)
    from engine.aniso_phase1.research_d.frozen_inputs import load_frozen_inputs
    space,inertia,state,verified=load_frozen_inputs(ROOT/PARENT_REL,ROOT,trusted_data_root=TRUSTED_PARENT,expected_sha256=PARENT_SHA)
    write_json(run/'independent-parent-load.json',dict(**verified,source_checkout=str(ROOT),state_shape=list(state.q.shape)))
    del space,inertia,state
    records={}
    for name,case in registered_cases().items():
        t=monotonic(); problem=Problem.from_archive(case,2,'construction')
        field,equilibrium=problem.equilibrate(np.empty((problem.n,0)))
        records[name]=dict(case=case.as_dict(),construction=problem.construction,equilibrium=equilibrium,
                          invariants=problem.invariant_errors(np.empty((problem.n,0))),seconds=monotonic()-t,
                          supports={family:len(supports(case,family)) for family in ('v22-original','v22-overlap','wide-overlap','fiber-rect')},
                          role='case construction; not hidden strategy evaluation',A9_completed=False)
        save_array(run,f'construction/{name}.npz',**field,degree=2,**{f'axis{k}':e for k,e in enumerate(problem.edges)})
        print('CONSTRUCTED',name,equilibrium['energy_J'],flush=True)
        del problem,field
    write_json(run/'case-construction.json',dict(cases=records,all_nine_constructed=True,peak_rss_bytes=peak(),production_mesh=False))


def reference(run):
    require_frozen(run); case=registered_cases()['F45']; edges=pilot_edges(case)
    old=ROOT/'docs/results/lite-aniso-mainline/v22/reference/level1-q4.npz'
    # Read topology only. The known v22 field is not a new hidden reference.
    with np.load(old,allow_pickle=False) as z: old_edges=[z[f'axis{k}'].copy() for k in range(3)]
    focused=[]
    for k,e in enumerate(old_edges):
        eligible=np.flatnonzero((e[:-1]>=.25)&(e[1:]<=.75)) if k==0 else np.arange(len(e)-1)
        ids=np.union1d(eligible[:1],eligible[-1:]); focused.append(np.union1d(e,(e[:-1]+e[1:])[ids]/2))
    design=dict(physical_domain=case.box,continuity='conforming Cartesian Qp; shared nodes; no hanging nodes',
       routes=dict(local_h=dict(kind='nearest transition/boundary axis strips; tensor closure explicitly counted',levels=[estimate(old_edges,4),estimate(focused,4)],status='preflight only'),
           local_p=dict(kind='local subdomain diagnostic requires trace constraints; not independently global-certifying',status='not implemented; no unsupported continuity claim'),
           global_control=dict(kind='fixed full physical domain, p=2/3/4 on construction grid',levels=[estimate(edges,p) for p in (2,3,4)],status='bounded executed p feasibility test')),
       limits=dict(peak_GiB=24,nodes=14000000,seconds=3600),independent_final_reference_available=False,
       scope='initial feasibility only; full final certification not claimed',old_reference_identity=sha(old),
       large_reference_status='not started without target-scale timing calibration and resource window')
    write_json(run/'reference-design.json',design)
    records={}; fields={}; comparisons={}
    for p in (2,3,4):
        u,r=solve(case,edges,p)
        fields[p]=(edges,p,u); records[str(p)]=r
        path=save_array(run,f'reference-pilot/q{p}.npz',u=u,degree=p,**{f'axis{k}':e for k,e in enumerate(edges)})
        r['field_sha256']=sha(path)
        if p>2: comparisons[f'q{p-1}-q{p}']=compare(case,fields[p-1],fields[p])
        print('REFERENCE',p,r['seconds'],r['relative_residual'],flush=True)
    tighter, tight=solve(case,edges,4,rtol=2e-10)
    numerical=compare(case,fields[4],(edges,4,tighter))
    reaction_diffs=[]
    for a,b in ((2,3),(3,4)):
        ra=np.asarray(records[str(a)]['reaction_vector_N']);rb=np.asarray(records[str(b)]['reaction_vector_N'])
        reaction_diffs.append(float(la.norm(ra-rb)/max(la.norm(rb),1e-5)))
    region_status={}
    for region in ('global','grip','interior','deep_interior'):
        vals=[comparisons[key]['regions'][region] for key in ('q2-q3','q3-q4')]
        region_status[region]=dict(stress_difference=[v['stress_relative'] for v in vals],fiber_difference=[v['fiber_strain_relative'] for v in vals],
            decreasing=bool(vals[1]['stress_relative']<vals[0]['stress_relative'] and vals[1]['fiber_strain_relative']<vals[0]['fiber_strain_relative']),
            threshold_passed=vals[-1]['stress_engineering_passed'] and vals[-1]['fiber_engineering_passed'],certified=False)
    write_json(run/'reference-pilot.json',dict(levels=records,comparisons=comparisons,regions=region_status,reaction_differences=reaction_diffs,
       tightened_solve=tight,iteration_difference=numerical,linear_quadrature='p+1 exact polynomial integration; independent operator equivalence tested',
       numerical_quadrature_refinement_executed=False,peak_rss_bytes=peak(),final_reference_certified=False,
       reason='Feasibility probe: no independent final holdout and no complete boundary/integration uncertainty accounting'))


def candidate(run,name):
    protocol=require_frozen(run)
    specs={s['name']:s for s in protocol['candidates']}
    if name not in specs: raise ValueError('Unregistered candidate')
    spec=specs[name]; case=registered_cases()[spec['case']]
    folder=run/'candidates'/name; folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'config.json',dict(**spec,case_sha256=case.signature,protocol_sha256=sha(run/'protocol.json'),model='linear quadratic',mesh='production'))
    t=monotonic()
    problem=Problem.from_archive(case,spec['degree'])
    resource_plan=estimate(problem.edges,spec['degree'],225+spec['budget'])
    if not resource_plan['within_budget']: raise RuntimeError('Candidate predicted memory exceeds budget')
    write_json(folder/'resource-preflight.json',dict(**resource_plan,construction_seconds=monotonic()-t,construction_peak_rss_bytes=peak()))
    definitions=supports(case,spec['family'])
    sequence=None
    if spec['sequence']:
        parent=run/'candidates'/spec['sequence']
        if not (parent/'summary.json').exists(): raise ValueError('Fixed-selection source must finish first')
        sequence=[json.loads((parent/f'round{i}.json').read_text())['chosen_patches'] for i in range(1,7)]
    def publish(record,field,raw,transform):
        step=record['round']; arr=save_array(run,f'candidates/{name}/round{step}.npz',**field,degree=spec['degree'],**{f'axis{k}':e for k,e in enumerate(problem.edges)})
        rawpath=run/'arrays/candidates'/name/f'raw{step}.npz'; sp.save_npz(rawpath,raw)
        trans=save_array(run,f'candidates/{name}/transform{step}.npz',transform=transform)
        record.update(field_sha256=sha(arr),raw_sha256=sha(rawpath),transform_sha256=sha(trans),peak_rss_bytes=peak())
        write_json(folder/f'round{step}.json',record)
        print('ROUND',name,step,record['scalar_local_dofs'],record['equilibrium']['energy_J'],record['cost'],flush=True)
    try:
        if spec['mode']=='reuse-F45':
            if spec['case']=='tall-shear': raise ValueError('F45 basis cannot be reused on changed geometry')
            original=next(s['name'] for s in protocol['candidates'] if s['case']=='F45' and s['budget']==144)
            W=load_basis(run,original);field,equilibrium=problem.equilibrate(W)
            raw=sp.csr_matrix(W);transform=np.eye(W.shape[1])
            rec=dict(round=6,scalar_local_dofs=W.shape[1],equilibrium=equilibrium,cost=dict(balance_seconds=monotonic()-t),mode='reuse-F45',source_candidate=original)
            publish(rec,field,raw,transform);reconstruction=0.
        else:
            result=enrich(problem,definitions,spec['score'],6,spec['budget']//18,spec['budget'],publish,fixed_sequence=sequence)
            reconstruction=result['basis_reconstruction_relative']
        write_json(folder/'summary.json',dict(completed=True,seconds=monotonic()-t,peak_rss_bytes=peak(),
            basis_reconstruction_relative=reconstruction,field_sha256=sha(run/'arrays/candidates'/name/'round6.npz'),
            all_static_passed=True,nonlinear_operator_audited=False,spatial_certified=False,mode=spec['mode'],
            reference_field_access=False,automatic_parent_replacement=False))
    except Exception as exc:
        write_json(folder/'failure.json',dict(completed=False,reason=str(exc),seconds=monotonic()-t,peak_rss_bytes=peak(),preserved_round_checkpoints=True))
        raise


def load_basis(run,name):
    folder=run/'arrays/candidates'/name
    raw=sp.load_npz(folder/'raw6.npz')
    with np.load(folder/'transform6.npz',allow_pickle=False) as z: transform=z['transform']
    return raw@transform


def audit(run,name):
    protocol=require_frozen(run); folder=run/'candidates'/name
    spec=json.loads((folder/'config.json').read_text()); summary=json.loads((folder/'summary.json').read_text())
    path=run/'arrays/candidates'/name/'round6.npz'
    if sha(path)!=summary['field_sha256']: raise ValueError('Candidate changed')
    case=registered_cases()[spec['case']];p=Problem.from_archive(case,spec['degree']);W=load_basis(run,name)
    from engine.aniso_phase1.research_a.export_adapter import FixedSpace
    from engine.aniso_phase1.tensor_metrics import evaluate_gradient, quadrature_axis
    space=FixedSpace(p,W)
    with np.load(path,allow_pickle=False) as z:q=np.vstack((p.Q.T@(z['y']-p.lift),z['local_coefficients']))
    rng=np.random.default_rng(protocol['seed']);v=rng.normal(size=q.shape);v/=la.norm(v)
    start=monotonic();base=space.response(q,v,order=6);curve=[]
    for eps in (3e-6,1e-6,3e-7):
        plus=space.response(q+eps*v,order=6);minus=space.response(q-eps*v,order=6)
        derivative=(plus['energy_J']-minus['energy_J'])/(2*eps)
        curve.append(dict(epsilon=eps,energy_derivative_absolute_J=abs(derivative-np.sum(base['force']*v)),
            tangent_relative=float(la.norm((plus['force']-minus['force'])/(2*eps)-base['tangent_action'])/max(la.norm(base['tangent_action']),1e-8))))
    Y=space.total_coefficients(q);rotation=la.expm(np.array([[0,-.6,.2],[.6,0,-.1],[-.2,.1,0]]));Z=Y@rotation.T;Z[:len(p.carrier_X)]+=[.013,-.021,.008]
    rotated=space.potential.evaluate(Z,order=6);original=space.potential.evaluate(Y,order=6)
    er=abs(rotated['U']-original['U']);fr=float(la.norm(rotated['force']-original['force']@rotation.T)/max(la.norm(original['force']),1e-8))
    fine=space.response(q,v,order=7)
    quadrature=dict(orders=[6,7],energy_relative=abs(fine['energy_J']-base['energy_J'])/max(abs(fine['energy_J']),1e-8),
        force_relative=float(la.norm(fine['force']-base['force'])/max(la.norm(fine['force']),1e-6)),
        tangent_relative=float(la.norm(fine['tangent_action']-base['tangent_action'])/max(la.norm(fine['tangent_action']),1e-6)))
    minimum=1.;points=[quadrature_axis(e,6)[0] for e in p.edges];u=space.nodal_displacement(q)
    for i in range(0,len(points[0]),12):
        L=evaluate_gradient((p.edges,p.degree,u),[points[0][i:i+12],*points[1:]])
        minimum=min(minimum,float(np.linalg.det(np.eye(3)+L).min()))
    gates=protocol['implementation_gates']
    passed=bool(all(c['energy_derivative_absolute_J']<=gates['energy_derivative_absolute_J'] and c['tangent_relative']<=gates['tangent_relative'] for c in curve)
        and er<=gates['rotation_energy_absolute_J'] and fr<=gates['rotation_force_relative'] and minimum>=gates['min_detF'])
    write_json(folder/'nonlinear-audit.json',dict(passed=passed,curve=curve,rotation_energy_error_J=er,rotation_force_relative=fr,min_detF=minimum,
        material_quadrature=quadrature,bounded_material_reference_passed=bool(max(quadrature['energy_relative'],quadrature['force_relative'],quadrature['tangent_relative'])<.001),
        state_coverage='this candidate static snapshot and seeded direction only',seconds=monotonic()-start,peak_rss_bytes=peak(),linear_spatial_accuracy=False))
    if not passed:raise RuntimeError('Nonlinear operator audit failed; research candidate cannot be recommended')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['construction','reference','candidate','audit']);parser.add_argument('--run',required=True);parser.add_argument('--name')
    args=parser.parse_args();run=Path(args.run)
    if args.action=='construction':construction(run)
    elif args.action=='reference':reference(run)
    elif args.action=='candidate':candidate(run,args.name)
    else:audit(run,args.name)

if __name__=='__main__':main()
