"""Coverage/threshold audit of actually executed quadrature experiments.

Reads numerical outputs, not just prior pass manifests. Gate failure is a
negative research result: it does not certify a full production MPM solver.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from benchmarks.aniso_quadrature_validation import save


def audit(base):
    def read(path):return json.loads((base/path).read_text())
    items=[]
    def item(name,status,evidence,detail):items.append(dict(requirement=name,status=status,evidence=evidence,detail=detail))
    static=[]
    for directory,grid in [('aligned-v1',17),('aligned-v1-fine',33)]:
        r=read(f'{directory}/grid{grid}.json');static.append(r)
        assert r['grid']==grid and r['convergence']['matrix_relative']<1e-8
        assert r['convergence']['probe_action_max']<1e-8
        assert r['convergence']['load_relative']<1e-8
        assert r['convergence']['constraint_rank6']==r['convergence']['constraint_rank8']
        assert r['convergence']['clamp_density_solution_relative']<1e-7
        rows={a['rule']:a for a in r['records']}
        assert {'gauss2','gauss3','gauss4','gauss5','group4x8','particle2','particle4'}<=rows.keys()
        for a in rows.values():
            assert abs(a['volume']-.0078125)<1e-12 and a['weight_min']>0
            assert a['free_residual']<1e-7 and a['clamp_max']<1e-10
            assert a['pcg_info']==0 and a['pcg_true_residual']<2e-7
            assert abs(a['reaction'][1]-1e-4)<1e-10
            probes=read(f"{directory}/{a['name']}-probes.json")
            assert {'bend_y','bend_z','twist','stretch','shear','random_2','worst_mode','reference_solution'}<=probes.keys()
            with np.load(base/directory/(a['name']+'-mode.npz')) as mode:assert np.isfinite(mode['displacement']).all()
        assert rows['gauss3']['rho_min']>.9 and rows['gauss3']['rho_max']<1.1
        assert abs(rows['gauss3']['tip_relative_to_dense'])<.05
        assert rows['gauss2']['rho_min']<.5 and rows['group4x8']['rho_min']<.05
    assert abs(static[1]['reference_relative_to_q1'])<abs(static[0]['reference_relative_to_q1'])
    item('A1-A5 / B1-B3','passed','aligned-v1; aligned-v1-fine','Two-grid physical-domain reference, loads, clamps, generalized modes, probes and Q1 trend verified.')
    adaptive=read('adaptive-v1/grid17.json');rules={a['rule']:a for a in adaptive['records']}
    assert {'adaptive2','adaptive3'}<=rules.keys() and rules['adaptive2']['rho_min']<.5 and rules['adaptive3']['rho_min']>.9
    assert next(a for a in read('particle-refinement-v1/grid17.json')['records'] if a['rule']=='particle8')['samples']==16384
    item('C1-C2 / E2','passed','adaptive-v1; particle-refinement-v1','Local promotion controls and fixed-volume PPC sweep executed; gauss3 retained, weak alternatives rejected.')
    finite=read('nonlinear-v2/nonlinear.json')
    assert {(a['field'],a['state']) for a in finite}=={(f,s) for f in ('uniform','crossed','smooth') for s in ('bend','stretch_shear')}
    for a in finite:
        assert a['min_det']>0 and a['host_force_relative']<1e-8 and a['symmetry_relative']<1e-8
        assert min(f['energy_gradient_relative'] for f in a['finite_differences'])<1e-5
        assert min(f['tangent_relative'] for f in a['finite_differences'])<1e-5
        assert a['rotation_energy_relative']<1e-8 and a['rotation_force_relative']<1e-8
        if a['rule'] in ('gauss3','gauss4'):
            assert max(abs(a['energy_relative']),a['free_force_relative'],a['action_relative'])<.05
    item('D1-D3','passed','nonlinear-v2; final-tests.json','Same-state material rules, real sparse node ordering, production kernels, derivatives, symmetry and frozen objectivity checked.')
    dyn=read('dynamics-v2/dynamics.json');slow=read('slow-cycle-v2/dynamics.json')
    assert len(dyn)==6 and len(slow)==4
    release=[a for a in dyn if a['scene']=='release'];assert {a['dt'] for a in release}=={.005,.0025}
    assert max(a['physical_time'] for a in release)-min(a['physical_time'] for a in release)<1e-12
    for a in dyn+slow:
        assert len(a['rows'])==a['steps']
        assert all(x['min_det']>0 and np.isfinite(x['mechanical']) and x['clamp_max']<1e-8 for x in a['rows'])
    item('D4 frozen dynamics','passed','dynamics-v2; slow-cycle-v2','Release, equal-time dt pair and slow loading pair stable; scope remains frozen material reference.')
    transfer=read('transfer-v1/results.json')+read('transfer-fine-v1/results.json')
    assert {(r['grid'],r['ppc_axis']) for r in transfer}=={(17,2),(17,4),(33,2)}
    for r in transfer:
        assert r['mass']['negative_lumped_nodes']>0 and r['mass']['lite_min_lumped']>0
        for a in r['records']:
            assert a['actual_vs_particle_stencil']<1e-9 and a['actual_position_vs_lite']<1e-12 and a['min_det']>0
            if a['field'] in ('affine','quadratic_bend'):assert a['gradient_relative_mismatch']<1e-9
        if r['grid']==17:
            assert next(a for a in r['records'] if a['field']=='solved_mls')['gradient_relative_mismatch']>.1
    assert transfer[0]['mass']['consistent_mass_soft_modes']>0
    item('D4 production promotion gate','failed_candidate','transfer-v1; transfer-fine-v1','Direct material-only replacement is incompatible with Lite history updates; naive MLS mass lumping has negative masses. Positive controls pass.')
    rebuilt=read('rebuild-v1/results.json');assert {r['grid'] for r in rebuilt}=={17,33}
    for r in rebuilt:
        assert {a['angle'] for a in r['records']}=={0,45,90}
        for a in r['records']:
            assert abs(a['exact_objectivity'])<1e-8 and abs(a['high_rule_history_convergence'])<.01
            assert all(v['min_det']>0 for v in a['readings'].values())
    item('D4 rule/history separation','passed_with_scope_limit','rebuild-v1','Exact motion plus refitted spatial history, 3/4/5 carried material rules on identical states; no arbitrary deformed Eulerian clipping claim.')
    repeats=read('solve-cost-v1/repeats.json');assert len(repeats)==3
    for trial in repeats:
        assert len(trial)==6
        for r in trial:assert r['build_seconds']>0 and r['whole_run_seconds']>0 and r['device_array_bytes']>0
    costs=read('nonlinear-v2/kernel-cost.json');assert {'gauss3','gauss4','particle2','particle4','particle8'}<={r['rule'] for r in costs}
    memory=read('nonlinear-v2/memory-observed.json');assert memory['gpu_memory_samples']>0 and memory['cpu_process_peak_rss_mib']>0
    item('E1-E3 comparable costs','passed_with_scope_limit','solve-cost-v1; nonlinear-v2','Repeated same-field solves, construction, iterations and CPU/GPU memory recorded; measured scope explicitly excludes MPM rebuild/transfer.')
    item('E3 full production step / E4 decision','gated_not_run','ANISO_QUADRATURE_PRODUCTION_GATE_ZH.md; plan sections 1,10','No eligible production candidate after D4 gate failure. Do not benchmark an inconsistent hybrid as a correct full MPM method. Retain reference tools; reject direct promotion.')
    tests=read('final-tests.json');assert tests['success'] and tests['tests']==25 and tests['failures']==tests['errors']==tests['skipped']==0
    viewer=read('viser-smoke.json');assert viewer['http_status']==200 and viewer['exit_code']==0
    item('Artifacts, tests, visualization','passed','final-tests.json; viser-smoke.json; reports; CLI files','25 regressions, data exports and runnable static viewer. No browser screenshot claimed.')
    return dict(validation_work_complete=True,production_candidate_accepted=False,full_mpm_speedup_proven=False,
        completion_basis='All applicable verification stages executed; dependent full-MPM promotion/performance stopped on an observed correctness gate failure, as permitted by plan sections 1 and 10.',items=items)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('docs/results/quadrature-validation'));args=p.parse_args()
    result=audit(args.data);save(args.data/'completion-audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
