"""Seal/check v11 evidence without conflating completed work with accuracy."""
import argparse,hashlib,json,re,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'docs/results/lite-aniso-mainline'
VERSIONS=('v11','v11-reference','v11-reference-q3','v11-exploration')


def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def verify(mapping):
    bad=[f for f,h in mapping.items() if digest(ROOT/f)!=h]
    if bad:raise RuntimeError('changed files: '+str(bad))
    return len(mapping)


def evidence():
    old={f.parent.name:verify(read(f)) for f in sorted(BASE.glob('*/artifact-sha256.json')) if not f.parent.name.startswith('v11')}
    # Historical source is checked inside its sealed zip, not against new code.
    with zipfile.ZipFile(BASE/'v10/source-delivered.zip') as z:
        for f,h in read(BASE/'v10/protocol.json')['source_sha256'].items():assert hashlib.sha256(z.read(f)).hexdigest()==h
    sources={v:verify(read(BASE/v/'protocol.json')['source_sha256']) for v in VERSIONS if v!='v11-exploration'}
    verify(read(BASE/'v11-reference/protocol.json')['input_sha256'])
    assert digest(BASE/'v11-reference/protocol.json')==read(BASE/'v11-reference-q3/protocol.json')['selection_protocol_sha256']
    t=read(BASE/'v11/tests.json');s=read(BASE/'v11/static-summary.json');d=read(BASE/'v11/summary.json');a=read(BASE/'v11/artifact-check.json');fields=read(BASE/'v11/time-field-audit.json')
    assert t['required_checks_passed'] and t['tests_run']==90
    assert len(s['records'])==24 and len(s['beams'])==16 and s['loading_allowed']
    assert all(r['gate']['passed'] and r['solved'] for r in s['records']+s['beams'])
    assert d['completed'] and d['physical_checks_passed'] and a['passed'] and len(a['snapshots'])==64
    assert len(d['checks'])==16 and sum(c['steps'] for c in d['checks'].values())==30000
    assert d['snapshot_count']==64 and d['history_closure_max']<1e-12 and d['particle_trial_commit_max']<1e-12
    reference_checks=0;reference_cases=0
    for v in ('v11-reference','v11-reference-q3'):
        r=read(BASE/v/'runs.json');tt=read(BASE/v/'tests.json');assert r['completed'] and all(x['passed'] for x in r['records']) and tt['passed']
        reference_cases+=len(r['records']);reference_checks+=tt['tests']
    rr=read(BASE/'v11-reference/summary.json');qr=read(BASE/'v11-reference-q3/summary.json')
    refined=read(BASE/'v11/local-reference-comparison.json');assert len(refined['records'])==12
    refined_q3=read(BASE/'v11/local-q3-reference-comparison.json');assert len(refined_q3['records'])==12
    assert qr['completed'] and rr['completed']
    mode=read(BASE/'v11/mode-boundary-diagnostic.json');assert all(r['selective_energy_min_on_material_null']>0 for r in mode['mode_checks'])
    boundary=next(r for r in refined['records'] if r['grid']==33 and r['mode']=='selective_patch')
    return dict(work_complete=True,implementation_checks_passed=True,overall_accuracy_accepted=False,
        regression_checks=90,reference_checks=reference_checks,reference_solutions=reference_cases,
        static_tensile_cases=24,static_beam_cases=16,boundary_sensitivity_cases=3,
        dynamic_trajectories=16,dynamic_steps=30000,independent_snapshots=64,
        history_closure_max=d['history_closure_max'],independent_state_error_max=a['max_state_error'],
        independent_grip_force_error_N=a['max_force_error_N'],independent_reference_error=a['max_reference_error'],
        candidate_static_gate=s['candidate_static_gate'],F45_overconstraint_reduced=s['F45_improved'],
        F45_g33_vs_local_Q2=boundary,F45_g33_vs_local_Q3=next(r for r in refined_q3['records'] if r['grid']==33 and r['mode']=='selective_patch'),full_slow_loading_completed=True,dynamic_physical_checks_passed=True,
        time_threshold_passed=d['time_threshold_passed'],reliable_reaction_trend=d['reliable_reaction_trend'],
        full_trajectory_F_P_time_passed=fields['all_passed'],
        reference_full_stress_certified=rr['full_stress_reference_certified'] and qr['full_stress_reference_certified'],
        old_artifact_files_verified=old,frozen_source_files_verified=sources,
        default_changed=False,device='cpu',precision='float64',cuda='not_run',
        report='docs/ANISO_LITE_SELECTIVE_STABILIZATION_ZH.md')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('seal','check'));args=p.parse_args();r=evidence()
    if args.action=='seal':
        if any((BASE/v/'artifact-sha256.json').exists() for v in VERSIONS):raise RuntimeError('preserve manifests')
        files=set()
        for folder in ('engine','utils','demos','benchmarks','tests'):files.update((ROOT/folder).rglob('*.py'))
        files.update(ROOT.glob('*.md'));files.update((ROOT/'docs').glob('*.md'));files.update(ROOT/f for f in ('pyproject.toml','uv.lock','.python-version'))
        archive=BASE/'v11/source-delivered.zip'
        with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
            for f in sorted(files):z.write(f,f.relative_to(ROOT))
        r.update(sealed_at=datetime.now(timezone.utc).isoformat(),source_archive_files=len(files),source_archive_sha256=digest(archive))
        write(BASE/'v11/completion-check.json',r)
        for v in VERSIONS:write(BASE/v/'artifact-sha256.json',{str(f.relative_to(ROOT)):digest(f) for f in sorted((BASE/v).rglob('*')) if f.is_file() and f.name!='artifact-sha256.json'})
    counts={v:verify(read(BASE/v/'artifact-sha256.json')) for v in VERSIONS}
    with zipfile.ZipFile(BASE/'v11/source-delivered.zip') as z:
        for f in z.namelist():assert hashlib.sha256(z.read(f)).hexdigest()==digest(ROOT/f),f
    for f in ('README.md','RUNNING_RESTORED.md','docs/ANISO_LITE_SELECTIVE_STABILIZATION_ZH.md'):
        path=ROOT/f;text=path.read_text();assert '<!--' not in text if f.endswith('SELECTIVE_STABILIZATION_ZH.md') else True
        for link in re.findall(r'\]\(([^)]+)\)',text):
            if '://' not in link and not link.startswith('#'):assert (path.parent/link.split('#')[0]).exists(),link
    print(json.dumps(dict(work_complete=True,overall_accuracy_accepted=False,manifest_files=counts,time_threshold_passed=r['time_threshold_passed'],reliable_reaction_trend=r['reliable_reaction_trend']),indent=2))


if __name__=='__main__':main()
