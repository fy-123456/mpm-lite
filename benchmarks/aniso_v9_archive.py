"""Seal or verify the completed v9 delivery without conflating accuracy gates.

Run only after every simulation/analysis log is closed. Existing manifests are
never overwritten. All paths in manifests are relative to the repository root.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'docs/results/lite-aniso-mainline'
OUTPUTS = ('v9', 'v9-space', 'v9-dynamic-space', 'v9-dynamic-refined')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_map(mapping):
    bad = [name for name, digest in mapping.items() if sha(ROOT/name) != digest]
    if bad:
        raise RuntimeError('changed files: '+str(bad))
    return len(mapping)


def evidence():
    old = {f'v{v}': verify_map(read(BASE/f'v{v}'/'artifact-sha256.json')) for v in range(1, 9)}
    source = {}
    for name in OUTPUTS:
        protocol = read(BASE/name/'protocol.json')
        source[name] = verify_map(protocol['source_sha256'])
    verify_map(read(BASE/'v9-dynamic-refined/protocol.json')['reference_sha256'])
    with zipfile.ZipFile(BASE/'v8/source-before-v9.zip') as archive:
        for name, digest in read(BASE/'v8/protocol.json')['source_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    t = read(BASE/'v9/tests.json')
    s = read(BASE/'v9/summary.json')
    q = read(BASE/'v9-space/summary.json')
    qt = read(BASE/'v9-space/tests.json')
    d = read(BASE/'v9-dynamic-refined/summary.json')
    a = read(BASE/'v9/artifact-check.json')
    da = read(BASE/'v9-dynamic-refined/artifact-check.json')
    assert t['tests_run'] == 69 and t['required_checks_passed']
    assert qt['run'] == 3 and qt['passed']
    assert s['run_completed'] and s['all_frozen_numeric_checks_passed']
    assert q['completed'] and q['all_linear_checks_passed'] and q['references_solved']
    assert d['completed'] and d['physical_checks_passed']
    assert a['passed'] and da['passed']
    assert len(a['snapshots']) == 20 and len(da['snapshots']) == 96
    return dict(work_complete=True, implementation_checks_passed=True, overall_accuracy_accepted=False,
        old_archives_verified=old, frozen_source_files_verified=source, v8_source_archive_verified=True,
        regression_tests=69, static_operator_tests=3, skipped=0,
        full_time_and_anchor_steps=8000, static_cases=60, static_references=8, massless_beam_cases=8,
        dynamic_trajectories=24, dynamic_steps=5600, independent_new_snapshots=116,
        time_snapshot_oracle_max=a['max_oracle_error'], dynamic_snapshot_oracle_max=da['max_oracle_error'],
        acceptance=dict(full_time_numeric_gate=s['time_threshold_passed'],
            full_time_reliable_trend=s['reliable_time_trend_screen_passed'],
            spatial_reference_resolved=q['reference_resolution_passed'],
            spatial_five_percent_screen=q['five_percent_screen_passed'],
            unique_static_displacement=q['unique_static_displacement_verified'],
            dynamic_time_separation=d['time_separation_passed'],
            dynamic_space_convergence=d['space_convergence_established']),
        default_changed=False, device='cpu', precision='float64', cuda='not_run',
        report='docs/ANISO_LITE_PROJECTED_HISTORY_ZH.md')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('seal', 'check'))
    args = parser.parse_args()
    result = evidence()
    if args.action == 'seal':
        if any((BASE/name/'artifact-sha256.json').exists() for name in OUTPUTS):
            raise RuntimeError('preserve existing manifests; use check')
        archive = BASE/'v9/source-delivered.zip'
        files = set()
        for folder in ('engine', 'utils', 'demos', 'benchmarks', 'tests'):
            files.update((ROOT/folder).rglob('*.py'))
        files.update((ROOT/'docs').glob('*.md'))
        files.update(ROOT.glob('*.md'))
        files.update(ROOT/name for name in ('pyproject.toml', 'uv.lock', '.python-version'))
        with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as out:
            for file in sorted(files):
                out.write(file, file.relative_to(ROOT))
        result.update(sealed_at=datetime.now(timezone.utc).isoformat(), source_archive_files=len(files),
            source_archive_sha256=sha(archive))
        (BASE/'v9/completion-check.json').write_text(json.dumps(result, indent=2)+'\n')
        for name in OUTPUTS:
            directory = BASE/name
            mapping = {str(f.relative_to(ROOT)): sha(f) for f in sorted(directory.rglob('*'))
                       if f.is_file() and f.name != 'artifact-sha256.json'}
            (directory/'artifact-sha256.json').write_text(json.dumps(mapping, indent=2)+'\n')
    manifests = {name: verify_map(read(BASE/name/'artifact-sha256.json')) for name in OUTPUTS}
    completion = read(BASE/'v9/completion-check.json')
    assert completion['source_archive_sha256'] == sha(BASE/'v9/source-delivered.zip')
    with zipfile.ZipFile(BASE/'v9/source-delivered.zip') as archive:
        for name in archive.namelist():
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha(ROOT/name), name
    print(json.dumps(dict(work_complete=True, new_manifest_files_verified=manifests,
        implementation_checks_passed=True, overall_accuracy_accepted=False,
        accuracy_gates=result['acceptance']), indent=2))


if __name__ == '__main__':
    main()
