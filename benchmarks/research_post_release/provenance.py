"""Explicit sealed parent, content identities and data-disk-only research outputs."""
from pathlib import Path
from datetime import datetime, timezone
import shutil
from benchmarks.research_sequential_next.provenance import (
    ROOT, read, write, sha, digest, check, utc, serial_lock, environment, resources)

PARENT = ROOT/'docs/results/sequential-next/20260930T122305Z-next-practical'
PARENT_RELEASE_SHA = 'ae8a1d2151056ab5472f8287d9bfbb950c78cb59a69f3b0f7763f0b3250d5296'
PLAN = ROOT/'docs/MPM_LITE_POST_RELEASE_FIVE_PRIORITY_PLAN_20261001_ZH.md'


def audit_parent(full=False):
    if sha(PARENT/'release.json') != PARENT_RELEASE_SHA:
        raise ValueError('sealed parent release changed')
    release = read(PARENT/'release.json')
    for key in ('mathematical_package', 'numerical_sources', 'artifacts'):
        entry = release[key]
        if sha(PARENT/entry['path']) != entry['sha256']:
            raise ValueError('sealed parent index mismatch: '+key)
    if sha(PARENT/'baseline-lock.json') != release['baseline_sha256']:
        raise ValueError('sealed parent baseline changed')
    base = read(PARENT/'baseline-lock.json')
    sources = read(PARENT/'final-source-sha256.json')
    counts = dict(parent_sources=check(ROOT, sources), historical_sources=check(ROOT, base['old_source_sha256']),
                  inputs=check(ROOT, base['input_sha256']))
    if full:
        counts.update(parent_source_snapshot=check(PARENT/'final-source', sources),
                      parent_artifacts=check(PARENT, read(PARENT/'artifact-sha256.json')))
    return release, base, counts


def source_files():
    # Only the live numerical adapters participate in case continuation. Study
    # drivers are additionally frozen by register(); renderers are output-only.
    result = {}
    for folder in ('engine/aniso_phase1/research_post_release', 'benchmarks/research_post_release'):
        for p in sorted((ROOT/folder).glob('*.py')):
            if folder.startswith('benchmarks') and p.name not in ('__init__.py','provenance.py','config.py','run.py'):
                continue
            result[str(p.relative_to(ROOT))] = sha(p)
    return result


def snapshot(folder, entries):
    for name, expected in entries.items():
        source = ROOT/name
        if sha(source) != expected: raise ValueError('source changed during snapshot')
        target = Path(folder)/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def register(run, name, payload):
    path = Path(run)/name
    if path.exists(): raise ValueError('study already registered: '+str(path))
    sources = source_files()
    for p in (ROOT/'benchmarks/research_post_release').glob('*.py'):
        if p.name != 'visualize.py': sources[str(p.relative_to(ROOT))] = sha(p)
    value = dict(payload, utc=utc(), source_sha256=sources, parent_release_sha256=PARENT_RELEASE_SHA)
    snapshot(path.parent/(path.stem+'-source'), sources)
    write(path, value)
    return value


def freeze(application_release=None):
    release, base, counts = audit_parent(full=True)
    application=None
    if application_release is not None:
        application_release=Path(application_release).resolve()
        published=read(application_release/'release.json')
        if published.get('schema')!='post-release-practical-v1' or published.get('parent_release_sha256')!=PARENT_RELEASE_SHA:
            raise ValueError('unsupported application release lineage')
        for key in ('sources','artifacts','qualification'):
            entry=published[key]
            if sha(application_release/entry['path'])!=entry['sha256']:raise ValueError('application release index changed')
        check(ROOT,read(application_release/published['sources']['path']))
        check(application_release,read(application_release/published['artifacts']['path']))
        application=dict(path=str(application_release),release_sha256=sha(application_release/'release.json'),
                         qualified_material=published['qualification'])
    inventory = {str(p.relative_to(ROOT)) for d in ('engine','benchmarks','tests','demos','utils')
                 for p in (ROOT/d).rglob('*.py') if 'research_post_release' not in p.parts}
    known = set(base['old_source_sha256']) | set(read(PARENT/'final-source-sha256.json'))
    if inventory-known: raise ValueError('unaccounted implementation: '+str(sorted(inventory-known)))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-post-release'
    physical = Path('/root/autodl-tmp/mpm-lite-post-release')/stamp
    physical.mkdir(parents=True, exist_ok=False)
    run = ROOT/'docs/results/post-release'/stamp
    run.parent.mkdir(parents=True, exist_ok=True); run.symlink_to(physical, target_is_directory=True)
    value = dict(schema='post-release-parent-lock-v1', utc=utc(), parent_path=str(PARENT),
                 parent_release_sha256=PARENT_RELEASE_SHA, parent_release=release, counts=counts,
                 plan_path=str(PLAN.relative_to(ROOT)), plan_sha256=sha(PLAN),
                 energy_scale_J=base['energy_scale_J'], initial_source_sha256=source_files(),application_parent=application)
    write(run/'parent-release-lock.json', value)
    write(run/'parent-release-lock-sha256.json', dict(sha256=sha(run/'parent-release-lock.json')))
    write(run/'environment.json', environment())
    if application is not None:
        target=run/'S4/qualification.json';target.parent.mkdir(parents=True,exist_ok=True)
        source=Path(application['path'])/application['qualified_material']['path']
        target.write_bytes(source.read_bytes())
    write(run/'S0/result.json', dict(status='passed_scoped', counts=counts, historical_python=len(inventory),
          resources=resources(), migration='not_triggered_at_start: system free space above 5 GiB',
          parent_unchanged=True, output_physical_path=str(physical)))
    Path('/tmp/mpm-post-run.txt').write_text(str(run)+'\n')
    return run


def verify(run):
    run = Path(run)
    if sha(run/'parent-release-lock.json') != read(run/'parent-release-lock-sha256.json')['sha256']:
        raise ValueError('changed parent release lock')
    value = read(run/'parent-release-lock.json')
    if value['parent_release_sha256'] != PARENT_RELEASE_SHA or Path(value['parent_path']) != PARENT:
        raise ValueError('wrong parent release')
    audit_parent()
    return value


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-release',type=Path,required=True,help='verified sealed post-release application to fork')
    args=parser.parse_args()
    with serial_lock():
        output=freeze(args.from_release)
        print(output,flush=True)
