"""Verify that the post-run support rejection adds no branch on formal states."""
import difflib, hashlib, zipfile
from benchmarks.aniso_compatible_history import OUT, ROOT, load, write, sources


def main():
    protocol=load(OUT/'protocol.json')
    executed=load(OUT/'executed-source-sha256.json')
    with zipfile.ZipFile(OUT/'source-executed-before-support-guard.zip') as z:
        for name,h in executed.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==h
        for name,h in protocol['source_sha256'].items():assert executed[name]==h
        name='engine/aniso_phase1/compatible_patch.py'
        old=z.read(name).decode();new=(ROOT/name).read_text()
    # Exact equality after removing ONLY the newly added prepare guard.
    a=new.index('    def prepare(self):\n',new.index('class CompatiblePatchEnhancements'))
    b=new.index('    def commit(self):\n',a)
    assert new[:a]+new[b:]==old
    diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                 fromfile='executed/'+name,tofile='delivered/'+name))
    (OUT/'post-run-support-guard.diff').write_text(diff)
    current=sources();changed=[n for n,h in protocol['source_sha256'].items() if current[n]!=h]
    assert set(changed)=={'engine/aniso_phase1/compatible_patch.py','tests/test_aniso_compatible_patch.py'}
    records=[]
    for case in protocol['cases']:
        rows=[__import__('json').loads(s) for s in (OUT/'cases'/case/'steps.jsonl').read_text().splitlines()]
        pairs=sorted(set((r['active_nodes'],r['material_carriers']) for r in rows))
        assert all(n<=c for n,c in pairs)
        records.append(dict(case=case,steps=len(rows),active_nodes_carriers=pairs,
                            rejected_by_delivered_guard=0))
    assert sum(r['steps'] for r in records)==9300
    tests=load(OUT/'support-guard-tests.json');assert tests['passed'] and tests['tests_run']==7
    write(OUT/'support-guard-provenance.json',dict(passed=True,
        formal_runs_executed_before_guard=True,formal_steps=9300,records=records,
        frozen_source_manifest_verified=True,only_engine_change='prepare support-capacity rejection',
        guard_inactive_on_all_formal_steps=True,numerical_formulas_unchanged_on_formal_states=True,
        source_changes_since_formal_runs=changed,post_guard_candidate_tests=tests,
        caveat='No full post-guard trajectory rerun; exact source diff plus all recorded support dimensions establish branch inactivity. Shape guard is necessary, not a general rank certificate.'))
    print('Verified executed source and inactive guard on 9300 formal steps.')

if __name__=='__main__':main()
