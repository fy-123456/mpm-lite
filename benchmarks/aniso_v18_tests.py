"""Reproducible test entry points; refuse to replace delivered test records."""
import argparse,unittest
from benchmarks.aniso_v18_runs import OUT,write

def main():
    p=argparse.ArgumentParser();p.add_argument('group',choices=['solver','space']);group=p.parse_args().group
    modules=['tests.test_aniso_carrier_driven','tests.test_aniso_carrier_avf','tests.test_aniso_carrier_joint'] if group=='solver' else ['tests.test_aniso_v18_space','benchmarks.aniso_local_reference.LocalReferenceTests']
    stem='tests' if group=='solver' else 'space-tests';assert not (OUT/f'{stem}.json').exists();OUT.mkdir(parents=True,exist_ok=True)
    suite=unittest.defaultTestLoader.loadTestsFromNames(modules)
    with (OUT/f'{stem}.log').open('x') as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write(OUT/f'{stem}.json',dict(passed=result.wasSuccessful(),tests_run=result.testsRun,full_repository_suite=False,modules=modules))
    raise SystemExit(0 if result.wasSuccessful() else 2)
if __name__=='__main__':main()
