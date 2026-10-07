"""Reproducible v17 test gate; refuses to overwrite existing results."""
import unittest
from benchmarks.aniso_v17_time import OUT,write


def main():
    assert not (OUT/'tests.json').exists(),'preserve recorded test outcome'
    modules=['tests.test_aniso_v17_modes','tests.test_aniso_carrier_avf','tests.test_aniso_carrier_joint']
    with (OUT/'tests.log').open('x') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(modules))
    write(OUT/'tests.json',dict(passed=result.wasSuccessful(),tests_run=result.testsRun,new_modal_tests=4,existing_AVF_tests=4,existing_joint_tests=8,full_repository_suite=False))
    with (OUT/'time-metrics-tests.log').open('x') as log:
        metrics=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.test_aniso_v17_time_metrics'))
    assert result.wasSuccessful() and result.testsRun==16
    assert metrics.wasSuccessful() and metrics.testsRun==2
if __name__=='__main__':main()
