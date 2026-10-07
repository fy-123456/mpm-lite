"""Execute the frozen first-delivery scope sequentially with append-only logs."""
import argparse
from datetime import datetime, timezone
import os
import subprocess
import sys
from time import monotonic
from .protocol import ROOT, run_path, write_json, read_json, freeze, environment


def stage(run, name, module, *args):
    start = monotonic()
    with (run / f"{name}.log").open("x") as log:
        result = subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    record = dict(returncode=result.returncode, seconds=monotonic()-start, environment=environment())
    write_json(run / f"{name}-status.json", record)
    print(name, record["returncode"], record["seconds"], flush=True)
    if result.returncode:
        raise RuntimeError(f"Stage failed: {name}; see {name}.log")


def execute(run_id):
    run = freeze(run_id)
    try:
        stage(run, "tests", "unittest", "tests.research_a.test_baseline_adapter", "tests.research_a.test_space_invariants", "tests.research_a.test_validation",
              "tests.test_aniso_v22_space", "tests.test_aniso_v22_metrics", "-v")
        status = read_json(run / "tests-status.json")
        write_json(run / "tests.json", dict(passed=status["returncode"] == 0, output="tests.log", seconds=status["seconds"]))
        stage(run, "baseline", "benchmarks.research_a.baseline", "--run-id", run_id)
        stage(run, "reference-plan", "benchmarks.research_a.reference_certification", "--run-id", run_id)
        configurations = [("v22-overlap", "stress-correction"), ("v22-overlap", "geometric"), ("wide-overlap", "stress-correction")]
        write_json(run / "pilot-matrix.json", dict(configurations=configurations, scalar_budget=144, rounds=6,
                   default_candidate="q4-v22-overlap-stress-correction-144", held_out_selection=False))
        names = []
        for family, estimator in configurations:
            name = f"q4-{family}-{estimator}-144"
            names.append(name)
            stage(run, name, "benchmarks.research_a.support_ablation", "--run-id", run_id,
                  "--family", family, "--estimator", estimator)
        for name in names:
            stage(run, name+"-audit", "benchmarks.research_a.nonlinear_audit", "--run-id", run_id, "--candidate", name)
        # Candidates, supports, budgets, and default are frozen before any final-field access.
        candidate_args = [item for name in names for item in ("--candidate", name)]
        stage(run, "training-metrics", "benchmarks.research_a.acceptance", "--run-id", run_id, *candidate_args, "--training")
        stage(run, "held-out-metrics", "benchmarks.research_a.acceptance", "--run-id", run_id, *candidate_args)
        stage(run, "package-export", "benchmarks.research_a.export", "--run-id", run_id, "--candidate", names[0])
        stage(run, "package-replay", "benchmarks.research_a.validate_package", "--run-id", run_id, "--candidate", names[0])
        stage(run, "seal", "benchmarks.research_a.report", "--run-id", run_id, *candidate_args)
        print("COMPLETED FIRST DELIVERY", run, flush=True)
    except Exception as error:
        write_json(run / "execution-failure.json", dict(reason=str(error), utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-A-pilot"))
    a = parser.parse_args()
    execute(a.run_id)
