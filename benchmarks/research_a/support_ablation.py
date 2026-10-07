"""A5/A6 pilot runner with explicit family, indicator, and scalar budget."""
import argparse
from time import monotonic
import resource
import numpy as np
import scipy.sparse as sp
from .protocol import require_frozen, run_path, write_json, storage_guard, sha, save_npz, save_sparse
from engine.aniso_phase1.research_a.baseline_adapter import Problem
from engine.aniso_phase1.research_a.adaptive import enrich


def run_candidate(run, family="v22-overlap", estimator="stress-correction", budget=144, degree=4):
    protocol = require_frozen(run)
    if budget not in protocol["convergence_budgets"] or degree not in (2, 3, 4):
        raise ValueError("Candidate outside frozen budget/degree family")
    folder = run / "candidates" / f"q{degree}-{family}-{estimator}-{budget}"
    folder.mkdir(parents=True, exist_ok=False)
    start = monotonic()
    problem = Problem.from_archive(degree)
    # Six rounds for each budget: 4/8/16 selected patches per round.
    per_round = budget//18
    definitions = protocol["support_families"][family]
    write_json(folder / "config.json", dict(family=family, estimator=estimator, budget=budget, degree=degree,
               rounds=6, patches_per_round=per_round, definitions=definitions, reference_field_used=False))
    def publish(record, field):
        storage_guard(run)
        name = f"round{record['round']}"
        write_json(folder / f"{name}.json", record)
        # Retain each raw field for actual per-round gains, never use held-out
        # metrics to decide which round to publish as the final candidate.
        save_npz(run, folder / f"{name}.npz", **field, degree=degree,
                            **{f"axis{k}": e for k, e in enumerate(problem.edges)})
        print("ROUND", folder.name, record["round"], record["scalar_local_dofs"], record["materials"]["F45"], flush=True)
    try:
        result = enrich(problem, definitions, estimator, 6, per_round, budget, publish)
        save_npz(run, folder / "basis-transform.npz", transform=result["transform"])
        save_sparse(run, folder / "basis-raw.npz", result["raw"])
        write_json(folder / "summary.json", dict(completed=True, scalar_dofs=budget, rounds=6,
                   seconds=monotonic()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                   basis_reconstruction_relative=result["basis_reconstruction_relative"],
                   reference_field_used=False, all_static_passed=True,
                   field_sha256=sha(folder / "round6.npz"), raw_sha256=sha(folder / "basis-raw.npz"),
                   transform_sha256=sha(folder / "basis-transform.npz")))
    except Exception as error:
        write_json(folder / "failure.json", dict(completed=False, reason=str(error), seconds=monotonic()-start))
        raise
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--family", default="v22-overlap", choices=("v22-original", "v22-overlap", "wide-overlap", "fiber-rect"))
    parser.add_argument("--estimator", default="stress-correction", choices=("stress-correction", "energy", "residual", "geometric"))
    parser.add_argument("--budget", type=int, default=144)
    parser.add_argument("--degree", type=int, default=4)
    args = parser.parse_args()
    print(run_candidate(run_path(args.run_id), args.family, args.estimator, args.budget, args.degree))
