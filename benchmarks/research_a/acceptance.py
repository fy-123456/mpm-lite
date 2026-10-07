"""Freeze final fields before opening held-out v22 data; no automatic promotion."""
import argparse
from time import monotonic
from .protocol import candidate_path, BASE, HELD_OUT, TRAINING, require_frozen, run_path, write_json, read_json, sha
from engine.aniso_phase1.research_a.baseline_adapter import read_field
from benchmarks.aniso_v22_common import compare


def evaluate(run, candidates, training=False):
    require_frozen(run)
    paths = {name: candidate_path(run, name) / "round6.npz" for name in candidates}
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("A nonempty unique candidate list is required")
    for name, path in paths.items():
        summary = read_json(path.parent / "summary.json")
        if not summary["completed"] or sha(path) != summary["field_sha256"]:
            raise RuntimeError(f"Candidate not sealed: {name}")
    name = "training" if training else "held-out"
    write_json(run / f"{name}-candidate-freeze.json", dict(candidates={n: sha(p) for n, p in paths.items()},
               rule="Each fixed budget uses round 6; no best-round or held-out candidate selection"))
    refpath = TRAINING if training else HELD_OUT
    ref = read_field(refpath)
    ref_reaction = read_json(refpath.with_suffix(".json"))["reaction_N"]
    records = {}
    for candidate, path in paths.items():
        started = monotonic()
        metric = compare(read_field(path), ref)
        reaction = read_json(path.with_suffix(".json"))["materials"]["F45"]["reaction_N"]
        metric.update(reaction_N=reaction, reaction_relative=abs(reaction-ref_reaction)/abs(ref_reaction),
                      seconds=monotonic()-started, field_sha256=sha(path))
        records[candidate] = metric
        write_json(path.parent / f"{name}-metrics.json", metric)
        print("METRICS", candidate, metric, flush=True)
    archived = read_json(BASE / "v22" / ("training-acceptance.json" if training else "final-spatial-acceptance.json"))
    comparison = {n: archived["cases"][n] for n in ("q2-local144", "q3-local144", "q4-local144", "q4-multiscale144", "q4-full")}
    result = dict(completed=True, reference=str(refpath), reference_sha256=sha(refpath), candidates=records,
        archived_controls=comparison, archived_controls_recomputed_here=False, no_smoothing=True,
        spatial_accuracy_certified=False, reference_global_grip_certified=False,
        precision_targets=dict(reaction=.01, stress=.02, fiber_strain=.02),
        reason="Full-domain reference is still uncertified; these are differences from a discrete reference")
    write_json(run / f"{name}-acceptance.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--candidate", action="append", required=True)
    p.add_argument("--training", action="store_true")
    args = p.parse_args()
    evaluate(run_path(args.run_id), args.candidate, args.training)
