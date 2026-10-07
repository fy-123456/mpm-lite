"""A3 resource preflight; oversized reference solves are never implicit."""
import argparse
import numpy as np
from .protocol import BASE, require_frozen, run_path, write_json


def transition_refinement(edges):
    refined, marks = [], []
    for k, e in enumerate(edges):
        e = np.asarray(e)
        middle = .5*(e[:-1]+e[1:])
        if k == 0:
            idx = np.flatnonzero(((middle >= .25) & (middle < .375)) | ((middle > .625) & (middle <= .75)))
        else:
            idx = np.flatnonzero((middle < .4375) | (middle > .5625))
        refined.append(np.union1d(e, middle[idx]))
        marks.append(idx.tolist())
    return refined, marks


def estimate(edges, degree):
    shape = [degree*(len(e)-1)+1 for e in edges]
    nodes = int(np.prod(shape))
    active_x = degree*(np.sum((edges[0] >= .25) & (edges[0] <= .75))-1)+1
    return dict(nodes=nodes, free_vector_dofs=int(3*(active_x-2)*shape[1]*shape[2]),
                predicted_peak_bytes=nodes*3*8*30, uncompressed_field_bytes=nodes*3*8)


def plan(run):
    protocol = require_frozen(run)
    # Coordinates only, not the held-out displacement or stress field.
    with np.load(BASE / "v22/reference/level1-q4.npz", allow_pickle=False) as data:
        edges = [data[f"axis{k}"].copy() for k in range(3)]
    levels = []
    for level in range(2):
        edges, marked = transition_refinement(edges)
        cases = {f"q{p}": estimate(edges, p) for p in (3, 4)}
        for case in cases.values():
            case["within_resources"] = bool(case["nodes"] <= protocol["resources"]["new_reference_node_limit"] and
                 case["predicted_peak_bytes"] <= protocol["resources"]["memory_limit_GiB"]*1024**3)
        levels.append(dict(level=level, edges=[e.tolist() for e in edges], marked_intervals=marked, cases=cases))
    record = dict(completed=True, reference_solves_executed=False, certified=False, levels=levels,
        status="preflight_only", initial_reference_status=protocol["reference_status"],
        boundary_rule="entire transition and transverse boundary bands; rigid volumes preserved",
        reason="Large h/p certification is a continuation; first delivery does not claim full-region accuracy",
        continuum_error_bound_proved=False)
    write_json(run / "reference-plan.json", record)
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    print(plan(run_path(p.parse_args().run_id)))
