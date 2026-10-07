import numpy as np
from benchmarks.aniso_v21_common import controlled_case, BASE
from engine.aniso_phase1.research_a.baseline_adapter import Problem


def small_problem():
    state, energy, *_ = controlled_case()
    with np.load(BASE / "v19/space/reconstruction16.npz", allow_pickle=False) as data:
        A = data["A"].copy()
        edges = [data[f"axis{k}"].copy() for k in range(3)]
    lift = np.zeros_like(state.Y)
    lift[state.Y[:, 0] >= .75, 0] = .005
    return Problem(edges, 2, A, state.Y.copy(), energy.Ks.copy(), energy.params, energy.A[0].copy(), A@lift)
