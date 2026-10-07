"""Frozen rectangular support families with explicit physical/artificial faces."""
from __future__ import annotations
import copy
import numpy as np
from engine.aniso_phase1.stress_local_space import patches, geometric_order


def support_family(name):
    result = copy.deepcopy(patches())
    if name == "v22-original":
        return result
    if name == "v22-overlap":
        intervals = ((.25, .5625), (.4375, .75), (.34375, .65625))
    elif name == "wide-overlap":
        intervals = ((.25, .625), (.375, .75), (.3125, .6875))
    elif name == "fiber-rect":
        # Axis-aligned envelope elongated in the F45 x/y plane. No rotation,
        # oblique quadrature, or changed physical boundary is implied.
        for item in result:
            c = np.asarray(item["center"])
            radius = np.array([.125, .125, .0625])
            item["lo"] = np.maximum(c-radius, [.25, .375, .375]).tolist()
            item["hi"] = np.minimum(c+radius, [.75, .625, .625]).tolist()
        intervals = ((.25, .5625), (.4375, .75), (.34375, .65625))
    else:
        raise ValueError(f"Unknown support family: {name}")
    for lo, hi in intervals:
        result.append(dict(center=[(lo+hi)/2, .5, .5], lo=[lo, .375, .375], hi=[hi, .625, .625]))
    return result


def deterministic_order(definitions, round_index=0):
    order = np.asarray(geometric_order(definitions))
    # Visit a predeclared spread of patches, independent of stress or labels.
    return np.roll(order, -8*round_index)
