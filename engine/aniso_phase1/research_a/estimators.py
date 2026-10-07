"""Reference-free indicators: only residuals and local correction operators.

No filesystem access, reference field, training labels, or fitted weights occur
in this module. Indicators are heuristics, not certified error bounds.
"""
from __future__ import annotations
import numpy as np


def correction_score(rhs, correction, kind="stress-correction", stress_action=None):
    rhs = np.asarray(rhs)
    correction = np.asarray(correction)
    if rhs.shape != correction.shape or rhs.ndim != 1:
        raise ValueError("Expected matching vector residual and correction")
    if not np.isfinite(rhs).all() or not np.isfinite(correction).all():
        raise ValueError("Non-finite local indicator input")
    if kind == "energy":
        value = .5*float(rhs@correction)
    elif kind == "residual":
        value = float(rhs@rhs)
    elif kind == "stress-correction":
        if stress_action is None:
            raise ValueError("Stress correction requires the local H.T@H operator")
        value = float(correction@stress_action(correction))
    else:
        raise ValueError(f"Unknown online indicator {kind}")
    if not np.isfinite(value) or value < -1e-12:
        raise ValueError("Indicator violated positive-energy assumptions")
    return max(0., value)


def ranked_indices(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a finite vector")
    return np.argsort(-scores, kind="stable")
