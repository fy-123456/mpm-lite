"""Explicit engineering acceptance, separate from relative diagnostics.

The absolute energy derivative gate is the same 1e-8 gate used in v22 and in
the frozen first pilot source. Relative differences near a zero directional
force remain visible; they are not silently substituted for that criterion.
"""
import math


def nonlinear_checks(curve, rotation_energy_error, rotation_force_error, gates):
    if not curve:
        raise ValueError("At least one perturbation is required")
    values = [rotation_energy_error, rotation_force_error]
    for row in curve:
        values.extend(row[k] for k in ("epsilon", "energy_derivative_absolute_error",
                      "energy_derivative_relative_error", "tangent_relative_error"))
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("Non-finite or negative validation measurement")
    energy = all(v["energy_derivative_absolute_error"] < gates["energy_derivative_absolute"] for v in curve)
    tangent = all(v["tangent_relative_error"] < gates["tangent_relative"] for v in curve)
    objective = (rotation_energy_error < gates["rotation_energy_absolute"] and
                 rotation_force_error < gates["rotation_force_relative"])
    relative = all(v["energy_derivative_relative_error"] < gates["derivative_relative_diagnostic"] for v in curve)
    return dict(engineering_passed=bool(energy and tangent and objective),
                energy_absolute_passed=bool(energy), tangent_passed=bool(tangent), objectivity_passed=bool(objective),
                relative_derivative_diagnostic_passed=bool(relative),
                acceptance_rule="absolute energy derivative, relative tangent, finite rigid-motion objectivity",
                relative_derivative_role="diagnostic only; record all perturbation scales",
                thresholds={k: gates[k] for k in ("energy_derivative_absolute", "tangent_relative",
                    "rotation_energy_absolute", "rotation_force_relative", "derivative_relative_diagnostic")})
