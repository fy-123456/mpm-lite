"""A7: original nonlinear potential checks on the actual final candidate."""
import argparse
from time import monotonic
import numpy as np
import scipy.linalg as la
from .protocol import candidate_path, require_frozen, run_path, write_json, sha
from engine.aniso_phase1.research_a.baseline_adapter import Problem, load_basis
from engine.aniso_phase1.research_a.export_adapter import FixedSpace
from engine.aniso_phase1.research_a.validation import nonlinear_checks


def audit(run, candidate):
    protocol = require_frozen(run)
    folder = candidate_path(run, candidate)
    with np.load(folder / "round6.npz", allow_pickle=False) as data:
        degree = int(data["degree"])
        y, alpha = data["y"].copy(), data["local_coefficients"].copy()
    start = monotonic()
    problem = Problem.from_archive(degree)
    W = load_basis(folder)
    space = FixedSpace(problem, W)
    del W
    q = np.vstack((problem.Q.T@(y-problem.lift), alpha))
    rng = np.random.default_rng(220930)
    direction = rng.normal(size=q.shape)
    direction /= la.norm(direction)
    base = space.response(q, direction)
    curve = []
    for eps in (3e-6, 1e-6, 3e-7):
        plus, minus = space.response(q+eps*direction), space.response(q-eps*direction)
        fd = (plus["energy_J"]-minus["energy_J"])/(2*eps)
        analytic = float(np.sum(base["force"]*direction))
        curve.append(dict(epsilon=eps, energy_derivative_absolute_error=abs(fd-analytic),
            energy_derivative_relative_error=abs(fd-analytic)/max(abs(analytic), 1e-6),
            tangent_relative_error=float(la.norm((plus["force"]-minus["force"])/(2*eps)-base["tangent_action"])/la.norm(base["tangent_action"]))))
        print("NONLINEAR derivative", curve[-1], flush=True)
    Y = space.total_coefficients(q)
    pot = space.potential
    original = pot.evaluate(Y)
    rotation = la.expm(np.array([[0., -.6, .2], [.6, 0., -.1], [-.2, .1, 0.]]))
    rotated = Y@rotation.T
    rotated[:len(problem.carrier_X)] += [.013, -.021, .008]
    objective = pot.evaluate(rotated)
    er = abs(objective["U"]-original["U"])
    fr = float(la.norm(objective["force"]-original["force"]@rotation.T)/max(la.norm(original["force"]), 1e-20))
    refined = space.response(q, order=degree+2)
    quadrature = dict(orders=[degree+1, degree+2], energy_relative=abs(refined["energy_J"]-base["energy_J"])/max(abs(refined["energy_J"]), 1e-12),
                      force_relative=float(la.norm(refined["force"]-base["force"])/max(la.norm(refined["force"]), 1e-8)))
    checks = nonlinear_checks(curve, er, fr, protocol["implementation_gates"])
    passed = checks["engineering_passed"]
    record = dict(completed=True, passed=passed, source_sha256=sha(folder / "round6.npz"),
        scalar_local_dofs=space.W.shape[1], curve=curve, acceptance_checks=checks, energy_rotation_error_J=er,
        force_rotation_relative=fr, quadrature=quadrature, invariants=problem.invariant_errors(space.W),
        seconds=monotonic()-start, dynamic_scene_tested=False, mass_included=False, stiffness_shift=0.)
    write_json(folder / "nonlinear-audit.json", record)
    if not passed:
        raise RuntimeError("Actual candidate nonlinear audit failed")
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--candidate", required=True)
    a = p.parse_args()
    print(audit(run_path(a.run_id), a.candidate))
