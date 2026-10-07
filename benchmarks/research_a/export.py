"""A10: immutable self-contained static spatial package and operator samples."""
import argparse
from time import monotonic
import numpy as np
import scipy.sparse as sp
from .protocol import candidate_path, BASE, ARCHIVE_SHA, require_frozen, run_path, write_json, save_npz, save_sparse, sha
from engine.aniso_phase1.research_a.baseline_adapter import Problem, load_basis
from engine.aniso_phase1.research_a.export_adapter import FixedSpace


def export(run, candidate):
    protocol = require_frozen(run)
    source = candidate_path(run, candidate)
    target = source / "space-package"
    target.mkdir(exist_ok=False)
    started = monotonic()
    with np.load(source / "round6.npz", allow_pickle=False) as z:
        degree, y, alpha = int(z["degree"]), z["y"].copy(), z["local_coefficients"].copy()
    problem = Problem.from_archive(degree)
    W = load_basis(source)
    space = FixedSpace(problem, W)
    del W
    q = np.vstack((problem.Q.T@(y-problem.lift), alpha))
    direction = np.random.default_rng(protocol["seed"]).normal(size=space.shape)
    direction /= np.linalg.norm(direction)
    points = [np.array([.26, .33, .5, .67, .74]), np.array([.38, .5, .62]), np.array([.38, .5, .62])]
    x, F = space.evaluate(q, points)
    dx, dF = space.jvp(direction, points)
    response = space.response(q, direction)
    with np.load(BASE / "v19/space/reconstruction32.npz", allow_pickle=False) as z:
        save_npz(run, target / "carrier-basis.npz", **{k: z[k] for k in z.files})
    save_npz(run, target / "geometry.npz", carrier_X=problem.carrier_X, Ks=problem.Ks,
             fiber_tensor=problem.fiber_tensor, degree=degree, **{f"axis{k}": e for k, e in enumerate(problem.edges)})
    save_sparse(run, target / "basis-raw.npz", sp.load_npz(source / "basis-raw.npz"))
    with np.load(source / "basis-transform.npz", allow_pickle=False) as z:
        save_npz(run, target / "basis-transform.npz", transform=z["transform"])
    save_npz(run, target / "test-vectors.npz", q=q, direction=direction, x=x, F=F, dx=dx, dF=dF,
             energy_J=response["energy_J"], force=response["force"], tangent_action=response["tangent_action"],
             **{f"points{k}": a for k, a in enumerate(points)})
    metadata = dict(schema_version=1, producer="research_a", baseline_source_sha256=ARCHIVE_SHA,
        producer_source_sha256=protocol["source_sha256"], input_sha256=sha(source / "round6.npz"),
        units=protocol["units"], dtype="float64", device="cpu", coordinate_system="Cartesian reference",
        material=protocol["material"], boundary_conditions=dict(box=protocol["physical_box"], grips=[.25, .75],
            left_displacement=[0., 0., 0.], right_displacement=[.005, 0., 0.], side_faces="natural traction"),
        q_shape=list(space.shape), dof_order="rows: free carriers in original order, then local scalar basis; columns: x,y,z",
        displacement_convention="x=X+u(q); F=I+grad(u)", constraint_lift="right rigid volume .005 in x",
        scalar_component_closure=True, integration="full positive reference-volume Gauss quadrature order p+1; no compression",
        region_labels=protocol["regions"], physical_volume=.75*.25*.25, seed=protocol["seed"],
        basis_reconstruction="interpolate bundled Q2 carrier basis to target Qp; local W=raw@transform",
        artificial_boundaries="zero trace on artificial patch faces; overlap uses a global basis transform",
        nonlinear_mapping="affine in q; no additional mapping Hessian",
        newton_rule="fixed package/basis for the entire Newton or line search; no history commit",
        fixed_test_gates=dict(mapping_absolute=1e-7, energy_relative=1e-5, force_relative=1e-5, tangent_relative=5e-4),
        applicability="F45 original-box static spatial adapter; dynamics and B/C/D combinations untested",
        seconds=monotonic()-started,
        files={p.name: sha(p) for p in sorted(target.glob("*.npz"))})
    write_json(target / "space-package.json", metadata)
    print("EXPORTED", target, flush=True)
    return target


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--candidate", required=True)
    a = p.parse_args()
    export(run_path(a.run_id), a.candidate)
