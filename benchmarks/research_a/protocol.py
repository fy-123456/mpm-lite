"""Freeze inputs before A experiments; never write to a v1--v22 archive."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/results/lite-aniso-mainline"
OUTPUT = ROOT / "docs/results/parallel-v22/A"
ARCHIVE_SHA = "136529c866133aac10af77126c839f12a2769073ee5ec964f46dcadeca7adee4"
GIB = 1024**3
TRAINING = BASE / "v21/reference-extension/level1-q4.npz"
HELD_OUT = BASE / "v22/reference/level1-q4.npz"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    """Atomic first publication, refusing to replace any prior result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".pending")
    with tmp.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def run_path(run_id):
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("run_id must be a single directory name")
    return OUTPUT / run_id


def verify_baseline():
    records = {}
    for name in ("source-delivered-sha256.json", "artifact-sha256.json"):
        manifest = read_json(BASE / "v22" / name)
        failures = []
        for relative, expected in manifest.items():
            path = ROOT / relative
            actual = sha(path) if path.is_file() else None
            if actual != expected:
                failures.append(dict(path=relative, expected=expected, actual=actual))
        records[name] = dict(count=len(manifest), mismatches=failures)
    archive = sha(BASE / "v22/source-delivered.zip")
    records["archive_sha256"] = archive
    records["passed"] = archive == ARCHIVE_SHA and all(
        not records[name]["mismatches"] for name in
        ("source-delivered-sha256.json", "artifact-sha256.json"))
    # New namespaces may coexist. Unarchived changes to the shared baseline may not.
    known = read_json(BASE / "v22/source-delivered-sha256.json")
    extra = []
    for folder in ("engine", "benchmarks", "tests", "demos", "utils"):
        for path in (ROOT / folder).rglob("*.py"):
            rel = path.relative_to(ROOT)
            if str(rel) not in known and not any(p.startswith("research_") for p in rel.parts):
                extra.append(str(rel))
    records["unarchived_shared_python"] = sorted(extra)
    records["newer_version_directories"] = sorted(str(p.relative_to(ROOT)) for p in BASE.glob("v[0-9]*")
                                                   if p.name[1:].isdigit() and int(p.name[1:]) > 22)
    records["passed"] &= not extra and not records["newer_version_directories"]
    return records


def environment():
    import numpy as np
    import scipy
    return dict(utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                cpu_model=next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines()
                                if s.startswith("model name")), platform.processor()),
                cpu_affinity=sorted(os.sched_getaffinity(0)), load_average=list(os.getloadavg()),
                threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                dtype="float64", device="cpu", gpu_used=False,
                system_free_bytes=shutil.disk_usage("/").free,
                data_free_bytes=shutil.disk_usage("/root/autodl-tmp").free,
                processes=subprocess.check_output(["ps", "-eo", "pid,pcpu,pmem,comm", "--sort=-pcpu"], text=True).splitlines()[:16])


def candidate_path(run, name):
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("candidate must be a single directory name")
    return Path(run) / "candidates" / name


def source_manifest():
    paths = []
    for folder in ("engine/aniso_phase1/research_a", "benchmarks/research_a", "tests/research_a"):
        paths.extend((ROOT / folder).glob("*.py"))
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}


def require_frozen(run):
    protocol = read_json(Path(run) / "protocol.json")
    if source_manifest() != protocol["source_sha256"]:
        raise RuntimeError("Research A source set changed after freeze; create a new run")
    for rel, expected in protocol["source_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"A source changed after freeze: {rel}; create a new run")
    return protocol


def storage_guard(run, required_bytes=0):
    """Move only this A run's regenerable scratch when root falls below 5 GiB.

    No other team's files, old archives, or shared caches are touched. The link
    preserves the logical path and each moved file is checked before deletion.
    """
    run = Path(run)
    free = shutil.disk_usage("/").free
    record = dict(system_free_bytes=free, threshold_bytes=5*GIB, migrated=[])
    scratch = run / "scratch"
    data = Path("/root/autodl-tmp/mpm-lite-research-a") / run.name / "scratch"
    if free < 5*GIB and scratch.exists() and not scratch.is_symlink():
        if os.stat("/root/autodl-tmp").st_dev == os.stat("/").st_dev:
            raise RuntimeError("Data path is not a separate disk")
        files = {str(p.relative_to(scratch)): sha(p) for p in scratch.rglob("*") if p.is_file()}
        size = sum(p.stat().st_size for p in scratch.rglob("*") if p.is_file())
        if shutil.disk_usage("/root/autodl-tmp").free < size + required_bytes + 5*GIB:
            raise RuntimeError("Insufficient data disk headroom")
        data.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(scratch, data)
        if any(sha(data / p) != digest for p, digest in files.items()):
            raise RuntimeError("Scratch migration checksum mismatch; originals retained")
        shutil.rmtree(scratch)
        scratch.symlink_to(data, target_is_directory=True)
        record["migrated"] = list(files)
        record["destination"] = str(data)
        write_json(run / f"storage-migration-{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json", record)
    # Reserve the floor before starting another allocation on the root disk.
    target = scratch if scratch.exists() else run
    if os.stat(target).st_dev == os.stat("/").st_dev and shutil.disk_usage("/").free - required_bytes < 5*GIB:
        raise RuntimeError("Allocation would cross 5 GiB system floor; use A data-disk scratch")
    return record


def freeze(run_id):
    run = run_path(run_id)
    run.mkdir(parents=True, exist_ok=False)
    verified = verify_baseline()
    write_json(run / "baseline-verification.json", verified)
    if not verified["passed"]:
        raise RuntimeError("Baseline changed; see baseline-verification.json")
    paths = [BASE / "v22" / p for p in ("source-delivered.zip", "source-delivered-sha256.json",
             "artifact-sha256.json", "multiscale/space/q4/round6.npz", "multiscale/space/q4/basis-raw.npz",
             "multiscale/space/q4/basis-transform.npz", "multiscale/space-protocol.json",
             "reference-summary.json", "final-spatial-acceptance.json", "training-acceptance.json")]
    paths += [BASE / p for p in ("v11-reference-q3/cases/local2.npz", "v19/space/reconstruction32.npz",
                               "v20/space/F45-r32-e0.npz")]
    paths += [TRAINING, HELD_OUT]
    write_json(run / "input-manifest.json", {str(p.relative_to(ROOT)): sha(p) for p in paths})
    from engine.aniso_phase1.research_a.support_family import support_family
    protocol = dict(schema_version=1, producer="research_a", baseline="v22 verified delivered snapshot",
        baseline_source_sha256=ARCHIVE_SHA, source_sha256=source_manifest(), environment=environment(), seed=220930,
        units=dict(length="m", stress="Pa", force="N", energy="J"), coordinates="Cartesian reference coordinates",
        displacement_convention="x=X+u(q), F=I+grad(u); q is displacement, not total position",
        physical_box=[[.125, .875], [.375, .625], [.375, .625]], grips=[.25, .75], displacement=.005,
        material=dict(mu=10., lam=20., k_f=200., fiber_angle_degrees=45.),
        original_stabilization=True, mass_included=False, stiffness_shift=0.,
        regions=dict(global_="entire original box", grip="x<=.3125 or x>=.6875 including rigid volumes",
                     interior=".3125<x<.6875", deep_interior=".375<x<.625"),
        reference_status="v22 global/grip not certified; interior/deep interior self-check passed",
        training_reference=str(TRAINING.relative_to(ROOT)), held_out_reference=str(HELD_OUT.relative_to(ROOT)),
        held_out_rule="Only acceptance after candidate freeze may open final field; never selection or tuning",
        reproduction_tolerances=dict(relative=1e-5, energy_absolute_J=1e-10, reaction_absolute_N=1e-8),
        implementation_gates=dict(true_residual=1e-7, work_identity=1e-6, energy_derivative_absolute=1e-8,
                                  derivative_relative_diagnostic=2e-4, rotation_energy_absolute=1e-9,
                                  rotation_force_relative=1e-6,
                                  tangent_relative=5e-4, orthogonality=1e-7, rigid_trace=1e-10),
        tolerance_reason="User prioritizes stable scene behavior; these are engineering checks, not spatial certification",
        spatial_research_targets=dict(reaction=.01, stress=.02, fiber_strain=.02),
        rounds=6, patches_per_round=8, scalar_budget=144, convergence_budgets=[72, 144, 288],
        final_round_rule="round 6 at 144 functions; never select an intermediate round using held-out error",
        support_families={n: support_family(n) for n in ("v22-original", "v22-overlap", "wide-overlap", "fiber-rect")},
        online_estimators=["energy", "stress-correction", "residual", "geometric"],
        online_default="stress-correction", estimator_weights="none; no calibration on held-out data",
        offline_control="archived v22 reference-stress-gain candidate, reproduced via frozen basis",
        comparisons=["q2-original144", "q3-original144", "q4-original144", "q4-overlap144", "q4-full"],
        resources=dict(threads=int(os.environ.get("OPENBLAS_NUM_THREADS", "1")), memory_limit_GiB=24,
                       new_reference_node_limit=14000000, max_reference_seconds=3600,
                       system_disk_floor_GiB=5, concurrent_large_references=1,
                       timing_claim="observed wall time only; no fair speedup claim", repeats=1, warmup=False),
        stage_scope="First independent A delivery: A1/A2, A3 resource plan, A5/A6 online pilot, A7, A10; later research remains explicit",
        dynamics_status="not integrated; C owns time evolution and inertia", joint_status="B/C/D pending")
    write_json(run / "protocol.json", protocol)
    import zipfile
    with zipfile.ZipFile(run / "source-at-run.zip", "x", zipfile.ZIP_DEFLATED) as bundle:
        for name in protocol["source_sha256"]:
            bundle.write(ROOT / name, name)
    return run




def prepare_scratch(run, estimate_bytes=GIB):
    """Use the data disk for new arrays if root headroom would fall below 5 GiB."""
    run = Path(run)
    scratch = run / "scratch"
    if scratch.exists():
        storage_guard(run)
        return scratch
    if shutil.disk_usage("/").free - estimate_bytes < 5*GIB:
        data_root = Path("/root/autodl-tmp")
        if os.stat(data_root).st_dev == os.stat("/").st_dev or shutil.disk_usage(data_root).free < estimate_bytes+5*GIB:
            raise RuntimeError("No separate data disk with sufficient capacity")
        target = data_root / "mpm-lite-research-a" / run.name / "scratch"
        target.mkdir(parents=True, exist_ok=False)
        scratch.symlink_to(target, target_is_directory=True)
    else:
        scratch.mkdir()
    write_json(run / "storage-layout.json", dict(scratch=str(scratch), physical_path=str(scratch.resolve()),
               predicted_array_bytes=estimate_bytes, system_free_bytes=shutil.disk_usage("/").free,
               reason="Reserve 5 GiB system headroom; numeric arrays are regenerable A outputs"))
    return scratch


def array_destination(run, logical_path):
    logical_path = Path(logical_path)
    relative = logical_path.relative_to(run)
    if logical_path.exists() or logical_path.is_symlink():
        raise FileExistsError(logical_path)
    physical = prepare_scratch(run) / relative
    physical.parent.mkdir(parents=True, exist_ok=True)
    if physical.exists():
        raise FileExistsError(physical)
    return physical


def save_npz(run, path, **arrays):
    import numpy as np
    destination = array_destination(run, path)
    np.savez_compressed(destination, **arrays)
    Path(path).symlink_to(os.path.relpath(destination, Path(path).parent))


def save_sparse(run, path, matrix):
    import scipy.sparse as sp
    destination = array_destination(run, path)
    sp.save_npz(destination, matrix)
    Path(path).symlink_to(os.path.relpath(destination, Path(path).parent))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-A-start"))
    args = parser.parse_args()
    print(freeze(args.run_id), flush=True)

