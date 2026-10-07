"""Complete common-input verification, above the basic byte-manifest checker.

The numerical execution protocol is immutable. Later validation-only sources
may be added, but every source named by that protocol must remain identical.
The full v22 source manifest also binds import-side dependencies (including
utils) that are not part of the numerical module's own source enumeration.
"""
import json
from pathlib import Path
from .common_bundle import verify_common_bundle
from .identity import sha


EVIDENCE = ("mapping-audit.json", "kinetic-audit.json", "transaction-audit.json",
            "archive-wiring-audit.json", "material-reference-audit.json",
            "baseline-audit.json", "boundary-audit.json")


def verify_frozen_inputs(folder, repository, *, trusted_data_root=None,
                         expected_sha256=None, require_dynamic=False):
    root, repo = Path(folder), Path(repository)
    result = verify_common_bundle(root,repo,trusted_data_root=trusted_data_root,
                                  expected_sha256=expected_sha256,require_dynamic=require_dynamic)
    bundle = json.loads((root/"bundle.json").read_text())
    required = set(EVIDENCE) | {"source-closure.json","tests-summary.json","tests.log",
                               "bundle-tests.log","protocol.json","acceptance.json"}
    if not required.issubset(bundle["files"]):
        raise ValueError("Missing mandatory acceptance evidence")
    protocol = json.loads((root/"protocol.json").read_text())
    if protocol["space_manifest_sha256"] != bundle["space_manifest_sha256"]:
        raise ValueError("Protocol and bundle freeze different spaces")
    if protocol["baseline_sha256"] != bundle["baseline_sha256"]:
        raise ValueError("Protocol baseline differs")
    if not protocol.get("code_sha256") or any(bundle["code_sha256"].get(k)!=v
            for k,v in protocol["code_sha256"].items()):
        raise ValueError("Executed numerical source identity is not preserved")
    # Baseline audit was run after numerical checks; bind and verify that same
    # unchanged v22 source closure on every future load.
    baseline_path = repo/"docs/results/lite-aniso-mainline/v22/source-delivered-sha256.json"
    if sha(baseline_path) != bundle["baseline_source_manifest_sha256"]:
        raise ValueError("Original source manifest changed")
    baseline = json.loads(baseline_path.read_text())
    if any(bundle["code_sha256"].get(k)!=v for k,v in baseline.items()):
        raise ValueError("Incomplete baseline and import dependency closure")
    for name in EVIDENCE:
        if json.loads((root/name).read_text()).get("passed") is not True:
            raise ValueError("Failed acceptance evidence: "+name)
    acceptance = json.loads((root/"acceptance.json").read_text())
    if acceptance.get("space_manifest_sha256") != bundle["space_manifest_sha256"]:
        raise ValueError("Acceptance names another space")
    if any(acceptance.get(k) is not False for k in
           ("dynamic_certified","continuum_spatial_certified","cuda_certified","default_changed")):
        raise ValueError("Freeze acceptance exceeds this stage")
    for name,key in (("boundary-audit.json","space_manifest_sha256"),
                     ("kinetic-state-contract.json","space_id"),
                     ("maps.json","package_manifest_sha256")):
        record=json.loads((root/name).read_text())
        if record.get(key) != bundle["space_manifest_sha256"]:
            raise ValueError("Cross-package space identity: "+name)
    material = json.loads((root/"material-reference-audit.json").read_text())
    if material["space_signature"] != bundle["space_manifest_sha256"] or material["protocol"] != protocol["material_reference"]:
        raise ValueError("Material evidence is not bound to this protocol")
    tests = json.loads((root/"tests-summary.json").read_text())
    if tests.get("passed") is not True or tests.get("test_log_sha256") != bundle["files"]["tests.log"] or tests.get("bundle_test_log_sha256") != bundle["files"]["bundle-tests.log"]:
        raise ValueError("Tests are missing or not bound to raw logs")
    closure = json.loads((root/"source-closure.json").read_text())
    if closure.get("execution_protocol_sha256") != bundle["files"]["protocol.json"] or closure.get("executed_sources_preserved") is not True:
        raise ValueError("Missing execution provenance")
    result["complete_evidence_verified"] = True
    return result


def load_frozen_inputs(folder, repository, *, trusted_data_root=None, expected_sha256=None):
    """Verify every source/data/evidence hash before exposing the shared space."""
    if Path(repository).resolve() != Path(__file__).resolve().parents[3]:
        raise ValueError("Load from the same source checkout that is being verified")
    result = verify_frozen_inputs(folder,repository,trusted_data_root=trusted_data_root,
                                  expected_sha256=expected_sha256)
    from .common_space import CommonSpace
    from .common_kinetic import PointInertia
    from .common_state import CommonState
    import numpy as np
    root = Path(folder)
    space = CommonSpace(root/"inputs",expected_manifest_sha256=result["space_manifest_sha256"])
    with np.load(root/"initial-state.npz",allow_pickle=False) as z:
        state = CommonState(z["q"].copy(),z["velocity"].copy(),float(z["time"]),int(z["step"]))
    return space, PointInertia(space,order=5,density=1.),state,result
