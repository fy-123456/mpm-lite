"""Verify a sealed first-stage common input, including external array bytes."""
import json
from pathlib import Path
from .identity import sha, BASELINE


def _member(root, relative, trusted_data_root=None):
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("Bundle members must be relative without traversal")
    path = root/rel
    target = path.resolve()
    allowed = target.is_relative_to(root.resolve())
    if trusted_data_root is not None:
        allowed |= target.is_relative_to(Path(trusted_data_root).resolve())
    if not allowed:
        raise ValueError("External input requires an explicit trusted data root")
    if not path.is_file():
        raise ValueError("Missing bundle member: " + relative)
    return path


def verify_common_bundle(folder, repository, *, trusted_data_root=None,
                         expected_sha256=None, require_dynamic=False):
    root, repo = Path(folder), Path(repository)
    manifest = root/"bundle.json"
    actual = sha(manifest)
    expected = expected_sha256 or (root/"bundle-sha256.txt").read_text().strip()
    if actual != expected:
        raise ValueError("Common bundle identity changed")
    data = json.loads(manifest.read_text())
    if data.get("schema_version") != 1 or data.get("baseline_sha256") != BASELINE:
        raise ValueError("Unsupported common baseline or schema")
    if require_dynamic:
        raise ValueError("Input freeze is not dynamic or continuum certification")
    members = data.get("files")
    required = {"protocol.json", "acceptance.json", "maps.json", "kinetic-state-contract.json",
                "initial-state.npz", "input-states.npz", "inputs/space-package.json",
                "inputs/geometry.npz", "inputs/carrier-basis.npz", "inputs/basis-raw.npz",
                "inputs/basis-transform.npz", "inputs/test-vectors.npz"}
    if not isinstance(members,dict) or not required.issubset(members):
        raise ValueError("Incomplete common inputs")
    for relative, digest in members.items():
        if sha(_member(root,relative,trusted_data_root)) != digest:
            raise ValueError("Changed common input or evidence: "+relative)
    sources = data.get("code_sha256",{})
    if not sources:
        raise ValueError("Missing bound implementation")
    for relative, digest in sources.items():
        if sha(_member(repo, relative)) != digest:
            raise ValueError("Changed common implementation: "+relative)
    acceptance = json.loads((root/"acceptance.json").read_text())
    if acceptance.get("common_input_freeze_passed") is not True:
        raise ValueError("Common input acceptance failed")
    if acceptance.get("dynamic_certified") or acceptance.get("continuum_spatial_certified"):
        raise ValueError("This stage cannot grant dynamic or continuum certification")
    package = json.loads(_member(root,"inputs/space-package.json",trusted_data_root).read_text())
    if sha(_member(root,"inputs/space-package.json",trusted_data_root)) != data["space_manifest_sha256"]:
        raise ValueError("Wrong frozen A space")
    for name, digest in package["files"].items():
        if members.get("inputs/"+name) != digest:
            raise ValueError("A package and common manifest disagree")
    if acceptance.get("protocol_sha256") != members["protocol.json"]:
        raise ValueError("Acceptance does not bind this protocol")
    return dict(passed=True, bundle_sha256=actual, space_manifest_sha256=data["space_manifest_sha256"],
                files=len(members), sources=len(sources), dynamic_certified=False,
                continuum_spatial_certified=False)
