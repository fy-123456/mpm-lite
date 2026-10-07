"""Seal numerical inputs, evidence and validation-only additions without overwrite."""
import argparse
import json
import zipfile
from pathlib import Path
from engine.aniso_phase1.research_d.identity import ROOT, BASELINE, sha, write_json
from engine.aniso_phase1.research_d.frozen_inputs import verify_frozen_inputs
from .freeze_common import source_manifest


def seal(out, trusted_data_root):
    out=Path(out)
    protocol=json.loads((out/"protocol.json").read_text())
    current=source_manifest()
    if any(current.get(p)!=h for p,h in protocol["code_sha256"].items()):
        raise ValueError("An executed source changed; new numerical evidence required")
    base_path=ROOT/"docs/results/lite-aniso-mainline/v22/source-delivered-sha256.json"
    baseline=json.loads(base_path.read_text())
    code=dict(current)
    for p,h in baseline.items():
        if sha(ROOT/p)!=h or p in code and code[p]!=h:
            raise ValueError("Original baseline changed: "+p)
        code[p]=h
    for name,count in (("tests.log",24),("bundle-tests.log",6)):
        content=(out/name).read_text()
        if f"Ran {count} tests" not in content or "\nOK" not in content or "FAILED" in content:
            raise ValueError("Targeted tests did not pass: "+name)
    write_json(out/"tests-summary.json",dict(passed=True,tests=30,
        numerical_and_existing_contract_tests=24,provenance_tests=6,
        test_log_sha256=sha(out/"tests.log"),bundle_test_log_sha256=sha(out/"bundle-tests.log"),
        full_repository_tests=False,cuda_tests=False))
    write_json(out/"source-closure.json",dict(execution_protocol_sha256=sha(out/"protocol.json"),
        executed_sources_preserved=True,
        baseline_source_manifest_sha256=sha(base_path),baseline_source_count=len(baseline),
        additional_bound_sources=sorted(set(code)-set(protocol["code_sha256"])),
        explanation="Original numerical sources unchanged; full v22 import closure plus later validation-only sources are bound.",
        final_code_sha256=code))
    # A code snapshot lets parallel workers restore exactly these dependencies
    # even when the active research working copy advances.
    with zipfile.ZipFile(out/"code-snapshot.zip","x",compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(code):
            archive.write(ROOT/relative,relative)
        archive.write(base_path,str(base_path.relative_to(ROOT)))
    paths=[p for p in out.rglob("*") if p.is_file()]
    paths.extend(p for p in (out/"inputs").iterdir() if p.is_file())
    files={str(p.relative_to(out)):sha(p) for p in sorted(set(paths))}
    forbidden={"bundle.json","bundle-sha256.txt","verification.json","verification.log","consumer-smoke.json"}
    if forbidden.intersection(files):
        raise FileExistsError("This output was already sealed or consumed")
    bundle=dict(schema_version=1,stage="first_stage_common_input_freeze",baseline_sha256=BASELINE,
        baseline_source_manifest_sha256=sha(base_path),
        space_manifest_sha256=protocol["space_manifest_sha256"],
        code_sha256=code,files=files,
        capabilities=["frozen_static_input","position_and_gradient_adjoint",
            "bounded_full_material_reference","reference_point_mass_action","owned_state_transaction"],
        unvalidated=["continuum_spatial_accuracy","full_mass_rank","dynamic_cycle","Eulerian_MPM","CUDA","coupled_physics"],
        storage="inputs is an independent copied dataset; explicit trusted data root required if outside repository")
    write_json(out/"bundle.json",bundle)
    with (out/"bundle-sha256.txt").open("x") as f:f.write(sha(out/"bundle.json")+"\n")
    result=verify_frozen_inputs(out,ROOT,trusted_data_root=trusted_data_root,
                               expected_sha256=sha(out/"bundle.json"))
    write_json(out/"verification.json",result)
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--output",required=True)
    p.add_argument("--trusted-data-root",required=True)
    a=p.parse_args();seal(a.output,a.trusted_data_root)

if __name__=="__main__":main()
