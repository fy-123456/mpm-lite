"""Adversarial provenance tests; these fixtures do not stand in for physics."""
import json
from pathlib import Path
import tempfile
import unittest
from engine.aniso_phase1.research_d.identity import sha, BASELINE
from engine.aniso_phase1.research_d.frozen_inputs import verify_frozen_inputs, EVIDENCE


class FrozenInputClosure(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name)/"repo";self.repo.mkdir()
        self.root=self.repo/"freeze";self.root.mkdir()
        self.put(self.repo/"code.py","immutable numerical implementation")
        self.put(self.repo/"utils/dependency.py","baseline dependency")
        self.code={p:sha(self.repo/p) for p in ("code.py","utils/dependency.py")}
        self.base=self.repo/"docs/results/lite-aniso-mainline/v22/source-delivered-sha256.json"
        self.j(self.base,self.code)
        for p in ("geometry.npz","carrier-basis.npz","basis-raw.npz","basis-transform.npz","test-vectors.npz"):
            self.put(self.root/"inputs"/p,"byte fixture")
        package=dict(files={p.name:sha(p) for p in (self.root/"inputs").iterdir()})
        self.j(self.root/"inputs/space-package.json",package)
        self.space=sha(self.root/"inputs/space-package.json")
        self.protocol=dict(code_sha256={"code.py":self.code["code.py"]},
             space_manifest_sha256=self.space,baseline_sha256=BASELINE,material_reference={"orders":[6,7]})
        self.j(self.root/"protocol.json",self.protocol)
        for p in ("initial-state.npz","input-states.npz"):
            self.put(self.root/p,"byte fixture")
        for p in EVIDENCE:self.j(self.root/p,dict(passed=True))
        self.j(self.root/"maps.json",dict(package_manifest_sha256=self.space))
        self.j(self.root/"kinetic-state-contract.json",dict(space_id=self.space))
        self.j(self.root/"boundary-audit.json",dict(passed=True,space_manifest_sha256=self.space))
        self.j(self.root/"material-reference-audit.json",dict(passed=True,
            space_signature=self.space,protocol=self.protocol["material_reference"]))
        self.acceptance=dict(common_input_freeze_passed=True,protocol_sha256=sha(self.root/"protocol.json"),
          space_manifest_sha256=self.space,dynamic_certified=False,continuum_spatial_certified=False,
          cuda_certified=False,default_changed=False)
        self.j(self.root/"acceptance.json",self.acceptance)
        self.put(self.root/"tests.log","targeted tests passed")
        self.put(self.root/"bundle-tests.log","closure tests passed")
        self.j(self.root/"tests-summary.json",dict(passed=True,test_log_sha256=sha(self.root/"tests.log"),
              bundle_test_log_sha256=sha(self.root/"bundle-tests.log")))
        self.j(self.root/"source-closure.json",dict(execution_protocol_sha256=sha(self.root/"protocol.json"),
              executed_sources_preserved=True))
        self.seal()

    def put(self,p,s):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
    def j(self,p,d):self.put(p,json.dumps(d))
    def seal(self,omit=()):
        files={str(p.relative_to(self.root)):sha(p) for p in self.root.rglob("*") if p.is_file()
               and p.name not in ("bundle.json","bundle-sha256.txt") and str(p.relative_to(self.root)) not in omit}
        self.j(self.root/"bundle.json",dict(schema_version=1,baseline_sha256=BASELINE,
               space_manifest_sha256=self.space,code_sha256=self.code,files=files,
               baseline_source_manifest_sha256=sha(self.base)))
        self.put(self.root/"bundle-sha256.txt",sha(self.root/"bundle.json"))
    def check(self,**kwargs):return verify_frozen_inputs(self.root,self.repo,**kwargs)

    def test_complete_inputs_and_pin(self):
        self.assertTrue(self.check()["complete_evidence_verified"])
        for name,key in (("maps.json","package_manifest_sha256"),("kinetic-state-contract.json","space_id"),
                         ("boundary-audit.json","space_manifest_sha256")):
            original=json.loads((self.root/name).read_text())
            self.j(self.root/name,dict(original,**{key:"f"*64}));self.seal()
            with self.assertRaisesRegex(ValueError,"Cross-package"):self.check()
            self.j(self.root/name,original);self.seal()
        with self.assertRaises(ValueError):self.check(expected_sha256="0"*64)
        with self.assertRaises(ValueError):self.check(require_dynamic=True)

    def test_array_corruption_and_external_root_rejected(self):
        path=self.root/"inputs/geometry.npz"
        path.write_text("changed");self.assertRaises(ValueError,self.check)
        path.write_text("byte fixture")
        external=Path(self.tmp.name)/"external";external.mkdir()
        path.replace(external/"geometry.npz");path.symlink_to(external/"geometry.npz")
        with self.assertRaises(ValueError):self.check()
        self.assertTrue(self.check(trusted_data_root=external)["passed"])

    def test_required_evidence_cannot_be_omitted_or_failed(self):
        self.seal(omit=("kinetic-audit.json",))
        with self.assertRaises(ValueError):self.check()
        self.j(self.root/"kinetic-audit.json",dict(passed=False));self.seal()
        with self.assertRaises(ValueError):self.check()

    def test_changed_protocol_cannot_be_relabelled(self):
        self.protocol["space_manifest_sha256"]="f"*64
        self.j(self.root/"protocol.json",self.protocol)
        self.acceptance["protocol_sha256"]=sha(self.root/"protocol.json")
        self.j(self.root/"acceptance.json",self.acceptance);self.seal()
        with self.assertRaises(ValueError):self.check()

    def test_baseline_dependency_and_execution_identity(self):
        self.put(self.repo/"utils/dependency.py","changed baseline dependency")
        with self.assertRaises(ValueError):self.check()
        self.put(self.repo/"utils/dependency.py","baseline dependency")
        self.protocol["code_sha256"]["code.py"]="1"*64
        self.j(self.root/"protocol.json",self.protocol)
        self.acceptance["protocol_sha256"]=sha(self.root/"protocol.json")
        self.j(self.root/"acceptance.json",self.acceptance)
        self.j(self.root/"source-closure.json",dict(execution_protocol_sha256=sha(self.root/"protocol.json"),executed_sources_preserved=True))
        self.seal()
        with self.assertRaisesRegex(ValueError,"Executed numerical source"):self.check()

    def test_cuda_and_dynamic_claims_rejected(self):
        self.acceptance["cuda_certified"]=True
        self.j(self.root/"acceptance.json",self.acceptance);self.seal()
        with self.assertRaises(ValueError):self.check()
