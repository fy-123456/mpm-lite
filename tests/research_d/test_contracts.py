import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from engine.aniso_phase1.research_contracts import HandoffMetadata,validate_package
from engine.aniso_phase1.research_d.identity import sha,BASELINE


class ContractChecks(unittest.TestCase):
    def test_incomplete_or_ambiguous_metadata_rejected(self):
        m=self.metadata({'a':'a'*64},{'b':'b'*64})
        m.validate()
        for change in (dict(schema_version=2),dict(q_convention='guess'),dict(units={}),
                       dict(capabilities=('dynamic',))):
            with self.assertRaises(ValueError):dataclasses.replace(m,**change).validate()

    def metadata(self,code,inputs):
        return HandoffMetadata(1,BASELINE,'D',code,inputs,{'length':'m','force':'N'},'float64','cpu',
            'Cartesian reference','displacement',{'mu':10}, {'left':'hard'},'positive reference volume',7,('static',))

    def test_sealed_hashes_capability_and_changed_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'code').write_text('implementation'); (root/'input').write_text('frozen geometry')
            code={'code':sha(root/'code')}; inputs={'input':sha(root/'input')}; m=self.metadata(code,inputs)
            (root/'acceptance').write_text(json.dumps(dict(producer='D',baseline_sha256=BASELINE,passed=True,
                input_sha256=inputs,code_sha256=code,validated_capabilities=['static'])))
            package=dict(metadata=dataclasses.asdict(m),acceptance=dict(path='acceptance',sha256=sha(root/'acceptance')))
            validate_package(package,root,BASELINE)
            with self.assertRaises(ValueError):validate_package(package,root,BASELINE,require_dynamic=True)
            (root/'input').write_text('changed geometry')
            with self.assertRaises(ValueError):validate_package(package,root,BASELINE)

    def test_append_only_json_handles_numpy_scalars_atomically(self):
        import numpy as np
        from engine.aniso_phase1.research_d.identity import write_json
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'result.json'
            write_json(path,{'passed':np.bool_(True),'count':np.int64(2)})
            self.assertEqual(json.loads(path.read_text()),{'passed':True,'count':2})
            with self.assertRaises(FileExistsError):write_json(path,{'passed':False})
            self.assertTrue(json.loads(path.read_text())['passed'])
            invalid=Path(folder)/'invalid.json'
            with self.assertRaises(ValueError):write_json(invalid,{'bad':float('nan')})
            self.assertFalse(invalid.exists())
