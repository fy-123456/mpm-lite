import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from benchmarks.research_restoring_rt0_next.lineage import default_source
from benchmarks.research_restoring_rt0_next.provenance import sha

class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def put(self,name,data):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data));return p
    def test_direct_scene(self):
        self.put('a/cases/d/identity.json',{});p=self.put('a/release.json',dict(default_case='d'))
        f,c=default_source(p.parent,sha(p));self.assertEqual(f,p.parent/'cases/d');self.assertEqual(len(c),1)
    def test_inherited_owner_and_corruption(self):
        ident=self.put('owner/cases/d/identity.json',dict(model=1));proto=self.put('owner/cases/d/execution-protocol.json',dict(dt=.1));owner=self.put('owner/release.json',dict(default_case='d'))
        child=self.put('child/release.json',dict(default_case='d',default_case_source=dict(release_root=str(owner.parent),release_sha256=sha(owner),case='d',identity_sha256=sha(ident),execution_protocol_sha256=sha(proto))))
        f,c=default_source(child.parent,sha(child));self.assertEqual(f,ident.parent);self.assertEqual(len(c),2)
        ident.write_text('{}')
        with self.assertRaisesRegex(ValueError,'binding'):default_source(child.parent,sha(child))
    def test_wrong_release(self):
        p=self.put('a/release.json',dict(default_case='d'))
        with self.assertRaisesRegex(ValueError,'hash'):default_source(p.parent,'bad')
    def test_cycle(self):
        a=self.root/'a';b=self.root/'b'
        for here,there in [('a',b),('b',a)]:self.put(here+'/release.json',dict(default_case_source=dict(release_root=str(there),release_sha256='same',case='missing')))
        with patch('benchmarks.research_restoring_rt0_next.lineage.sha',return_value='same'):
            with self.assertRaisesRegex(ValueError,'cyclic'):default_source(a,'same')
    def test_missing_scene(self):
        p=self.put('a/release.json',dict(default_case='missing'))
        with self.assertRaisesRegex(ValueError,'missing'):default_source(p.parent,sha(p))

if __name__=='__main__':unittest.main()
