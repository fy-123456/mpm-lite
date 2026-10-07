"""Resolve inherited scenes by authenticated release records, without GPU work."""
from pathlib import Path
from .provenance import read,sha

def default_source(root,expected_sha):
    root=Path(root);seen=set();chain=[]
    while True:
        key=str(root.resolve())
        if key in seen:raise ValueError('cyclic inherited default source')
        seen.add(key)
        if sha(root/'release.json')!=expected_sha:raise ValueError('release hash mismatch')
        pub=read(root/'release.json');chain.append(dict(root=str(root),sha256=expected_sha))
        ref=pub.get('default_case_source')
        if ref is None:
            case=pub['default_case'];folder=root/'cases'/case
            if not (folder/'identity.json').is_file():raise ValueError('default case is missing')
            return folder,chain
        root=Path(ref['release_root']);expected_sha=ref['release_sha256'];case=ref['case']
        folder=root/'cases'/case
        if (folder/'identity.json').exists():
            if sha(root/'release.json')!=expected_sha:raise ValueError('owner release changed')
            for key,file in [('identity_sha256','identity.json'),('execution_protocol_sha256','execution-protocol.json')]:
                if sha(folder/file)!=ref[key]:raise ValueError('default case binding changed')
            return folder,chain+[dict(root=str(root),sha256=expected_sha)]
