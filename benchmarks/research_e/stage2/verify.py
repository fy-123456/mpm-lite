"""Fresh-process source/evidence/operator consumer, run from the independent copy."""
import argparse
import json
from pathlib import Path
from engine.aniso_phase1.research_e.stage2.handoff import sha,load
from .run import ROOT,check_manifest


def verify(out):
    ext=json.loads((out/'extension-source-sha256.json').read_text());source=check_manifest(ROOT,ext)
    a=json.loads((out/'acceptance.json').read_text());e=json.loads((out/'E10_handoff.json').read_text())
    if sha(out/'extension-source-sha256.json')!=a['extension_source_sha256'] or sha(out/'protocol.json')!=a['protocol_sha256']:
        raise ValueError('unbound extension/protocol')
    m,M,v,residual=load(out/'handoff',expected_manifest_sha256=e['manifest_sha256'],expected=e['expected'])
    for name,record in m['physics_evidence'].items():
        if sha(out/(name+'.json'))!=record['artifact_sha256']:raise ValueError('changed physics evidence '+name)
    d=json.loads((out/'D-contract-consumer.json').read_text())
    if d.get('D_contract_consumed'):
        from engine.aniso_phase1.research_e.stage2.d_adapter import consume
        result=consume(None,m,M,v,out/'D-contract-source.py')
        if not result['passed'] or result['D_source_sha256']!=d['D_source_sha256']:raise ValueError('D consumer mismatch')
    else:result={'passed':False,'status':'not_run'}
    return dict(passed=True,source=source,source_root=str(ROOT),physics_evidence_count=len(m['physics_evidence']),
                original_residual_blocks=residual,D_contract_reload=result,scope='fresh process; no active-repository E imports')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);args=ap.parse_args()
    print(json.dumps(verify(args.output),indent=2))


if __name__=='__main__':main()
