"""Read-only verification of this portable release, not the absent historical archive."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(smoke=False):
    m=json.loads((ROOT/'github-upload-manifest.json').read_text())
    for n,h in m['payload_sha256'].items():
        p=ROOT/n
        if not p.resolve().is_relative_to(ROOT):raise ValueError('external path: '+n)
        if not p.is_file() or sha(p)!=h:raise ValueError('missing/changed file (run python tools/restore_space_cache.py if applicable): '+n)
    for n,h in m['source_inventory'].items():
        if sha(ROOT/n)!=h:raise ValueError('changed frozen numerical source: '+n)
    run=ROOT/m['release']
    if sha(run/'release.json')!=m['release_sha256']:raise ValueError('changed original release')
    pub=json.loads((run/'release.json').read_text());index=run/pub['artifacts']['path']
    if sha(index)!=pub['artifacts']['sha256']:raise ValueError('changed artifact index')
    artifacts=json.loads(index.read_text())
    for n,h in artifacts.items():
        if sha(run/n)!=h:raise ValueError('changed sealed artifact: '+n)
    result=dict(payload_files=len(m['payload_sha256']),frozen_sources=len(m['source_inventory']),sealed_artifacts=len(artifacts),release_sha256=m['release_sha256'],full_historical_archive=False)
    if smoke:
        sys.path.insert(0,str(ROOT))
        import numpy as np
        from engine.aniso_phase1.research_absolute_state_next.state import MaterialStateModel
        from engine.aniso_phase1.research_formal_pressure_next.state import load
        model=MaterialStateModel();case=run/'S4/closed-compressed'
        identity=json.loads((case/'identity.json').read_text());state=load(case/'checkpoint.npz',model,identity)
        if state.digest()!=json.loads((case/'summary.json').read_text())['state_digest']:raise ValueError('changed restored state')
        with np.load(case/'replay-probes.npz',allow_pickle=False) as d:
            fields=model.fields(state,d['X']);gaps={name:float(abs(x-d[name]).max()) for name,x in zip(('x','F','v'),fields)}
        if max(gaps.values())>1e-10:raise ValueError('changed probe reconstruction')
        result['smoke']=dict(state_digest=state.digest(),probe_gaps=gaps,read_only=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    print(json.dumps(verify(args.smoke),indent=2))
