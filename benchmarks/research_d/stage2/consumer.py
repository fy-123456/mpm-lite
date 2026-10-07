"""Independent extension/parent reload and concrete refusal checks."""
import argparse,json,time
from pathlib import Path
import numpy as np
import warp as wp
from engine.aniso_phase1.research_d.stage2.extension import verify_extension
from engine.aniso_phase1.research_d.stage2.contracts import sha
from engine.aniso_phase1.research_d.frozen_inputs import load_frozen_inputs
from engine.aniso_phase1.research_d.stage2.gpu_operator import GPUOperator
from benchmarks.research_d.stage2.bootstrap import PARENT,PIN

def consume(folder,expected):
    root=Path.cwd();folder=Path(folder);begun=time.perf_counter();refusals={}
    def check(**kw):return verify_extension(folder,root,expected_sha256=kw.pop('pin',expected),**kw)
    result=check(require=['static_operator','cuda'])
    for name,kwargs in [('wrong_extension',dict(pin='0'*64)),('dynamic_claim',dict(require=['dynamic_cycle'])),('mixed_claim',dict(require=['coupled_physics']))]:
        try:check(**kwargs)
        except ValueError as exc:refusals[name]=str(exc)
        else:raise AssertionError('not rejected: '+name)
    manifest=json.loads((folder/'extension.json').read_text())
    try:load_frozen_inputs(root/PARENT,root,expected_sha256='0'*64)
    except ValueError as exc:refusals['wrong_parent']=str(exc)
    else:raise AssertionError('wrong parent accepted')
    missing=root/PARENT/'inputs/basis-transform.npz';original=missing.read_bytes()
    try:
        missing.unlink()
        try:check()
        except ValueError as exc:refusals['missing_parent_member']=str(exc)
        else:raise AssertionError('missing member accepted')
    finally:missing.write_bytes(original)
    for label,path in [('changed_extension_source',root/'engine/aniso_phase1/research_d/stage2/contracts.py'),('changed_parent_source',root/'engine/aniso_phase1/research_d/common_space.py'),('changed_input',root/PARENT/'inputs/geometry.npz')]:
        original=path.read_bytes()
        try:
            path.write_bytes(original+b'\n# corruption refusal test\n')
            try:check()
            except ValueError as exc:refusals[label]=str(exc)
            else:raise AssertionError('not rejected: '+label)
        finally:path.write_bytes(original)
    path=root/PARENT/'inputs/geometry.npz';outside=root.parent/(root.name+'-untrusted-geometry.npz')
    original=path.read_bytes()
    try:
        outside.write_bytes(original);path.unlink();path.symlink_to(outside)
        try:check()
        except ValueError as exc:refusals['untrusted_external_input']=str(exc)
        else:raise AssertionError('external link accepted without trusted root')
    finally:
        path.unlink();path.write_bytes(original);outside.unlink()
    s,_,_,_=load_frozen_inputs(root/PARENT,root,expected_sha256=PIN)
    wp.config.kernel_cache_dir='/root/autodl-tmp/mpm-lite-research-d/stage2/consumer-warp-cache'
    op=GPUOperator(s)
    with np.load(folder/'gpu-equilibrium.npz') as z:q=z['q'];energy=float(z['U']);force=z['full_force']
    r=op.evaluate(q)
    agreement=abs(r['U']-energy)<1e-10 and np.linalg.norm(r['full_force']-force)<1e-9 and np.linalg.norm(r['force'])<1e-8
    if not agreement:raise AssertionError('independent static reload differs')
    return dict(passed=True,extension_sha256=expected,parent=check()['parent'],refusals=refusals,
        independent_checkout=str(root),dereferenced_inputs=True,final_residual=float(np.linalg.norm(r['force'])),
        min_detF=r['min_detF'],seconds=time.perf_counter()-begun)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--sha256',required=True);p.add_argument('--receipt',required=True);a=p.parse_args()
    result=consume(a.folder,a.sha256);Path(a.receipt).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
