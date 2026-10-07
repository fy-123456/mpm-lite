"""Supplemental physical hard-grip checks; no candidate or threshold changes."""
import argparse
import numpy as np
from engine.aniso_phase1.research_d.common_space import CommonSpace
from engine.aniso_phase1.research_d.identity import write_json
from .freeze_common import PIN


def audit(s):
    inner=[np.array([.15,.20,.80,.85]),np.array([.38,.50,.62]),np.array([.38,.50,.62])]
    faces=[np.array([.125,.25,.75,.875]),inner[1],inner[2]]
    errors={}
    for name,pts in (("interior",inner),("faces",faces)):
        x,F=s.evaluate(s.q0,pts)
        dx,dF=s.jvp(s.test_vectors["direction"],pts)
        X=np.stack(np.meshgrid(*pts,indexing="ij"),axis=-1)
        lift=np.zeros_like(X);lift[X[...,0]>=.75,0]=.005
        errors[name+"_displacement"]=float(np.max(abs(x-X-lift)))
        errors[name+"_direction"]=float(np.max(abs(dx)))
        if name=="interior":
            errors["rigid_gradient"]=float(np.max(abs(F-np.eye(3))))
            errors["rigid_gradient_direction"]=float(np.max(abs(dF)))
    return dict(passed=bool(max(errors.values())<=1e-7),absolute_errors=errors,
        tolerance=1e-7,space_manifest_sha256=s.signature,
        scope="physical rigid volumes; interface gradient is deliberately not constrained")


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument("--output",required=True);a=p.parse_args()
    from pathlib import Path
    out=Path(a.output);s=CommonSpace(out/"inputs",expected_manifest_sha256=PIN)
    result=audit(s);write_json(out/"boundary-audit.json",result)
    print(result)
    if not result["passed"]:raise SystemExit(2)

if __name__=="__main__":main()
