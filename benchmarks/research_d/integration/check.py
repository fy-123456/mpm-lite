"""Validate one sealed interface package without changing a production default."""
import argparse
import json
from engine.aniso_phase1.research_contracts import validate_package
from engine.aniso_phase1.research_d.identity import ROOT,BASELINE


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('package'); p.add_argument('--dynamic',action='store_true')
    a=p.parse_args()
    with open(a.package) as f: data=json.load(f)
    meta=validate_package(data,ROOT,BASELINE,require_dynamic=a.dynamic)
    print(json.dumps(dict(producer=meta.producer,validated=True,capabilities=meta.capabilities,
                         default_changed=False),indent=2))

if __name__=='__main__':main()
