"""Verify the complete frozen common input, without running a simulation."""
import argparse
import json
from engine.aniso_phase1.research_d.frozen_inputs import verify_frozen_inputs
from engine.aniso_phase1.research_d.identity import ROOT


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("folder")
    p.add_argument("--trusted-data-root")
    p.add_argument("--sha256")
    p.add_argument("--dynamic",action="store_true")
    a=p.parse_args()
    print(json.dumps(verify_frozen_inputs(a.folder,ROOT,trusted_data_root=a.trusted_data_root,
          expected_sha256=a.sha256,require_dynamic=a.dynamic),indent=2))

if __name__=="__main__":main()
