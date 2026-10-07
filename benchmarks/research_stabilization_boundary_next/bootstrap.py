from pathlib import Path
import argparse
from .provenance import *
from .lineage import default_source

def prepare(run):
    run=Path(run);verify(run);folder,chain=default_source(APP,APP_SHA)
    names=['base_config.py','config.py','spaces.py','run.py','physics.py']
    same={name:sha(ROOT/'benchmarks/research_stabilization_boundary_next'/name)==sha(ROOT/'benchmarks/research_restoring_rt0_next'/name) for name in names}
    if not all(same.values()):raise ValueError('solid core changed')
    write(run/'S0/source-map.json',dict(status='passed_scoped',direct_parent=str(APP),default_owner=str(folder.parent.parent),default_case=str(folder),chain=chain,
        observable_protocol=dict(path=str(OBS/'S3/new-scene-protocol.json'),sha256=sha(OBS/'S3/new-scene-protocol.json')),
        candidate_initial=read(APP/'S1/reference-protocol.json'),formal_reference=str(OBS/'cases/phase-quarter'),history=str(OLD)))
    write(run/'S0/impact-and-compatibility.json',dict(status='inherited',byte_identical_solid_modules=same,new_numeric_equations=False,new_dynamic_attempts=0,
        compatibility=dict(path=str(APP/'S0/compatibility.json'),sha256=sha(APP/'S0/compatibility.json')),
        rollback=dict(path=str(APP/'S2/fault-B0/failure.json'),sha256=sha(APP/'S2/fault-B0/failure.json')),
        restart=dict(path=str(APP/'cases/grid32-h/summary.json'),sha256=sha(APP/'cases/grid32-h/summary.json')),new_host_RSS_observer=True))
    print('SOURCE_MAP',folder,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):prepare(a.run)
