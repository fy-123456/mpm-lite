"""Freeze the latest lineage and distinguish inherited production ownership."""
from .provenance import *
from .lineage import default_source

def prepare():
    run=freeze(); folder,chain=default_source(APP,APP_SHA)
    names=['base_config.py','config.py','spaces.py','run.py','physics.py']
    same={n:sha(ROOT/'benchmarks/research_pressure_startup_next'/n)==sha(ROOT/'benchmarks/research_stabilization_boundary_next'/n) for n in names}
    if not all(same.values()):raise ValueError('unexpected solid numerical change')
    write(run/'S0/source-map.json',dict(status='passed_scoped',direct_parent=str(APP),previous=str(PREV),observable=str(OBS),default_case=str(folder),chain=chain,candidate_prefix=str(APP/'cases/candidate-prefix-half'),origin=read(APP/'S3/short-reference-protocol.json')['origin_digest']))
    write(run/'S0/impact-and-compatibility.json',dict(status='inherited',byte_identical_solid_modules=same,new_S0_steps=0,source= str(APP/'S0/impact-and-compatibility.json'),rollback=str(APP/'S5/fault-B0/failure.json'),restart=str(APP/'S5/restart.json'),implementation_bridge_not_method_history=True))
    print(run,flush=True)

if __name__=='__main__':prepare()
