"""Read-only load of the explicitly inherited default; never advances a sealed case."""
from pathlib import Path
import argparse
from .provenance import *
from .publication import audit_release
from .spaces import load_selected
from benchmarks.research_sequential_next.checkpoint import GenerationStore


def inspect(run):
    run=Path(run);verify(run)
    if (run/'release.json').exists():audit_release(run,True)
    decision=read(run/'S6/default-scene-decision.json');name=decision['default_case'];folder=APP/'cases'/name;cfg=read(folder/'execution-protocol.json');identity=read(folder/'identity.json');store=GenerationStore(folder,identity);store.history();item=store.load()
    from benchmarks.research_pressure_window_next.provenance import source_files as original_sources
    if identity['numerical_source_sha256']!=original_sources():raise ValueError('inherited numerical identity changed')
    if cfg['post_release']['rule_policy']!='full_only':
        from engine.aniso_phase1.research_basis_allocation_next.rules import permission
        certificate=Path(cfg['post_release']['qualification']['path'])
        if sha(certificate)!=cfg['post_release']['qualification']['sha256']:raise ValueError('inherited certificate changed')
        allowed,reason=permission(read(certificate),cfg,numerical_sources=original_sources())
        if not allowed:raise ValueError('inherited permission rejected: '+reason)
    from engine.aniso_phase1.research_phase_boundary_next.segments import VectorizedModel
    from engine.aniso_phase1.research_phase_stress_next.warm import install_warm
    from engine.aniso_phase1.research_sequential_next.scenarios import install_peak
    import warp as wp
    wp.config.kernel_cache_dir=str(run.resolve()/'warp-cache');r,_=load_selected(cfg['physical_space']);m=install_warm(VectorizedModel(r,order=cfg['material_order'],device=cfg['device']));install_peak(m,cfg['scenario']['peak_m'])
    if identity['model']!=m.identity:raise ValueError('inherited model mismatch')
    m.validate(item['state'],material=True)
    if item['state'].child_states['identity']!=m.identity:raise ValueError('unexpected active fallback material in inherited final state')
    return dict(status='passed_scoped',case=name,source_release=str(APP),source_release_sha256=APP_SHA,accepted_rows=len(item['rows']),state_step=item['state'].step,time_s=item['state'].time,state_digest=item['state'].digest(),mass_sha256=digest(m.M.tolist()),active_material=item['state'].child_states['identity']['material'],zero_step=True,in_place_resume_allowed=False,permission_scope='parent certificate only; no child q5 extension')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--report',type=Path);a=p.parse_args()
    with serial_lock(a.run):
        result=inspect(a.run)
        if a.report:
            if (a.run/'release.json').exists():raise ValueError('sealed inspection is read only')
            write(a.report,result)
        print(result,flush=True)
