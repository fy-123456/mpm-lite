"""Negative checks on the actual final-space certificate, without integration."""
from pathlib import Path
import argparse,copy
from .provenance import *
from engine.aniso_phase1.research_basis_allocation_next.rules import permission

def review(run):
    run=Path(run);q=read(run/'S6/qualification-final.json');cfg=read(run/'cases/final-full/execution-protocol.json');records=[]
    if not q['qualified']:
        write(run/'S5/scope-boundaries.json',dict(status='not_triggered',reason='main compression not qualified'));return
    allowed,_=permission(q,cfg,numerical_sources=source_files())
    if not allowed:raise ValueError('actual main path rejected')
    for name in ('old_space','old_sources','different_peak','skip_time_node','unknown_initial','mass_order','beyond_end'):
        bad=copy.deepcopy(cfg);proof=copy.deepcopy(q)
        if name=='old_space':bad['physical_space']=read(PHASE_REFERENCE/'selected-space.json')['package']
        elif name=='old_sources':proof['numerical_source_sha256']={}
        elif name=='different_peak':bad['scenario']['peak_m']=.0075
        elif name=='skip_time_node':bad['times'].pop(2)
        elif name=='unknown_initial':bad['initial_state']={'path':'unused','sha256':'unregistered'}
        elif name=='mass_order':bad['mass_order']=8
        else:bad['times'].append(1.6125)
        ok,reason=permission(proof,bad,numerical_sources=source_files())
        if ok:raise ValueError('final permission escaped: '+name)
        records.append(dict(request=name,rejected=True,reason=reason))
    write(run/'S5/scope-boundaries.json',dict(status='passed_scoped',records=records,certificate_sha256=sha(run/'S6/qualification-final.json'),main_path_allowed=True,new_sensitive_window_license=False))
    print('ACTUAL_SCOPE_REJECTIONS',len(records),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):review(a.run)
