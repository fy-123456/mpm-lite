"""Separate same-state assembly work from complete profiled step timings."""
from pathlib import Path
import argparse,time
import numpy as np
import scipy.linalg as la
from .provenance import *
from .physics import baseline_model
from engine.aniso_phase1.research_candidate_observable_next.fixture import ObservableCoupling
from engine.aniso_phase1.research_cross_direction_next.det_rt0 import det3
from benchmarks.research_phase_stress_next.time_study import history

def assemble_parts(g,F):
    t=g.topology;times=dict(indexing=0.,tensor=0.,basis=0.,contraction=0.,scatter=0.);H=np.zeros((t.nflux,t.nflux));axes=np.repeat(np.arange(3),2);invk=la.solve(g.mobility,np.eye(3),assume_a='pos');start=time.perf_counter()
    for cell in range(t.cells):
        tick=time.perf_counter();indices=np.flatnonzero(g.cell_ids==cell);local=np.zeros((6,6));bounds=t.cell_bounds[cell];ends=bounds[axes,np.tile([1,0],3)];times['indexing']+=time.perf_counter()-tick
        for start0 in range(0,len(indices),16384):
            tick=time.perf_counter();ix=indices[start0:start0+16384];f=F[ix];J=det3(f);A=(np.swapaxes(f,1,2)@(invk@f))/J[:,None,None];times['tensor']+=time.perf_counter()-tick
            tick=time.perf_counter();phi=(g.X[ix][:,axes]-ends)/t.V0[cell]*t.signs;w=g.total_weights[ix];times['basis']+=time.perf_counter()-tick
            tick=time.perf_counter()
            for a in range(6):
                wa=w*phi[:,a]
                for b in range(a,6):
                    v=float(np.dot(wa*phi[:,b],A[:,axes[a],axes[b]]));local[a,b]+=v
                    if a!=b:local[b,a]+=v
            times['contraction']+=time.perf_counter()-tick
        tick=time.perf_counter();face=t.faces[cell];H[np.ix_(face,face)]+=local;times['scatter']+=time.perf_counter()-tick
    return H,dict(times,total=time.perf_counter()-start)

def study(run):
    import warp as wp
    run=Path(run);verify(run);old=read(APP/'S5/profile-protocol.json');p=read(APP/'S3/new-scene-protocol.json');case=Path(old['source_case']);h=history(case);m,cfg=baseline_model(run,pressure=True);c=ObservableCoupling(m,cfg,p,'coarse',state=h[1]['state']);g=c.geometry;records=[]
    for index in (1,3):
        q=h[index]['state'].q;wp.synchronize_device(m.device);tick=time.perf_counter();F=g.field(q);wp.synchronize_device(m.device);field=time.perf_counter()-tick
        tick=time.perf_counter();host=F.numpy();transfer=time.perf_counter()-tick
        tick=time.perf_counter();H,J=g.topology.assemble(g.X,g.total_weights,g.cell_ids,host,g.mobility);normal=time.perf_counter()-tick
        replica,parts=assemble_parts(g,host)
        if not np.allclose(H,replica,atol=1e-8,rtol=2e-5):raise ValueError('timing decomposition changes H')
        records.append(dict(index=index,source_sha256=sha(h[index]['folder']/'state.json'),F_field_s=field,F_download_s=transfer,download_bytes=host.nbytes,original_assembly_s=normal,subparts=parts,detF=J,points=len(host)))
    ratio=np.median([(r['subparts']['tensor']+r['subparts']['contraction']+r['F_download_s'])/(r['original_assembly_s']+r['F_download_s']) for r in records]);eligible=ratio>.5
    write(run/'S2/hotspot-breakdown.json',dict(status='passed_scoped',records=records,isolated_same_state_no_dynamic_steps=True,old_step_profile_sha256=sha(APP/'S5/profile.json'),dominant_target_fraction=float(ratio)))
    register(run,'S2/optimization-protocol.json',dict(status='registered' if eligible else 'not_triggered',eligible=bool(eligible),one_candidate='device current-F RT0 chunk contraction; small local H download',source_case=str(case),indices=[1,3],with_frames=[False,True],source_protocol_sha256=sha(APP/'S3/new-scene-protocol.json'),extra_bytes_cap=268435456,initial_free_fraction=.02,paired_order=['A0','B0','B1','A1'],min_gain=.05,max_attempts=8,no_material_or_mass_change=True))
    print('HOTSPOT',eligible,ratio,records,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
