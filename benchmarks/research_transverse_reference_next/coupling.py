"""Source-bound DV bridge and a single bounded continuation; sealed parents read only."""
import argparse, time
import numpy as np
import warp as wp
from .provenance import *
from .runtime import attempt, update
from .fixture import setup
from benchmarks.research_phase_stress_next.time_study import history
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_transverse_next.trajectory import equivalence, instrument
from benchmarks.research_transverse_next.observables import modes, face_signals, spatial_statistics
from benchmarks.research_sequential_next.compare import regions, metric
from benchmarks.research_pressure_window_next.coupling_study import frame
from benchmarks.research_continuous_geometry_next.review import balances
from engine.aniso_phase1.research_transverse_next.recovery import SafePublication
from engine.aniso_phase1.research_post_release.fields import CachedProbes


def bind(folder, ident):
    write(folder/'identity.json',ident);snapshot(folder/'source',ident['numerical_sources'])


def field_stats(f, fiber):
    result={}
    for region,w in regions(f['X']).items():
        r={k:spatial_statistics(f['X'],f[k]-f['X'] if k=='x' else f[k],w) for k in ('x','velocity','PK1','PK1_total','Cauchy_skeleton','Cauchy_total')}
        r['fiber_PK1_total']=spatial_statistics(f['X'],np.einsum('i,...ij,j->...',fiber,f['PK1_total'],fiber),w)
        result[region]=r
    return result


def start_store(run, name, c, ident, source):
    folder=Path(run)/name;store=GenerationStore(folder,ident)
    if store.load() is None:
        c.restore(source['state']);bind(folder,ident);store.save(c.state,[])
        write(folder/'parent-history.json',dict(path=str(source['folder']),state_sha256=sha(source['folder']/'state.json'),digest=source['state'].digest(),ledger_sha256=sha(source['folder']/'ledger.json'),original_global_step=c.state.step,initial_contract_sha256=c.contract_id,local_ledger_starts_empty=True,cumulative_physical_history_preserved=True))
    else:
        if read(folder/'identity.json')!=ident:raise ValueError('executed source changed')
        store.history();c.restore(store.load(validator=c.validate)['state'])
    return folder,store


def bridge(run):
    run=Path(run);mutable(run);tick=time.perf_counter();c,m,cfg,ident=setup(run);build=time.perf_counter()-tick
    h=history(APP/'cases/YZ128');a,b=h[8:10]
    folder,store=start_store(run,'S2/bridge',c,ident,a);safe=SafePublication(c,store);cache=CachedProbes(m)
    source_frame=frame(c,cache,b['state']);c.geometry.cache.clear();c.geometry.profile={};timings=instrument(c,m)
    row=attempt(run,'S2','DV-bridge',safe.advance);actual=frame(c,cache,c.state)
    checks=equivalence(c.state,b['state'],row,b['rows'][-1]);checks.update({'field_'+k:metric(actual[k],source_frame[k],1e-8,2e-5) for k in ('x','velocity','PK1','PK1_total','Cauchy_total')})
    bad=c.state.clone();bad.child_states['transverse_initial_contract']='foreign'
    try:c.validate(bad)
    except ValueError:rejected=True
    else:rejected=False
    bad=c.state.clone();bad.child_states['explicit_pressure_grid']='foreign'
    try:c.validate(bad)
    except ValueError:grid_rejected=True
    else:grid_rejected=False
    passed=all(x['passed'] for x in checks.values()) and rejected and grid_rejected and balances(store.history())['passed']
    write(run/'S2/bridge-check.json',dict(status='passed_scoped' if passed else 'failed',checks=checks,foreign_initial_rejected=rejected,foreign_grid_rejected=grid_rejected,balances=balances(store.history()),physical_identity_equal=ident['coupling']==read(APP/'cases/YZ128/identity.json')['coupling']))
    write(run/'S2/backend-binding.json',dict(status='passed_scoped',actual_class=type(c.geometry).__name__,identity=ident,full_mass_sha256=digest(m.M.tolist()),full_material_order=7,full_mass_order=7,source_initial_digest=h[0]['state'].digest(),constructed_initial_digest=c.initial_digest))
    write(run/'S3/DV-profile.json',dict(status='limited',geometry=c.geometry.profile,material=timings,publication=safe.profile,build_s=build,seconds=time.perf_counter()-tick,scope='one diagnostic bridge; local_gradient includes gather/adjoint/download/P; material hooks nested, no additive total claim',extra_profile_steps=0))
    write(run/'S2/recovery-scope.json',dict(status='inherited',source=str(APP/'S2/transaction-check.json'),sha256=sha(APP/'S2/transaction-check.json'),reason='same SafePublication, GenerationStore and transient DV lifetime; execution adapter only',new_fault_steps=0,real_driver_loss_recovery=False))
    if not passed:raise ValueError('DV bridge differs')
    update(run,'S2.1/2：新研究工厂显式接入DV，第8→9步物理状态、同位置场、账本与旧结果等价；错初态/错网格拒绝通过。恢复事务与DV临时缓冲区语义未改，继承已有受控故障证据。')
    print('BRIDGE',passed,c.geometry.profile,flush=True)


def whole_balance(h, local):
    first=h[0]['state'];last=local[-1]['state'];f=first.child_states['fluid'];g=last.child_states['fluid'];rows=h[18]['rows']+local[-1]['rows']
    E0=read(APP/'cases/YZ128/execution-protocol.json')['initial_energy_J']
    mass=float(np.sum(np.array(g['content_m3'])-f['content_m3'])+g['cumulative_boundary_m3']-np.sum(g['cumulative_source_m3']))
    energy=rows[-1]['total_energy_J']-E0+sum(r['darcy_dissipation_J']+r['numerical_dissipation_J']-r['external_work_J']-r['source_work_J']-r['reservoir_work_J'] for r in rows)
    return dict(mass_defect_m3=mass,energy_defect_J=energy,original_initial_energy_J=E0,global_rows=len(rows),passed=abs(mass)<=1e-10 and abs(energy)<=1e-9+.01*abs(E0))


def extension(run, stop):
    run=Path(run);mutable(run);decision=read(run/'S1/window-and-time-decision.json')
    if decision['decision']!='extend4' or read(run/'S2/bridge-check.json')['status']!='passed_scoped':raise ValueError('extension gate not passed')
    if stop not in (22,26) or (stop==26 and not decision['additional_four_if_needed']):raise ValueError('unregistered extension')
    tick=time.perf_counter();c,m,cfg,ident=setup(run);h=history(APP/'cases/YZ128')
    folder,store=start_store(run,'cases/YZ128-DV',c,ident,h[18]);start=c.state.step
    safe=SafePublication(c,store);cache=CachedProbes(m);samples=[]
    while c.state.step<stop:
        endpoint=c.state.step+1 in (22,26)
        row=attempt(run,'S2','YZ128-extension',lambda:safe.advance(frame_builder=(lambda s:frame(c,cache,s)) if endpoint else None))
        f=frame(c,cache,c.state)
        sample=dict(step=c.state.step,time_s=c.state.time,modes=modes(c.geometry.topology,c.state.child_states['fluid']['pressure_Pa']),faces=face_signals(c.geometry.topology,c.state.child_states['fluid']['flux_interval_m3_s']),fields=field_stats(f,m.parent.params.fiber_direction))
        write(folder/f'observation-{c.state.step}.json',sample);samples.append(sample)
        local=store.history();bal=balances(local)
        if not bal['passed'] or abs(row['energy_balance_J'])>1e-9 or not whole_balance(h,local)['passed']:raise ValueError('extension physical hard condition failed')
        print('EXTEND',c.state.step,c.state.time,sample['modes'],flush=True)
    local=store.history();bal=balances(local)
    write(run/'S2/extension-review.json',dict(status='passed_scoped',start_global_step=18,final_global_step=c.state.step,time_s=c.state.time,new_steps=c.state.step-18,balances=bal,whole_history=whole_balance(h,local),spatial_accuracy=False,temporal_accuracy=False,scope='actual coupled stability and ledgers only; fixed-skeleton comparison is model difference',backend='DV'))
    write(folder/f'profile-{start}-{stop}.json',dict(geometry=c.geometry.profile,publication=safe.profile,seconds=time.perf_counter()-tick))
    update(run,f'S2.3：YZ128使用DV从第{start}步推进至第{stop}步/{c.state.time*1e6:g}微秒，区间和全历史守恒通过。最大含量缺口{bal["mass_defect_m3"]:.3g}m³，能量缺口{bal["incremental_energy_balance_J"]:.3g}J；不宣称动态参考精度。')


def load(run):
    run=Path(run);c,m,cfg,ident=setup(run);folder=run/'cases/YZ128-DV'
    if not (folder/'identity.json').exists():folder=run/'S2/bridge'
    if read(folder/'identity.json')!=ident:raise ValueError('fresh-process identity differs')
    store=GenerationStore(folder,ident);store.history();record=store.load(validator=c.validate);c.restore(record['state'])
    write(run/'S5/load-check.json',dict(status='passed_scoped',fresh_process=True,zero_steps=True,state_digest=c.state.digest(),step=c.state.step,initial_contract=c.contract_id,physical_history_preserved=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['bridge','extension','load']);p.add_argument('--run',type=Path,required=True);p.add_argument('--stop',type=int,default=22);a=p.parse_args()
    if a.phase=='bridge':bridge(a.run)
    elif a.phase=='extension':extension(a.run,a.stop)
    else:load(a.run)
