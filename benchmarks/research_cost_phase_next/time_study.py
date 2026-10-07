"""Read-only reuse of authenticated modal windows; no redundant time sweep."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import APP,read,write,sha,register,verify,serial_lock
from benchmarks.research_sequential_next.checkpoint import GenerationStore

def failures(value,path=''):
    out=[]
    if isinstance(value,dict):
        if value.get('passed') is False:out.append(dict(path=path,**{k:v for k,v in value.items() if not isinstance(v,(dict,list))}))
        for k,v in value.items():out+=failures(v,path+'/'+k)
    elif isinstance(value,list):
        for i,v in enumerate(value):out+=failures(v,path+'/'+str(i))
    return out

def study(run):
    run=Path(run);verify(run)
    register(run,'P2/protocol.json',dict(windows=[[1.,1.2],[1.2,1.4]],steps=[.0125,.00625,.003125],
        scope='physical space unchanged, equivalent implementation checked in P1; authenticate existing numerical evidence',
        candidate='fine from .6 s onward has identical local windows to existing .00625 trials; no new local evidence expected'))
    if read(run/'selected-space.json')['package']['sha256']!=read(APP/'selected-space.json')['package']['sha256']:
        raise ValueError('window reuse requires same physical space')
    records=[];results=[]
    for i in (0,1):
        histories=[]
        for suffix in ('0125','00625','003125'):
            folder=APP/'cases'/f'phase-{i}-{suffix}';identity=read(folder/'identity.json')
            h=GenerationStore(folder,identity).history();histories.append(h)
            records.append(dict(case=str(folder),identity_sha256=sha(folder/'identity.json'),
                protocol_sha256=sha(folder/'execution-protocol.json'),initial_sha256=sha(h[0]['folder']/'state.json'),generations=len(h)))
        first=histories[0][0]['state']
        for h in histories[1:]:
            for key in ('q','velocity','predictor'):
                assert np.array_equal(getattr(first,key),getattr(h[0]['state'],key))
            assert first.child_states==h[0]['state'].child_states
        result=read(APP/f'N2/window{i}.json')
        results.append(dict(window=result['window'],band_gain=result['band_gain'],phase_passed=result['phase_passed'],
            accepted=result['accepted'],failed_metrics=failures(result),events=result['medium_vs_fine']['modal_events']))
    modal=read(APP/'N2/modal-definition.json');T=np.array(modal['periods_s']);high=T<=.025
    resolution=[]
    for h in (.0125,.00625,.003125):
        ratio=2*np.arctan(np.pi*h/T)/(h*2*np.pi/T)
        resolution.append(dict(dt_s=h,high_band_modes=int(high.sum()),high_band_below_four_steps_per_period=int(np.sum(high&(T/h<4))),
            high_band_below_eight_steps_per_period=int(np.sum(high&(T/h<8))),
            high_band_linear_frequency_ratio_range=[float(ratio[high].min()),float(ratio[high].max())]))
    write(run/'P2/reuse-audit.json',dict(status='passed_scoped',same_initial_q_v_predictor_history=True,
        release_sha256=sha(APP/'release.json'),records=records,source_equivalence='P1/operator-equivalence.json'))
    write(run/'P2/failure-breakdown.json',dict(windows=results))
    write(run/'P2/phase-cause-analysis.json',dict(status='diagnostic',resolution=resolution,
        excitation_source_sha256=sha(APP/'N2/excitation-diagnostic.json'),
        established='existing finer windows improve main-band error, not all joint physical/event gates',
        inference='many high modes remain underresolved even by local fine comparator; linear dispersion plus common-time aliasing can limit interpretation',
        not_proved='not a complete root cause for nonlinear velocity or reaction differences',boundary_unchanged=True,no_damping=True))
    write(run/'P2/window-comparison.json',dict(status='reused_no_new_identical_sweep',windows=results,extra_windows=0))
    write(run/'P2/time-decision.json',dict(status='retain_dt_joint_gate_not_met',dt_s=.0125,
        times=read(APP/'N2/time-decision.json')['times'],temporal_accuracy=False,scheme='unchanged explicit grid',
        reason='proposed .6-onward fine schedule repeats already failed local .00625 gates; no justified fourth step size',source=records))
    print('TIME_REUSED retained .0125',[(x['window'],len(x['failed_metrics'])) for x in results],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
