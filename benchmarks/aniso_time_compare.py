"""Compare identical F45 load trajectories without changing the acceptance norms."""
import json
from pathlib import Path
import numpy as np
from benchmarks.aniso_dynamic_space import stress,relative
from benchmarks.aniso_projected_history import read_case
from benchmarks.aniso_mainline import write_json
LEVELS=('coarse','fine','finest','fourth')
ROOT=Path(__file__).resolve().parents[1]


def compare(out,configs,prefix='F45',save=None):
    data={};frames={};checks={};times=None;energies=[]
    for level in LEVELS:
        name=prefix+'-'+level;folder=out/'cases'/name
        data[level],checks[level]=read_case(folder)
        with np.load(folder/'frames.npz') as z:
            t=z['time'].copy();F=z['F'].copy()
        keep=t>=.05-1e-12;t=t[keep];F=F[keep]
        if times is None:times=t
        else:np.testing.assert_allclose(t,times,rtol=0,atol=1e-12)
        P=stress(F.reshape(-1,3,3),configs[name]).reshape(F.shape)
        frames[level]=dict(F=F-np.eye(3),P=P)
        rows=data[level];rebuild=sum(r['stabilization_rebuild_delta'] for r in rows)
        energies.append(dict(level=level,rebuild_net_J=rebuild,rebuild_absolute_J=sum(abs(r['stabilization_rebuild_delta']) for r in rows),final_elastic_J=rows[-1]['elastic'],final_stabilization_J=rows[-1]['stabilization_energy'],rebuild_over_elastic=rebuild/rows[-1]['elastic'],max_kinetic_J=max(r['kinetic'] for r in rows),final_reaction_N=rows[-1]['right_force']))
    tR=np.array([r['time'] for r in data['coarse']]);tR=tR[tR>=.05-1e-12]
    rmsR=lambda a:float(np.sqrt(np.trapezoid(a*a,tR)/(tR[-1]-tR[0])))
    norm=lambda a:float(np.sqrt(np.trapezoid(np.mean(np.sum(a*a,axis=(2,3)),axis=1),times)/(times[-1]-times[0])))
    pairs=[]
    for i,(a,b) in enumerate(zip(LEVELS[:-1],LEVELS[1:])):
        ra=np.interp(tR,[r['time'] for r in data[a]],[r['right_force'] for r in data[a]])
        rb=np.interp(tR,[r['time'] for r in data[b]],[r['right_force'] for r in data[b]])
        r=dict(pair=[a,b],reaction_absolute=rmsR(ra-rb),reaction_relative=rmsR(ra-rb)/max(rmsR(rb),.001),energy_relative=abs(data[a][-1]['elastic']-data[b][-1]['elastic'])/abs(data[b][-1]['elastic']))
        for key in ('F','P'):
            fa,fb=frames[a][key],frames[b][key];r[key+'_absolute']=norm(fa-fb);r[key+'_relative']=r[key+'_absolute']/norm(fb);r[key+'_terminal_relative']=relative(fa[-1],fb[-1])
        if i:
            for key in ('reaction','F','P'):r[key+'_order']=-float(np.log2(r[key+'_absolute']/pairs[-1][key+'_absolute']))
        pairs.append(r)
    audits=[json.loads(f.read_text()) for level in LEVELS for f in (out/'cases'/(prefix+'-'+level)).glob('audit-*.json')]
    r=dict(completed=all(x['completed'] for x in checks.values()),physical_passed=all(x['passed'] for x in checks.values()),checks=checks,pairs=pairs,energy=energies,history_closure_max=max(x['frozen_F_max'] for x in audits),particle_commit_max=max(x['particle_F_update_max'] for x in audits),time_2pct_passed=max(pairs[-1][key] for key in ('reaction_relative','F_relative','P_relative','F_terminal_relative','P_terminal_relative','energy_relative'))<=.02,orders_passed=all(r[k+'_order']>=.5 for r in pairs[1:] for k in ('reaction','F','P')),field_samples=len(times),volume_weight='uniform particle reference volumes')
    if save:write_json(save,r)
    return r

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(__doc__);p.add_argument('folder',type=Path);p.add_argument('--prefix',default='F45');a=p.parse_args();protocol=json.loads((a.folder/'protocol.json').read_text())
    r=compare(a.folder,protocol['configs'],a.prefix,a.folder/(a.prefix+'-comparison.json'));print(json.dumps({k:v for k,v in r.items() if k!='checks'},indent=2))
