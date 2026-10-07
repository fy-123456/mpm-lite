"""Additional full-trajectory field/order audit at the saved common time samples."""
import json
from pathlib import Path
import numpy as np
from benchmarks.aniso_dynamic_space import stress
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/results/lite-aniso-mainline/v11'


def main():
    p=json.loads((OUT/'protocol.json').read_text());all_rows=[];energy_rows=[]
    levels=['coarse','fine','finest','fourth']
    for label in ('ISO','F0','F45','F90'):
        fields=[];times=None
        for level in levels:
            name=label+'-'+level
            with np.load(OUT/'cases'/name/'frames.npz') as z:
                t=z['time'].copy();F=z['F'].copy()
            keep=t>=.05-1e-12;t=t[keep];F=F[keep]
            if times is None:times=t
            else:np.testing.assert_allclose(t,times,atol=1e-12,rtol=0)
            P=stress(F.reshape(-1,3,3),p['configs'][name]).reshape(F.shape)
            fields.append({'F':F-np.eye(3),'P':P})
            data=[json.loads(line) for line in (OUT/'cases'/name/'steps.jsonl').read_text().splitlines()]
            net=sum(r['stabilization_rebuild_delta'] for r in data);absolute=sum(abs(r['stabilization_rebuild_delta']) for r in data)
            energy_rows.append(dict(case=name,stabilization_rebuild_net_J=net,stabilization_rebuild_absolute_J=absolute,final_stabilization_energy_J=data[-1]['stabilization_energy'],final_elastic_J=data[-1]['elastic'],net_rebuild_over_final_elastic=net/data[-1]['elastic']))
        norm=lambda a:float(np.sqrt(np.trapezoid(np.mean(np.sum(a*a,axis=(2,3)),axis=1),times)/(times[-1]-times[0])))
        results={}
        for key in ('F','P'):
            pairs=[]
            for i in range(3):
                a,b=fields[i][key],fields[i+1][key];error=norm(a-b);row=dict(pair=levels[i]+'-'+levels[i+1],absolute_space_time_RMS=error,relative=error/max(norm(b),1e-30))
                if i:row['rho']=error/pairs[-1]['absolute_space_time_RMS'];row['observed_order']=-float(np.log2(row['rho']))
                pairs.append(row)
            results[key]=dict(pairs=pairs,last_relative_passed=pairs[-1]['relative']<=.02,order_screen_passed=all(r['observed_order']>=.5 for r in pairs[1:]))
        all_rows.append(dict(case=label,fields=results,passed=all(r['last_relative_passed'] and r['order_screen_passed'] for r in results.values())))
    result=dict(completed=True,records=all_rows,all_passed=all(r['passed'] for r in all_rows),energy_rebuild=energy_rows,
        scope='additional diagnostic, not a retrospectively changed frozen primary gate; volume-normalized field RMS over common stored times in [.05,.5] s; only 19 stored times, not every integration step',samples=len(times))
    (OUT/'time-field-audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(all_rows,indent=2),flush=True)


if __name__=='__main__':main()
