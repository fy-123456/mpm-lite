"""Same-state projection attribution. These are NOT alternative trajectories."""
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v19_runs import OUT,write
from benchmarks.aniso_v19_analysis import arrays,readrows
from benchmarks.aniso_v17_modes import controlled_case
from engine.aniso_phase1.carrier_joint import State,Geometry
from engine.aniso_phase1.unresolved_velocity import pack

def basis(A,q):
    U,s,_=la.svd(np.sqrt(q)[:,None]*A,full_matrices=False);return U[:,s>1e-12*s[0]]
def main():
    _,e,m,h,_=controlled_case();records=[]
    for level in range(4):
        folder=OUT/'cases'/f'cycle-L{level}';rows=readrows(folder/'steps.jsonl')
        for path in sorted(folder.glob('audit-*.npz')):
            a=arrays(path);k=int(path.stem.split('-')[-1]);row=rows[k-1];old=State(a['x_before'],a['Y_before'],a['v_before'],a['C_before']);g=Geometry(old,e,m,h);fixed=(g.nodes[:,0]*h<=.25)|(g.nodes[:,0]*h>=.75);grid=np.zeros((len(g.nodes),3));grid[g.nodes[:,0]*h>=.75,0]=1;lift=la.lstsq(g.E[fixed],grid[fixed],cond=1e-11)[0]
            trial=a['midpoint_trial_z'];variants=[('old_geometry',a['J'],a['Q'],a['q'],lift),('end_maps_old_clamp_old_metric',a['endJ'],a['Q'],a['q'],lift),('end_maps_old_clamp_end_metric',a['endJ'],a['Q'],a['endq'],lift),('actual_end',a['endJ'],a['endQ'],a['endq'],a['endpoint_lift'])];rr=[]
            for name,J,Q,q,l in variants:
                U=basis(J,q);V=basis(J@Q,q);root=np.sqrt(q)[:,None]
                def normal(z):return (U@(U.T@(root*z))-V@(V.T@(root*z)))/root
                delta=normal(J@l*row['endpoint_speed']-trial);imp=J.T@(q[:,None]*delta);rec=dict(name=name,reaction_N=float(np.sum(imp*l)/row['dt']),loss_J=.5*float(np.sum(q[:,None]*delta*delta)))
                if name=='actual_end':
                    U0=basis(a['J'],a['q']);root0=np.sqrt(a['q'])[:,None];z0=pack(old.v,old.C);null=z0-U0@(U0.T@(root0*z0))/root0;dn=-normal(null);dr=delta-dn
                    rec.update(null_history_reaction_N=float(np.sum((J.T@(q[:,None]*dn))*l)/row['dt']),resolved_reaction_N=float(np.sum((J.T@(q[:,None]*dr))*l)/row['dt']),null_history_kinetic_J=.5*float(np.sum(a['q'][:,None]*null*null)))
                    assert abs(rec['reaction_N']-row['endpoint_reaction_N'])<1e-8
                rr.append(rec)
            records.append(dict(case=folder.name,time=row['time'],variants=rr))
    write(OUT/'constraint-projection-parts.json',dict(completed=True,records=records,scope='Same saved midpoint trial velocity and endpoint speed. Sequential map/metric/clamp substitutions do not solve new trajectories and are order dependent. Null/resolved contributions include cancellation; no isolated causal claim.'))
    print('completed',len(records),flush=True)

def reaction_error_parts():
    from benchmarks.aniso_v19_analysis import rms
    a=readrows(OUT/'cases/cycle-L2/steps.jsonl');b=readrows(OUT/'cases/cycle-L3/steps.jsonl')[1::2];records={}
    for lo,hi,name in [(0,1.6,'whole'),(1.1,1.6,'final_hold')]:
        mask=np.array([lo-1e-12<x['time']<=hi+1e-12 for x in a]);d={k:np.array([x[k]-y[k] for x,y in zip(a,b)])[mask] for k in ['reaction_N','midpoint_reaction_N','endpoint_reaction_N']}
        records[name]={k:rms(v) for k,v in d.items()};records[name]['squared_error_cross_N2']=2*float(np.mean(d['midpoint_reaction_N']*d['endpoint_reaction_N']))
    write(OUT/'reaction-error-parts.json',dict(completed=True,comparison='125 vs 62.5 microseconds; same raw-force convention as formal gate',records=records,scope='Exact additive error split; cross terms retained. Not an independent causal experiment.'))
if __name__=='__main__':
    import sys
    if len(sys.argv)>1 and sys.argv[1]=='reaction':reaction_error_parts()
    else:main()
