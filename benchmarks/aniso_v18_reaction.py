"""Post-acceptance diagnosis of raw boundary reaction; never changes gates."""
import json
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v18_runs import ROOT,OUT,load,write,sha,initial
from benchmarks.aniso_dynamic_check import pk1
from engine.aniso_phase1.carrier_joint import State,Geometry,gradient
from engine.aniso_phase1.unresolved_velocity import pack

def rms(x):return float(np.sqrt(np.mean(np.square(x))))
def main():
    assert not (OUT/'reaction-diagnosis.json').exists();protocol=load(OUT/'protocol.json');cases={};records=[]
    write(OUT/'reaction-diagnosis-protocol.json',dict(source_sha256=sha(ROOT/'benchmarks/aniso_v18_reaction.py'),purpose='diagnose failed raw-reaction time gate after formal acceptance, not replace it',common_impulse_window_s=.0025,high_band='above 0.8 times Nyquist frequency; one-sided Parseval weights, mean removed',snapshots=40))
    for level in range(4):
        name=f'cycle-L{level}';spec=protocol['cases'][name];dt=spec['dt'];rows=[json.loads(s) for s in (OUT/'cases'/name/'steps.jsonl').read_text().splitlines()];R=np.array([r['reaction_N'] for r in rows]);t=dt*(np.arange(len(R))+.5);cases[name]=(dt,R,t,rows)
        phase={}
        for label,bounds in protocol['cycle'].items():
            if label=='peak_displacement':continue
            lo,hi=bounds;r=R[round(lo/dt):round(hi/dt)];center=r-r.mean();power=abs(np.fft.rfft(center))**2;power[1:-1]*=2;freq=np.fft.rfftfreq(len(r),dt);pair=r.reshape(-1,2)
            phase[label]=dict(raw_rms_N=rms(r),two_step_average_rms_N=rms(pair.mean(1)),alternating_rms_N=rms((pair[:,0]-pair[:,1])/2),high_band_power_fraction=float(power[freq>.8*.5/dt].sum()/power.sum()),lag1_correlation=float(np.corrcoef(center[:-1],center[1:])[0,1]))
        records.append(dict(case=name,dt=dt,phases=phase))
    pairs=[]
    for level in range(3):
        ca,fi=f'cycle-L{level}',f'cycle-L{level+1}';da,ra,ta,_=cases[ca];db,rb,tb,_=cases[fi]
        interp=rb.reshape(-1,2).mean(1);wa,wb=round(.0025/da),round(.0025/db);a,b=ra.reshape(-1,wa).mean(1),rb.reshape(-1,wb).mean(1)
        pairs.append(dict(coarse=ca,fine=fi,raw_as_formal_relative=rms(ra-rb[1::2])/rms(rb[1::2]),
            fine_midpoint_interpolated_relative=rms(ra-interp)/rms(interp),common_window_impulse_rate_relative=rms(a-b)/rms(b),
            common_window_impulse_rate_absolute_N=rms(a-b)))
    audits=[]
    for name,(dt,R,t,rows) in cases.items():
        spec=protocol['cases'][name];_,e,m,h,_=initial(spec)
        for path in sorted((OUT/'cases'/name).glob('audit-*.npz')):
            with np.load(path) as z:data={k:z[k] for k in z.files}
            s=State(data['x_before'],data['Y_before'],data['v_before'],data['C_before']);g=Geometry(s,e,m,h);W=data['W'];fixed=(g.nodes[:,0]*h<=.25)|(g.nodes[:,0]*h>=.75);grid=np.zeros((len(g.nodes),3));grid[g.nodes[:,0]*h>=.75,0]=1.;lift=la.lstsq(g.E[fixed],grid[fixed],cond=1e-11)[0]
            fm=np.zeros_like(s.Y);fs=fm.copy()
            for a in (.5-np.sqrt(3)/6,.5+np.sqrt(3)/6):
                Y=s.Y+a*dt*W;P=pk1(gradient(e.B,Y),e.A,200.);fm+=.5*sum(b.T@(e.V[:,None]*P[:,:,j]) for j,b in enumerate(e.B));r=e.P@Y[e.ids];local=e.weights[:,None,None]*(e.P.swapaxes(1,2)@r)
                f=np.zeros_like(Y);np.add.at(f,e.ids.ravel(),local.reshape(-1,3));fs+=.5*f
            zl=g.J@lift;z0=pack(s.v,s.C);z1=pack(data['v'],data['C']);q=g.metric
            Ri=2/dt*float(np.sum(q[:,None]*zl*(g.J@W-z0)));Rm=float(np.sum(lift*fm));Rs=float(np.sum(lift*fs));direct=float(np.sum(q[:,None]*zl*(z1-z0))/dt)
            step=int(path.stem.split('-')[-1]);logged=rows[step-1]['reaction_N'];err=abs(Ri+Rm+Rs-logged);assert err<1e-6 and abs(Ri-direct)<1e-6
            audits.append(dict(case=name,step=step,time=rows[step-1]['time'],inertia_N=Ri,material_N=Rm,stabilization_N=Rs,total_N=Ri+Rm+Rs,logged_N=logged,independent_error_N=err,momentum_identity_error_N=abs(Ri-direct)))
    write(OUT/'reaction-diagnosis.json',dict(completed=True,formal_gate_preserved=True,formal_passed=load(OUT/'cycle-acceptance.json')['passed'],records=records,pairs=pairs,independent_component_audits=audits,
        scope='Averaging is only a diagnostic of alternating constraint impulses. Raw reaction acceptance remains failed. Work can be accurate while a stationary constraint reaction oscillates.'))
    print('REACTION PAIRS',pairs,flush=True)
    print('FINAL HOLDS',[r['phases']['final_hold'] for r in records],flush=True)
    print('TERMINALS',[r for r in audits if abs(r['time']-1.6)<1e-10],flush=True)
if __name__=='__main__':main()
