"""Frozen-state template remap gates, rigid-pose controls and a short release."""
import argparse
import os
import csv
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import eigh

from engine.aniso_phase1.consistent_transfer import MaterialQ1,bent_nodes
from engine.aniso_phase1.template_remap import RemappedQ1,axes_for,audit_remap,audit_mode
from engine.aniso_phase1.mechanics_comparison import physical_modes
from engine.aniso_phase1.beam_reference import beam_matrices
from benchmarks.aniso_consistent_transfer import guard


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def plot_results(out,results):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/mpm-lite-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,result in zip(axes[:2],results):
        rows=[r for r in result['records'] if r['kind'] in ('shift','coarsen_x')]
        x=np.arange(len(rows))
        for j,(key,label) in enumerate([('remap_energy_relative','Net energy'),('absolute_local_energy_change_relative','Absolute local energy change'),('stress_relative','Stress difference')]):
            ax.bar(x+(j-1)*.23,[100*r[key] for r in rows],width=.23,label=label)
        ax.set_xticks(x,[r['kind'] for r in rows]);ax.set(title=f"Source grid {result['grid']}: rejected proposals",ylabel='Percent')
        ax.legend(fontsize=7);ax.grid(axis='y',alpha=.25)
    for result in results:
        for name,d in result['dynamics'].items():
            if not isinstance(d,dict):continue
            axes[2].plot([0]+[1000*r['time'] for r in d['rows']],np.r_[1.,[r['mechanical']/d['initial_energy'] for r in d['rows']]],
                         label=f"g{result['grid']} {name}")
    axes[2].set(xlabel='Time (ms)',ylabel='Mechanical energy / initial',title='Accepted nested refinement release')
    from matplotlib.ticker import MaxNLocator
    axes[2].xaxis.set_major_locator(MaxNLocator(5))
    axes[2].legend(fontsize=7);axes[2].grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'summary.png',dpi=160);plt.close(fig)


def run(grid,out,steps,dt):
    guard();run_start=time.perf_counter();start=time.perf_counter()
    old=MaterialQ1(grid,ppc_axis=4,field='smooth')
    source_setup_seconds=time.perf_counter()-start
    x=bent_nodes(old);xp=old.particles.N@x;Fp=old.gradient(old.particles,x)
    immutable=[a.copy() for a in (xp,Fp,old.particles.A,old.mass,old.particles.X)]
    physical,_=physical_modes(old.X)
    modes={name:physical[:,:,i] for i,name in enumerate(('stretch','shear','bend_y','bend_z','twist'))}
    _,_,free,_,_,matrices=beam_matrices(grid)
    _,vectors=eigh(matrices[1][free][:,free].toarray(),subset_by_index=(0,1))
    for i in range(2):
        p=np.zeros(x.size);p[free]=vectors[:,i];modes[f'weak_{i}']=p.reshape(-1,3)
    records=[];accepted={}
    for kind in ('identity','refine_x','shift','coarsen_x'):
        guard();new=RemappedQ1(old,axes_for(old,kind))
        proposed,r=audit_remap(old,new,x,xp,Fp)
        r.update(kind=kind,grid=grid,modes={name:audit_mode(old,new,v) for name,v in modes.items()})
        r['max_mode_increment_relative']=max(v['increment_relative'] for v in r['modes'].values())
        r['moderate_static_screen']=abs(r['remap_energy_relative'])<.05 and r['stress_relative']<.1 and r['max_mode_increment_relative']<.15
        records.append(r)
        if r['dynamic_accepted'] and kind=='refine_x':accepted[kind]=new
        np.savez_compressed(out/f'g{grid}-{kind}.npz',reference=old.particles.X,positions_before=xp,
                            proposed_positions=new.particles.N@proposed,
                            strain_difference=np.linalg.norm(new.gradient(new.particles,proposed)-Fp,axis=(1,2)),
                            old_nodes=x,new_nodes=proposed,old_reference_nodes=old.X,new_reference_nodes=new.X,
                            old_shape=np.array(old.shape),new_shape=np.array(new.shape))
        print('REMAP',grid,kind,'dU',r['remap_energy_relative'],'stress',r['stress_relative'],
              'mode',r['max_mode_increment_relative'],'accept',r['dynamic_accepted'],flush=True)
    for a,b in zip((xp,Fp,old.particles.A,old.mass,old.particles.X),immutable):np.testing.assert_array_equal(a,b)
    result=dict(grid=grid,records=records,particles_unchanged=True,source_setup_seconds=source_setup_seconds,
                scope='one material-reference Q1 template change; no Eulerian production remeshing',poses=[],dynamics={})
    if 'refine_x' in accepted:
        new=accepted['refine_x']
        for angle,shift in ((0,[2.3*old.h,.2*old.h,0]),(45,[0,0,0])):
            theta=np.deg2rad(angle);Q=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
            posed=(x-.5)@Q.T+.5+shift
            px=(xp-.5)@Q.T+.5+shift;pF=Q@Fp
            _,r=audit_remap(old,new,posed,px,pF)
            r.update(angle=angle,shift=shift,old_pose_energy_relative=old.elastic(posed)[0]/old.elastic(x)[0]-1)
            result['poses'].append(r)
        initial_U=old.elastic(x)[0]
        for switch in (False,True):
            model=old;px=xp.copy();pF=Fp.copy();v=np.zeros_like(xp)
            rows=[];frames=[px.copy()];wall_start=time.perf_counter();switch_record=None
            for step in range(steps):
                guard()
                if switch and step==1:
                    before_switch_positions=px.copy();before_switch_F=pF.copy();before_switch_v=v.copy()
                    xcurrent,_=old.recover(px,pF)
                    # Rebuild maps/mass at the switch and charge that full cost.
                    switched=RemappedQ1(old,axes_for(old,'refine_x'))
                    _,switch_record=audit_remap(old,switched,xcurrent,px,pF)
                    if not switch_record['dynamic_accepted']:raise RuntimeError('nested dynamic remap rejected')
                    model=switched
                    switch_record.update(position_jump=float(np.max(np.abs(px-before_switch_positions))),
                                         F_jump=float(np.max(np.abs(pF-before_switch_F))),
                                         velocity_jump=float(np.max(np.abs(v-before_switch_v))),physical_time=step*dt)
                begin=time.perf_counter()
                px,pF,v,row=model.step(px,pF,v,dt)
                row.update(step=step+1,time=(step+1)*dt,step_seconds=time.perf_counter()-begin,
                           template='refine_x' if switch and step>=1 else 'old')
                if row['min_det']<=0 or not np.isfinite(px).all():raise RuntimeError('invalid dynamic state')
                rows.append(row);frames.append(px.copy())
            name='switched' if switch else 'control'
            result['dynamics'][name]=dict(rows=rows,switch=switch_record,wall_seconds=time.perf_counter()-wall_start,
                                           initial_energy=initial_U,final_energy=rows[-1]['mechanical'])
            np.savez_compressed(out/f'g{grid}-{name}.npz',reference=old.particles.X,positions=np.array(frames),
                                times=np.arange(steps+1)*dt,energy=np.r_[initial_U,[r['mechanical'] for r in rows]])
            with (out/f'g{grid}-{name}.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    else:result['dynamics']['status']='gated: no genuinely changed template accepted'
    identity,nested,shifted,coarse=records
    checks=dict(identity=identity['dynamic_accepted'],nested=nested['dynamic_accepted'],
                nested_modes=nested['max_mode_increment_relative']<1e-7,
                unchanged_particles=result['particles_unchanged'],
                incompatible_switch_rejected=not shifted['dynamic_accepted'] and not coarse['dynamic_accepted'],
                objective_poses=all(r['dynamic_accepted'] and abs(r['old_pose_energy_relative'])<1e-8 for r in result['poses']),
                short_dynamics='switched' in result['dynamics'])
    if 'switched' in result['dynamics']:
        d=result['dynamics']['switched']
        checks['no_history_jump']=abs(d['switch']['remap_energy_relative'])<1e-7
        checks['bounded_release']=d['final_energy']<=initial_U*1.01 and min(r['min_det'] for r in d['rows'])>.5
    result.update(checks=checks,passed=all(checks.values()),total_run_seconds_including_audits=time.perf_counter()-run_start)
    save(out/f'g{grid}.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grids',nargs='+',type=int,default=[17,33])
    p.add_argument('--steps',type=int,default=4);p.add_argument('--dt',type=float,default=.0005)
    p.add_argument('--out',type=Path,default=Path('docs/results/template-remap/v1'))
    a=p.parse_args()
    if a.steps<2 or not np.isfinite(a.dt) or a.dt<=0:p.error('at least two steps and positive finite dt required')
    storage=guard();a.out.mkdir(parents=True,exist_ok=True);results=[]
    for grid in a.grids:
        results.append(run(grid,a.out,a.steps,a.dt))
        save(a.out/'results.json',dict(storage=storage,results=results,device='cpu',production_defaults_changed=False))
    plot_results(a.out,results)
    if not all(r['passed'] for r in results):raise SystemExit('one or more remap gates failed; inspect results')


if __name__=='__main__':main()
