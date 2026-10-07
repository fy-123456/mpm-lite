"""Reuse frozen snapshots to compare paired samples and conditional moments."""
import argparse,json,time
from pathlib import Path
import numpy as np
from engine.aniso_phase1 import AnisotropicMaterialParams
from engine.aniso_phase1.material_snapshot import MaterialSnapshot,compare_snapshot
from engine.aniso_phase1.joint_sampling import METHODS,compare_joint
from benchmarks.aniso_material_snapshot import make_snapshot
from benchmarks.aniso_rotation_dissipation import guard,table
from demos.aniso import DATA_ROOT
from utils.resource_guard import inspect_storage


def representative_control(snapshot,h,params,budget):
    """Bypass F fitting to diagnose representative selection, not tune weights."""
    from engine.aniso_phase1.material_snapshot import particle_response,center_support
    from engine.aniso_phase1.joint_sampling import paired_samples
    ep,_,_=particle_response(snapshot,params);mean=snapshot.volume@snapshot.X/snapshot.volume.sum()
    second=np.einsum('p,pi->i',snapshot.volume,(snapshot.X-mean)**2);selected_second=np.zeros(3);E=0.
    for _,ids,w in center_support(snapshot,h):
        selected,W=paired_samples(snapshot,ids,w,h,budget)
        E+=float(W@ep[selected]);selected_second+=np.einsum('p,pi->i',W,(snapshot.X[selected]-mean)**2)
    return dict(budget=budget,original_F_energy_relative_error=E/float(snapshot.volume@ep)-1,
        reference_position_second_moment_relative_error=(selected_second/second-1).tolist())


def controls(out,records,params):
    rows=[]
    for r in records:
        if r['kind']!='bend' or r['grid']!=17 or r['angle'] or any(r['shift']):continue
        s=MaterialSnapshot.load(out/(r['name']+'.npz'))
        for budget in (8,16):rows.append(dict(name=r['name'],**representative_control(s,1/16,params,budget)))
    (out/'representative-controls.json').write_text(json.dumps(rows,indent=2,allow_nan=False))


def plot(out,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in records if r['kind']=='bend' and r['field']=='smooth' and not r['angle'] and not any(r['shift'])]
    methods=('moment8',)+METHODS;colors=('#888888','#287bbc','#70aee0','#d8882b','#159469','#91bd38')
    fig,axs=plt.subplots(1,3,figsize=(15,4.7),layout='constrained');width=.13
    for i,(name,color) in enumerate(zip(methods,colors)):
        values=[r['methods'][name] for r in selected]
        x=np.arange(len(selected))+(i-2.5)*width
        axs[0].bar(x,[100*v['energy_relative_error'] for v in values],width,color=color,label=name)
        axs[1].bar(x,[100*v['center_P_rms_relative_error'] for v in values],width,color=color)
        axs[2].bar(x,[v.get('material_evaluations',r['eight_point_evaluations'])/r['particles'] for r,v in zip(selected,values)],width,color=color)
    labels=[f"g{r['grid']}, {r['particles']} particles" for r in selected]
    for ax,title in zip(axs,('Signed material energy error (%)','Center mean stress RMS error (%)','Material evaluations / particle count')):
        ax.set(title=title);ax.set_xticks(np.arange(len(selected)),labels,rotation=15);ax.grid(axis='y',alpha=.2)
    axs[0].legend(fontsize=8);fig.suptitle('Frozen smooth-fiber bending: accuracy and sampling cost')
    fig.savefig(out/'summary.png',dpi=160);fig.savefig(out/'summary.svg');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('docs/results/material-snapshot'))
    p.add_argument('--out',type=Path,default=Path('docs/results/joint-sampling'));args=p.parse_args()
    guard();args.out.mkdir(parents=True,exist_ok=True);params=AnisotropicMaterialParams(10,20,200)
    (args.out/'environment.json').write_text(json.dumps(dict(device='cpu',production_solver_modified=False,
        storage=vars(inspect_storage(DATA_ROOT))),indent=2))
    source=json.loads((args.source/'snapshots.json').read_text());records=[];flat=[]
    cases=[(r,MaterialSnapshot.load(args.source/(r['name']+'.npz'))) for r in source]
    # High-PPC/finer-grid smooth fields prevent apparent success from simply
    # retaining every particle in a low-population center. Only five new states.
    for grid,ppc,angle,shift in ((17,4,0.,(0.,0.,0.)),(17,4,45.,(0.,0.,0.)),
                                (33,2,0.,(0.,0.,0.)),(33,2,45.,(0.,0.,0.)),(17,2,0.,(.037,.013,0.))):
        s=make_snapshot(grid,ppc,'bend','smooth',angle,shift);r,_=compare_snapshot(s,1/(grid-1),params)
        r.update(name=f'bend-smooth-g{grid}-ppc{ppc}-r{angle:g}'+('-shift' if any(shift) else ''),
                 grid=grid,ppc_axis=ppc,kind='bend',field='smooth',angle=angle,shift=list(shift))
        cases.append((r,s))
    for original,s in cases:
        guard();before=[a.copy() for a in (s.x,s.X,s.F,s.A,s.volume)];start=time.perf_counter()
        result,details=compare_joint(s,1/(original['grid']-1),params)
        for a,b in zip(before,(s.x,s.X,s.F,s.A,s.volume)):
            if not np.array_equal(a,b):raise AssertionError('snapshot mutated')
        r=dict(original);r['methods']=dict(original['methods']);r['methods'].update(result['methods'])
        r.update(joint_diagnostic_seconds=time.perf_counter()-start,max_rule_volume_error=result['max_rule_volume_error'],
                 min_sample_weight=result['min_sample_weight'],particle_support_entries=result['particle_support_entries'],snapshot_unchanged=True)
        s.save(args.out/(r['name']+'.npz'));table(args.out/(r['name']+'-joint-centers.csv'),details);records.append(r)
        for method in METHODS:
            m=r['methods'][method];flat.append(dict(case=r['name'],method=method,**{k:v for k,v in m.items() if not isinstance(v,list)}))
        (args.out/'snapshots.json').write_text(json.dumps(records,indent=2,allow_nan=False));table(args.out/'summary.csv',flat)
        print(r['name'],{name:[round(100*r['methods'][name]['energy_relative_error'],4),round(100*r['methods'][name]['center_P_rms_relative_error'],4),r['methods'][name]['material_evaluations']] for name in METHODS},flush=True)
    controls(args.out,records,params);plot(args.out,records)


if __name__=='__main__':main()
