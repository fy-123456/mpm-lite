"""Summarize archived reconstruction comparisons without rerunning simulation."""
import json
from pathlib import Path
import numpy as np


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path('docs/results/quadratic');old=Path('docs/results/corotated')
    rotations=json.loads((root/'rotation.json').read_text());statics=json.loads((root/'static.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    colors={'corotated':'#9855aa','quadratic':'#287bbc','material_quadratic':'#16936b'}
    for grid,ax in zip((17,33),axes[0]):
        for mode in colors:
            folder=old if mode=='corotated' else root;name=f'rotation-g{grid}-dt0.01-{mode}'
            data=np.genfromtxt(folder/(name+'.csv'),delimiter=',',names=True)
            budget=np.genfromtxt(folder/(name+'-budget.csv'),delimiter=',',names=True)
            ax.plot(data['stabilization_sampling_angle_degrees'],100*((data['center_material_energy']+data['stabilization_energy'])/budget['elastic'][0]-1),
                label=mode,color=colors[mode],marker='o',markersize=3)
        ax.set(title=f'Bent beam rotation about z, grid {grid}',xlabel='Sampling orientation (degrees)',ylabel='Elastic energy change (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax=axes[1,0]
    for mode in ('quadratic','material_quadratic'):
        d=np.genfromtxt(root/f'rotation-g17-dt0.01-{mode}.csv',delimiter=',',names=True)
        ax.plot(d['stabilization_sampling_angle_degrees'],100*(d['stabilization_energy']/d['stabilization_energy'][0]-1),label=mode,color=colors[mode],marker='o')
    ax.set(title='Stabilization alone, grid 17',xlabel='Sampling orientation (degrees)',ylabel='Stabilization energy change (%)');ax.legend(fontsize=8);ax.grid(alpha=.2)
    ax=axes[1,1];ax.bar([str(r['grid']) for r in statics],[100*r['relative_to_full'] for r in statics],color='#287bbc')
    ax.set(title='Static beam, no mass term; zero soft modes',xlabel='Grid',ylabel='Tip displacement error vs full Q1 (%)')
    fig.savefig(root/'summary.png',dpi=170);fig.savefig(root/'summary.svg');plt.close(fig)
    groups=dict(solve=['solve_delta'],velocity_transfer=['p2c_delta','c2g_delta','g2p_delta'],
        history=['state_reset_delta','state_transport_delta','volume_remap_delta'],boundary=['boundary_projection_delta','final_projection_damping_delta'])
    release=[]
    for folder,name in ((old,'beam-corotated-dt0.0005'),(root,'beam-quadratic'),(root,'beam-material_quadratic')):
        a=np.genfromtxt(folder/(name+'.csv'),delimiter=',',names=True);E=a['mechanical'][0]
        release.append(dict(name=name,initial_energy=E,relative_change=float(a['mechanical'][-1]/E-1),
            budget_relative={k:float(sum(a[t][1:].sum() for t in terms)/E) for k,terms in groups.items()},
            stabilization_rebuild_relative=float(a['stabilization_rebuild_delta'][1:].sum()/E)))
    (root/'comparison.json').write_text(json.dumps(dict(release=release,
        preferred_experimental_mode='quadratic',reason='Comparable rotation/translation errors; rotating template adds complexity without a material accuracy gain in these cases.'),indent=2))


if __name__=='__main__':main()
