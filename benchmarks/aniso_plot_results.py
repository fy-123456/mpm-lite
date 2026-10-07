"""Render shareable quantitative figures from recorded experiment CSV/JSON."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_csv(path):
    with path.open() as f:
        rows=list(csv.DictReader(f))
    return {k:np.array([float(r[k]) if r[k] else 0. for r in rows]) for k in rows[0] if k not in ('linear_solver',)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('docs/results/quantitative'))
    args=p.parse_args();root=args.root
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for flip in (0.,.9,1.):
        r=read_csv(root/f'energy-fixed-dt0.001-flip{flip}-cg0.0001-vt1e-10.csv')
        axes[0,0].plot(r['time'],100*r['cumulative_delta']/r['mechanical'][0],label=f'FLIP={flip:g}')
    axes[0,0].set(xlabel='Physical time (s)',ylabel='Mechanical energy change (%)',title='Fixed block: equal physical time')
    axes[0,0].legend()
    summary=json.loads((root/'energy-summary.json').read_text())
    row=next(r for r in summary if r['scene']=='fixed' and r['dt']==.001 and r['flip']==.9 and r['cg_tol']==1e-4)
    terms=['p2c_delta','c2g_delta','boundary_projection_delta','solve_delta','g2p_delta','volume_remap_delta']
    values=[100*row[k]/row['initial_energy'] for k in terms]
    axes[0,1].bar(['P2C','C2G','Boundary','Solve','G2P','Volume'],values,color=['#cc6677' if v<0 else '#4477aa' for v in values])
    axes[0,1].axhline(0,color='k',lw=.5)
    axes[0,1].set(ylabel='Signed change / initial energy (%)',title='Fixed block budget: FLIP=.9, dt=.001')
    for angle in (0,45,90):
        r=read_csv(root/f'tensile-angle{angle}.csv')
        axes[1,0].plot(r['displacement'],r['right_force'],label=f'{angle} deg')
    axes[1,0].set(xlabel='Commanded grip displacement',ylabel='Actuator force (+x)',title='Dynamic loading / unloading')
    axes[1,0].legend()
    axes[1,0].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    axes[1,0].locator_params(axis='x',nbins=5)
    fair=root/'fair-aggregate.json'
    if fair.exists():
        summary=json.loads(fair.read_text())
        for method in ('center','particle'):
            rows=sorted([r for r in summary if r['method']==method and r['grid']==17],key=lambda r:r['ppc'])
            axes[1,1].plot([r['ppc'] for r in rows],[r['warm_step_mean_seconds']*1e3 for r in rows],'-o',label=method)
        axes[1,1].set(xlabel='Particles per physical grid cell',ylabel='Median warm step (ms)',title='Same physics, grid 17³, three repeats',xscale='log')
        axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.savefig(root/'overview.png',dpi=180)
    fig.savefig(root/'overview.svg')
    plt.close(fig)


if __name__=='__main__':main()
