"""Standalone plots of the variational implementation's archived evidence."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('docs/results/variational'))
    args=p.parse_args();root=args.data
    trajectories=json.loads((root/'trajectories.json').read_text())
    operators=json.loads((root/'operator.json').read_text())
    nonlinear=json.loads((root/'nonlinear.json').read_text())
    def csvdata(r):
        with (root/(r['name']+'.csv')).open() as f:return list(csv.DictReader(f))
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for r in trajectories:
        if r['scene']=='fixed' and r['dt']==.001 and r['flip']==.9 and r['cg_tol']==1e-4:
            rows=csvdata(r);t=[float(x['time']) for x in rows];e=[100*(float(x['mechanical'])/r['initial_energy']-1) for x in rows]
            axes[0,0].plot(t,e,'-' if r['formulation']=='variational' else '--',label=r['formulation'])
        if r['scene']=='tensile':
            rows=csvdata(r);d=[1000*float(x['displacement']) for x in rows];f=[float(x['right_force']) for x in rows]
            idx=int(r['angle']/45);new=r['formulation']=='variational'
            axes[0,1].plot(d,f,linestyle='-' if new else '--',color=f'C{idx}',alpha=1 if new else .5,label=f'{r["angle"]:g} deg' if new else None)
    axes[0,0].set(title='Fixed block: mechanical energy (curves nearly overlap)',xlabel='Time [s]',ylabel='Change / initial energy [%]')
    axes[0,0].legend()
    axes[0,1].set(title='Loading / unloading: solid new, dashed legacy',xlabel='Commanded displacement [mm]',ylabel='Loading-grip force [N]')
    axes[0,1].legend()
    for i,kf in enumerate((0.,200.,20000.)):
        rows=sorted([r for r in operators if r['quadrature']=='center' and r['state']=='compressed' and r['kf']==kf],key=lambda r:r['dt'])
        for mode in ('exact','modified'):
            axes[1,0].plot([r['dt'] for r in rows],[r[mode]['mass_scaled_min'] for r in rows],marker='o',linestyle='-' if mode=='exact' else '--',color=f'C{i}',label=f'kf={kf:g}' if mode=='exact' else None)
    axes[1,0].axhline(0,color='gray',linewidth=.7)
    axes[1,0].set(xscale='log',yscale='symlog',ylim=(-5e5,3),title='Minimum mass-scaled eigenvalue: exact / modified',xlabel='Time step [s]',ylabel='Minimum eigenvalue (modified >= 1)')
    axes[1,0].legend()
    for i,r in enumerate(nonlinear):
        trace=r['trace'];energies=[trace[0]['potential_before']]+[x['potential_after'] for x in trace]
        axes[1,1].semilogy(range(len(energies)),energies,label=f'kf={r["kf"]:g}, dt={r["dt"]:g}',color=f'C{i}')
        indices=[j+1 for j,x in enumerate(trace) if x['projected_tangent']]
        axes[1,1].scatter(indices,[energies[j] for j in indices],color=f'C{i}',marker='x')
    axes[1,1].set(title='Original potential: crosses mark modified directions',xlabel='Accepted Newton update',ylabel='Incremental potential')
    axes[1,1].legend()
    for ax in axes.flat:ax.grid(True,alpha=.2)
    fig.savefig(root/'overview.png',dpi=180)
    fig.savefig(root/'overview.svg')


if __name__=='__main__':main()
