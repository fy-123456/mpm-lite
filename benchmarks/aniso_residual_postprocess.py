"""Locate candidate stabilization energy and plot already completed static data."""
import json
from pathlib import Path
import numpy as np
from benchmarks.aniso_residual_gate import geometry, matrix, CORNERS
from benchmarks.aniso_boundary_reference import CASES, hessian
from engine.aniso_phase1.beam_reference import element_stiffness

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/results/lite-aniso-mainline/v10'


def main():
    summary=json.loads((OUT/'static-summary.json').read_text());records=[]
    for r in summary['records']:
        if r['mode']!='residual_corotated':continue
        grid,label=r['grid'],r['case'];g=geometry(grid);h=1/(grid-1);H=hessian(label)
        with np.load(OUT/f'static/{label}-g{grid}-residual_corotated.npz') as z:u=z['u'].copy()
        K=matrix(g,H,'residual_center');material=.5*float(u.T.ravel()@(K@u.T.ravel()))
        local=element_stiffness(h,H,True)-element_stiffness(h,H,False)
        lookup={tuple(np.rint(x/h).astype(int)):i for i,x in enumerate(g['nodes'])}
        ids=np.array([[lookup[tuple(c+o)] for o in CORNERS] for c in g['centers']])
        uloc=u[ids].reshape(len(ids),24)
        values=.5*g['volume']/h**3*np.einsum('ci,ij,cj->c',uloc,local,uloc)
        hg=float(values.sum());total=material+hg
        assert abs(total-r['energy_J'])<1e-10*max(total,1e-20)
        x=(g['centers'][:,0]+.5)*h;near=np.minimum(abs(x-.25),abs(x-.75))<=.0625+1e-12
        records.append(dict(case=label,grid=grid,material_J=material,stabilization_J=hg,
            stabilization_fraction=hg/total,stabilization_near_grip_fraction=float(values[near].sum()/hg),
            total_energy_J=total))
    (OUT/'stabilization-localization.json').write_text(json.dumps(dict(records=records,
        region='center locations within 1/16 m of either grip interface; center-volume weighted diagnostic'),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,label in zip(axes.flat,CASES):
        for mode in ('projected_center','residual_center','residual_corotated'):
            selected=sorted([r for r in summary['records'] if r['case']==label and r['mode']==mode],key=lambda r:r['grid'])
            ax.plot([1/(r['grid']-1) for r in selected],[r['reaction_N'] for r in selected],'-o',label=mode)
        ax.axhline(selected[-1]['reaction_reference_N'],color='k',linestyle='--',label='finite reference (uncertainty remains)')
        ax.set(title=label,xlabel='grid spacing (m)',ylabel='static reaction (N)');ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.savefig(OUT/'static-reactions.png',dpi=150);plt.close(fig)
    root=ROOT/'docs/results/lite-aniso-mainline'
    ref=json.loads((root/'v10-reference/summary.json').read_text())['pairs']
    ref+=json.loads((root/'v10-reference-fine/summary.json').read_text())['pairs']
    q2=json.loads((root/'v10-reference-q2/summary.json').read_text())['pairs']
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for boundary in ('hard','smooth'):
        data=sorted([r for r in ref if r['case']=='F45' and r['boundary']==boundary],key=lambda r:r['grids'][-1])
        for ax,key in zip(axes,('reaction','whole','interior')):
            y=[r['reaction_relative'] if key=='reaction' else r['regions'][key]['P_relative'] for r in data]
            ax.loglog([r['grids'][-1] for r in data],100*np.array(y),'-o',label='Q1 '+boundary)
    for ax,key in zip(axes,('reaction','whole','interior')):
        y=[r['reaction_relative'] if key=='reaction' else r['regions'][key]['P_relative'] for r in q2]
        ax.loglog([2*r['cells_per_unit'][-1]+1 for r in q2],100*np.array(y),'-s',label='Q2 hard')
        ax.set(title='F45 '+key,xlabel='equivalent nodes per unit axis',ylabel='adjacent relative difference (%)')
        ax.axhline(1 if key=='reaction' else 2,color='k',linestyle=':');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(OUT/'reference-regions.png',dpi=150);plt.close(fig)
    print('wrote localization for',len(records),'static solutions and two plots')


if __name__=='__main__':main()
