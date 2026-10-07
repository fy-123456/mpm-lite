"""Regenerate compact plots and machine-readable audit from saved experiments."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmarks.aniso_quadrature_validation import plot,save


def main():
    base=Path('docs/results/quadrature-validation');results=[];seen=set()
    for name in ('aligned-v1','aligned-v1-fine','adaptive-v1','particle-refinement-v1'):
        for r in json.loads((base/name/'results.json').read_text()):
            r['records']=[a for a in r['records'] if a['name'] not in seen]
            seen.update(a['name'] for a in r['records']);results.append(r)
    plot(base,results)
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    coarse=json.loads((base/'aligned-v1/grid17.json').read_text());lookup={r['rule']:r for r in coarse['records']}
    for column,rule in enumerate(('gauss3','gauss2','group4x8')):
        root=base/'aligned-v1'/f'g17-{rule}'
        with np.load(str(root)+'.npz') as a:
            X=a['reference'];mask=X[:,2]==X[:,2].min()
            for key,color,label in [('displacement_mls','tab:blue','candidate'),('dense_displacement','tab:green','dense MLS'),('reference_displacement','tab:orange','Q1')]:
                points=X+50*a[key];axes[0,column].scatter(points[mask,0],points[mask,1],s=4,color=color,label=label)
        with np.load(str(root)+'-mode.npz') as a:
            X=a['reference'];u=a['displacement'];points=X+.02*u/max(np.linalg.norm(u,axis=1).max(),1e-20)
            axes[1,column].scatter(X[mask,0],X[mask,1],s=3,color='gray',label='rest')
            axes[1,column].scatter(points[mask,0],points[mask,1],s=6,color='tab:red',label='weakest mode')
        axes[0,column].set_title(f'{rule}: load shape x50')
        axes[1,column].set_title(f"Worst stiffness ratio {lookup[rule]['rho_min']:.4g}")
        for row in range(2):axes[row,column].set_aspect('equal');axes[row,column].legend(fontsize=7);axes[row,column].grid(alpha=.2)
    fig.tight_layout();fig.savefig(base/'shapes-and-modes.png',dpi=160);plt.close(fig)
    d=json.loads((base/'dynamics-v2/dynamics.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,4))
    for r in d:
        a=r['rows'];t=[v['time'] for v in a]
        if r['scene']=='release':axes[0].plot(t,[v['mechanical']/r['initial_energy'] for v in a],label=r['name'])
        else:axes[1].plot(t,[v['tip_displacement'] for v in a],label=r['name'])
    axes[0].set(xlabel='Physical time',ylabel='Mechanical energy / initial energy');axes[1].set(xlabel='Physical time',ylabel='Load-conjugate tip displacement')
    for ax in axes:ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(base/'short-dynamics.png',dpi=160);plt.close(fig)
    transfer=[]
    for folder in ('transfer-v1','transfer-fine-v1'):
        source=base/folder/'results.json'
        if source.exists():transfer+=json.loads(source.read_text())
    if transfer:
        labels=[f"g{r['grid']} PPC {r['ppc_axis']}^3" for r in transfer]
        rows=[next(a for a in r['records'] if a['field']=='solved_mls' and a['dt']==.001) for r in transfer]
        index=np.arange(len(rows));fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].bar(index-.2,[100*r['gradient_relative_mismatch'] for r in rows],.4,label='Gradient mismatch')
        axes[0].bar(index+.2,[100*r['actual_history_relative'] for r in rows],.4,label='Rebuilt energy change')
        axes[0].set_ylabel('Relative difference (%)');axes[0].axhline(0,color='gray',lw=.8);axes[0].legend()
        axes[1].bar(index,[100*r['mass']['negative_mass_fraction'] for r in transfer])
        axes[1].set_ylabel('Negative lumped mass magnitude / total (%)')
        for ax in axes:ax.set_xticks(index,labels);ax.grid(axis='y',alpha=.2)
        fig.suptitle('Direct MLS material + Lite transfer fails the compatibility gate')
        fig.tight_layout();fig.savefig(base/'transfer-gate.png',dpi=160);plt.close(fig)
    audit=dict(reference_max_matrix_error=max(r['convergence'].get('matrix_relative',0) for r in results),
        reference_max_clamp_density_error=max(r['convergence']['clamp_density_solution_relative'] for r in results),
        tests=json.loads((base/'tests.json').read_text()),viewer=json.loads((base/'viser-smoke.json').read_text()),
        production_changed=False,scope='frozen physical-domain quadrature diagnostics',
        production_gate='failed direct replacement: transfer/history mismatch and unsafe lumped mass',
        conditional_stop='full MPM performance not run because correctness gate failed; not a performance pass')
    save(base/'audit.json',audit)


if __name__=='__main__':main()
