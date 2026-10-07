"""v11 comparisons against locally refined reference and delivered plots."""
import json,itertools
from pathlib import Path
import numpy as np
from engine.aniso_phase1 import local_reference as ref
from benchmarks.aniso_residual_gate import geometry
from benchmarks.aniso_static_space import center_interpolation
from benchmarks.aniso_boundary_reference import hessian
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'docs/results/lite-aniso-mainline';OUT=BASE/'v11'


def local_comparison(degree=2):
    reference=BASE/('v11-reference/cases/F45-local2-q2' if degree==2 else 'v11-reference-q3/cases/local2')
    if degree==3:
        from benchmarks.aniso_local_q3 import gradient as evaluate
    else:evaluate=ref.gradient
    with np.load(reference.with_suffix('.npz')) as z:edges=[z[f'axis{k}'].copy() for k in range(3)];ur=z['u'].copy()
    rr=json.loads(reference.with_suffix('.json').read_text());rows=[];H=hessian('F45')
    # Include EVERY interpolation knot (center locations) in integration cells.
    for grid in (9,17,33):
        g=geometry(grid);h=1/(grid-1)
        common=[np.union1d(e,(np.arange(-1,grid+1)+.5)*h) for e in edges]
        common=[e[(e>=ref.LO[k])&(e<=ref.HI[k])] for k,e in enumerate(common)]
        inputs=[]
        for version,mode in [('v10','residual_corotated'),('v10','residual_center'),('v11','matrix_only'),('v11','selective_patch')]:
            f=BASE/version/f'static/F45-g{grid}-{mode}'
            with np.load(f.with_suffix('.npz')) as z:u=z['u'].copy()
            r=json.loads(f.with_suffix('.json').read_text());Lc=np.stack([D@u for D in g['D']],axis=2).reshape(-1,9)
            inputs.append((version,mode,Lc,r))
        totals=np.zeros((len(inputs),4,4));keys=['whole','near_grip','interior','deep_interior']
        for X,V in ref.chunks(common,order=degree+1):
            Lr=evaluate(X,edges,degree,ur);Pr=(Lr.reshape(-1,9)@H.T).reshape(-1,3,3);I=center_interpolation(X,g['centers'],h)
            masks=[np.ones(len(X),bool),np.minimum(abs(X[:,0]-.25),abs(X[:,0]-.75))<.0625,(X[:,0]>.3125)&(X[:,0]<.6875),(X[:,0]>.375)&(X[:,0]<.625)]
            norm=lambda q:np.sum(q*q,axis=(1,2))
            for j,(_,_,Lc,_) in enumerate(inputs):
                L=(I@Lc).reshape(-1,3,3);P=(L.reshape(-1,9)@H.T).reshape(-1,3,3);vals=np.column_stack((norm(L-Lr),norm(Lr),norm(P-Pr),norm(Pr)))*V[:,None]
                for k,mask in enumerate(masks):totals[j,k]+=vals[mask].sum(axis=0)
        for j,(version,mode,_,r) in enumerate(inputs):
            regions={k:dict(F_relative=float(np.sqrt(t[0]/t[1])),P_relative=float(np.sqrt(t[2]/t[3]))) for k,t in zip(keys,totals[j])}
            rows.append(dict(grid=grid,version=version,mode=mode,reaction_N=r['reaction_N'],reference_reaction_N=rr['reaction_N'],reaction_signed_relative=(r['reaction_N']-rr['reaction_N'])/rr['reaction_N'],regions=regions))
        print('local comparison grid',grid,flush=True)
    (OUT/('local-reference-comparison.json' if degree==2 else 'local-q3-reference-comparison.json')).write_text(json.dumps(dict(records=rows,reference=f'v11 local2 Q{degree}; uncertainty in reference remains; common integration resolves reference and Lite gradient knots'),indent=2)+'\n')


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=json.loads((OUT/'static-summary.json').read_text());old=json.loads((BASE/'v10/static-summary.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,label in zip(axes.flat,('ISO','F0','F45','F90')):
        for data,mode in [(old,'residual_corotated'),(s,'matrix_only'),(s,'selective_patch')]:
            rows=sorted([r for r in data['records'] if r['case']==label and r['mode']==mode],key=lambda r:r['grid'])
            ax.plot([1/(r['grid']-1) for r in rows],[r['reaction_N'] for r in rows],'-o',label=mode)
        ax.axhline(rows[-1]['reaction_reference_N'],ls='--',color='k',label='v10 finite reference')
        ax.set(title=label,xlabel='h (m)',ylabel='static reaction (N)');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(OUT/'static-reactions.png',dpi=150);plt.close(fig)
    d=json.loads((OUT/'summary.json').read_text());fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,key in zip(axes,('reaction_relative','P_relative')):
        for r in d['refinement']:ax.plot([.0005,.00025,.000125],[100*x[key] for x in r['pairs']],'-o',label=r['case'])
        ax.axhline(2,ls=':',color='k');ax.set_xscale('log');ax.set(title=key,xlabel='finer dt (s)',ylabel='adjacent difference (%)');ax.grid(alpha=.25);ax.legend()
    fig.savefig(OUT/'slow-time-differences.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    import sys
    local_comparison(3 if sys.argv[1]=='local-q3' else 2) if len(sys.argv)>1 and sys.argv[1] in ('local','local-q3') else plots()
