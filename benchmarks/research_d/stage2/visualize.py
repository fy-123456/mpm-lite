"""Export raw static field samples and a labelled scientific comparison plot."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from engine.aniso_phase1.research_d.frozen_inputs import load_frozen_inputs
from engine.aniso_phase1.consistent_transfer import material_response
from benchmarks.research_d.stage2.bootstrap import PARENT,PIN
from benchmarks.research_d.stage2.validate import OUT

def main():
    root=Path.cwd();s,_,_,_=load_frozen_inputs(root/PARENT,root,expected_sha256=PIN)
    with np.load(OUT/'gpu-equilibrium.npz') as z:q=z['q']
    axes=[np.linspace(s.edges[0][0],s.edges[0][-1],101),np.linspace(s.edges[1][0],s.edges[1][-1],31),np.array([np.mean(s.edges[2][[0,-1]])])]
    x,F=s.evaluate(q,axes);X=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1)
    _,P=material_response(F.reshape(-1,3,3),np.broadcast_to(s.A,(F.size//9,3,3)),s.params)
    P=P.reshape(F.shape);np.savez(OUT/'static-midplane.npz',X=X,x=x,F=F,PK1=P)
    report=json.loads((OUT/'static-solves.json').read_text())
    fig,ax=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for name,r in report.items():ax[0].semilogy([v['iteration'] for v in r['trace']],[v['residual_N'] for v in r['trace']],'-o',label=name.upper(),alpha=.8)
    ax[0].axhline(1e-8,color='grey',linestyle='--',label='tight threshold');ax[0].set(xlabel='Accepted update',ylabel='Original free-gradient norm (N)',title='Same 0.005 m grip displacement');ax[0].legend();ax[0].grid(alpha=.3)
    shown=X+10*(x-X);field=ax[1].scatter(shown[:,:,0,0].ravel(),shown[:,:,0,1].ravel(),c=P[:,:,0,0,0].ravel(),s=5,marker='s',cmap='viridis',linewidths=0)
    fig.colorbar(field,ax=ax[1],label='PK1 xx (Pa)');ax[1].set(xlabel='x (m)',ylabel='y (m)',title='Static midplane; displacement shown at 10x');ax[1].set_aspect('equal')
    fig.savefig(OUT/'static-validation.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
