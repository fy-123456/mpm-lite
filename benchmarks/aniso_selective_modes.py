"""Classify normal polynomial fields and true free-space material null modes.

Deleting exterior-centered terms is a diagnostic only (it changes total weight),
never a second tuned production option. No baseline artifacts are changed.
"""
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from benchmarks.aniso_residual_gate import geometry,matrix,hourglass_matrix,rank_gate,static_solve
from benchmarks.aniso_boundary_reference import hessian,CASES
from engine.aniso_phase1.selective_patch import scalar_matrix
from engine.aniso_phase1.boundary_reference import LO,HI
OUT=Path(__file__).resolve().parents[1]/'docs/results/lite-aniso-mainline/v11'


def main():
    rows=[];sensitivity=[]
    for label in CASES:
        g=geometry(9);h=.125;Km=matrix(g,hessian(label),'residual_center');S,ids,P=scalar_matrix(g['nodes'],g['centers'],g['volume'],h);Ks=sp.block_diag([S]*3,format='csr');Kh=hourglass_matrix(g,hessian(label))
        A=Km[g['free']][:,g['free']].toarray();vals,Z=np.linalg.eigh(A);scale=np.max(np.abs(A).sum(axis=1));Z=Z[:,vals<=1e-9*max(scale,1.)]
        projected=Z.T@(Ks[g['free']][:,g['free']]@Z);lift=np.linalg.eigvalsh(projected)
        x=g['nodes']-.5;fields={'stretch':np.column_stack((x[:,0],np.zeros((len(x),2)))),'shear':np.column_stack((x[:,1],np.zeros((len(x),2)))),'bend_xy':np.column_stack((-x[:,0]*x[:,1],.5*x[:,0]**2,np.zeros(len(x)))),'bend_xz':np.column_stack((-x[:,0]*x[:,2],np.zeros(len(x)),.5*x[:,0]**2))}
        patches={name:dict(v10_stabilizer_J=float(.5*u.T.ravel()@(Kh@u.T.ravel())),selective_stabilizer_J=float(.5*u.T.ravel()@(Ks@u.T.ravel()))) for name,u in fields.items()}
        rows.append(dict(case=label,grid=9,material_zero_or_soft_count=Z.shape[1],selective_energy_min_on_material_null=float(lift[0]),selective_energy_max_on_material_null=float(lift[-1]),fields=patches))
    for grid in (9,17,33):
        g=geometry(grid);h=1/(grid-1);xc=(g['centers']+.5)*h;inside=np.all((xc>=LO)&(xc<=HI),axis=1);weights=g['volume']*inside
        S,_,_=scalar_matrix(g['nodes'],g['centers'],weights,h);K=matrix(g,hessian('F45'),'residual_center')+sp.block_diag([S]*3,format='csr');gate=rank_gate(g,K);_,r=static_solve(g,K,gate['passed'])
        r.update(grid=grid,gate=gate,retained_weight_fraction=float(weights.sum()/g['volume'].sum()),label='delete exterior-center contributions without renormalization; diagnostic, not production')
        sensitivity.append(r)
    (OUT/'mode-boundary-diagnostic.json').write_text(json.dumps(dict(mode_checks=rows,boundary_sensitivity=sensitivity,adopted=False),indent=2)+'\n')
    print('mode and exterior support diagnostics complete',flush=True)


if __name__=='__main__':main()
