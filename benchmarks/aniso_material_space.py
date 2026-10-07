"""Independent massless spatial/density/integration audit of v14 initial tangent.

Material carriers coincide with grid nodes initially, hence N=I. Assemble this
matrix explicitly and compare F45 against the archived locally refined Q3 FEM.
This is small-strain spatial evidence, not a finite-motion spatial certificate.
"""
import hashlib,itertools,time
import numpy as np
import scipy.sparse as sp
from benchmarks.aniso_material_history import BASE,ROOT,load,write
from benchmarks.aniso_static_space import rule,center_interpolation
from benchmarks.aniso_residual_gate import CORNERS,matrix,rank_gate,static_solve
from benchmarks.aniso_boundary_reference import hessian
from benchmarks.aniso_local_q3 import gradient
from engine.aniso_phase1 import local_reference as ref
from engine.aniso_phase1.projected_history import frozen_maps
from engine.aniso_phase1.selective_patch import scalar_matrix
from engine.aniso_phase1.material_patch import carrier_map
OUT=BASE/'v14-space';QUADS=[('midpoint',2),('midpoint',4),('gauss',2),('gauss',4)]

def geometry(grid,order,gauss):
    h=1/(grid-1);x,Vp=rule(grid,order,gauss);base=np.floor(x/h-.5).astype(int)
    centers=np.unique((base[:,None,:]+CORNERS).reshape(-1,3),axis=0);nodes=np.unique((centers[:,None,:]+CORNERS).reshape(-1,3),axis=0)
    maps,F,V,W,S,D=frozen_maps(x,Vp,np.tile(np.eye(3),(len(x),1,1)),centers,nodes,h)
    fixed=(nodes[:,0]*h<=.25+1e-12)|(nodes[:,0]*h>=.75-1e-12)
    return dict(grid=grid,nodes=nodes*h,centers=centers,points=x,particle_volume=Vp,volume=V,D=D,maps={'particle':tuple(S@d for d in D)},fixed=fixed,free=np.flatnonzero(np.tile(~fixed,3)),beam=False)

def main():
    OUT.mkdir(exist_ok=False);reference=BASE/'v11-reference-q3/cases/local2';paths=[reference.with_suffix('.npz'),reference.with_suffix('.json'),ROOT/'benchmarks/aniso_material_space.py']
    write(OUT/'protocol.json',dict(grids=[9,17,33],quadrature=QUADS,case='F45',physical_geometry_fixed=True,linearized_at='F=I,Y=X',mass_included=False,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},reference_certified=False))
    with np.load(reference.with_suffix('.npz')) as z:edges=[z[f'axis{k}'].copy() for k in range(3)];ur=z['u'].copy()
    rr=load(reference.with_suffix('.json'));H=hessian('F45');records=[]
    for grid in (9,17,33):
        h=1/(grid-1);inputs=[]
        for quad,order in QUADS:
            start=time.monotonic();g=geometry(grid,order,quad=='gauss');S,ids,P=scalar_matrix(g['nodes'],g['centers'],g['volume'],h)
            _,_,N=carrier_map(g['nodes'],np.rint(g['nodes']/h).astype(int),h);err=float(abs(N-sp.eye(N.shape[0])).max());assert err<1e-14
            Ks=sp.block_diag([N.T@S@N]*3,format='csr');K=matrix(g,H,'residual_center')+Ks;gate=rank_gate(g,K);assert gate['passed']
            u,r=static_solve(g,K,True);assert r['solved'];r.update(grid=grid,quadrature=quad,order=order,particles=len(g['points']),gate=gate,initial_carrier_identity_max=err,reference_reaction_N=rr['reaction_N'],reaction_signed_relative=(r['reaction_N']-rr['reaction_N'])/rr['reaction_N'],seconds=time.monotonic()-start)
            key=f'F45-g{grid}-{quad}{order}';np.savez_compressed(OUT/(key+'.npz'),u=u,nodes=g['nodes']);Lc=np.stack([D@u for D in g['D']],axis=2).reshape(-1,9);inputs.append((key,g,Lc,r));print(key,'rank',gate['passed'],'R',r['reaction_signed_relative'],flush=True)
        common=[np.union1d(e,(np.arange(-1,grid+1)+.5)*h) for e in edges];common=[e[(e>=ref.LO[k])&(e<=ref.HI[k])] for k,e in enumerate(common)]
        totals=np.zeros((len(inputs),4,4));keys=['whole','near_grip','interior','deep_interior']
        for X,V in ref.chunks(common,order=4):
            Lr=gradient(X,edges,3,ur);Pr=(Lr.reshape(-1,9)@H.T).reshape(-1,3,3);I=center_interpolation(X,inputs[0][1]['centers'],h)
            masks=[np.ones(len(X),bool),np.minimum(abs(X[:,0]-.25),abs(X[:,0]-.75))<.0625,(X[:,0]>.3125)&(X[:,0]<.6875),(X[:,0]>.375)&(X[:,0]<.625)]
            norm=lambda a:np.sum(a*a,axis=(1,2))
            for j,(_,g,Lc,_) in enumerate(inputs):
                assert np.array_equal(g['centers'],inputs[0][1]['centers']);L=(I@Lc).reshape(-1,3,3);Pfield=(L.reshape(-1,9)@H.T).reshape(-1,3,3);vals=np.column_stack((norm(L-Lr),norm(Lr),norm(Pfield-Pr),norm(Pr)))*V[:,None]
                for k,m in enumerate(masks):totals[j,k]+=vals[m].sum(axis=0)
        for j,(key,g,Lc,r) in enumerate(inputs):
            r['regions']={k:dict(F_relative=float(np.sqrt(t[0]/t[1])),P_relative=float(np.sqrt(t[2]/t[3]))) for k,t in zip(keys,totals[j])};write(OUT/(key+'.json'),r);records.append(r)
        print('integrated grid',grid,flush=True)
    write(OUT/'summary.json',dict(completed=True,records=records,all_massless_ranks_passed=all(r['gate']['passed'] for r in records),all_solves_passed=all(r['solved'] for r in records),scope='initial material-carried tangent; N=I independently verified; common integration resolves all interpolation knots; no time integration',reference='archived v11 local2 Q3; reference itself not fully certified',space_accuracy_certified=False))

if __name__=='__main__':main()
