"""Time-independent spatial audit of the actual Lite small-strain operators.

The geometry, material Hessian, displacement and grid-grip coordinates are
fixed. Center, projected-center and particle integration share Lite gradients.
An independently assembled, fully integrated Q1 FEM on the physical body is
the reference. Singular reduced-integration systems are reported explicitly.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import itertools
import json
from pathlib import Path
import time
import unittest

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import minres,spsolve

from engine.aniso_phase1.projected_history import frozen_maps
from engine.aniso_phase1.beam_reference import reference_hessian,beam_matrices,element_stiffness,shape_gradients
from benchmarks.aniso_mainline import write_json

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'docs/results/lite-aniso-mainline/v9-space'
CORNERS=np.array(list(itertools.product((0,1),repeat=3)))
LO=np.array([.125,.375,.375]);HI=np.array([.875,.625,.625])
CASES={'ISO':(0.,0.),'F0':(200.,0.),'F45':(200.,45.),'F90':(200.,90.)}
MODES=('center','projected_center','particle')


def source_hashes():
    files=[Path(__file__),ROOT/'engine/aniso_phase1/projected_history.py',ROOT/'engine/aniso_phase1/beam_reference.py',
           ROOT/'engine/aniso_phase1/particle_quadrature.py',ROOT/'engine/aniso_phase1/solver.py',ROOT/'demos/aniso.py']
    return {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def rule(grid,order,gauss):
    h=1/(grid-1);counts=np.rint((HI-LO)/h).astype(int)
    if np.max(abs(counts*h-(HI-LO)))>1e-12:raise ValueError('unaligned physical body')
    if gauss:
        z,w=np.polynomial.legendre.leggauss(order);z=(z+1)/2;w=w/2
    else:z=(np.arange(order)+.5)/order;w=np.full(order,1/order)
    cells=np.array(list(itertools.product(*[range(n) for n in counts])))
    local=np.array(list(itertools.product(z,repeat=3)))
    weights=np.prod(np.array(list(itertools.product(w,repeat=3))),axis=1)*h**3
    return LO+h*(cells[:,None,:]+local).reshape(-1,3),np.tile(weights,len(cells))


def q1_maps(points,grid):
    h=1/(grid-1);counts=np.rint((HI-LO)/h).astype(int);shape=tuple(counts+1)
    local=(points-LO)/h;cell=np.minimum(np.floor(local).astype(int),counts-1);cell=np.maximum(cell,0);f=local-cell
    vertices=cell[:,None,:]+CORNERS;ids=np.ravel_multi_index(vertices.reshape(-1,3).T,shape)
    factors=np.where(CORNERS[None,:,:],f[:,None,:],1-f[:,None,:]);rows=np.repeat(np.arange(len(points)),8)
    G=[]
    for k in range(3):
        values=(2*CORNERS[None,:,k]-1)*np.prod(factors[:,:,[j for j in range(3) if j!=k]],axis=2)/h
        G.append(sp.coo_matrix((values.ravel(),(rows,ids)),shape=(len(points),np.prod(shape))).tocsr())
    nodes=LO+h*np.array(list(itertools.product(*[range(n) for n in shape])))
    return nodes,tuple(G)


def lite_geometry(grid,samples):
    h=1/(grid-1);x,Vp=rule(grid,samples,False);base=np.floor(x/h-.5).astype(int)
    centers=np.unique((base[:,None,:]+CORNERS).reshape(-1,3),axis=0)
    nodes=np.unique((centers[:,None,:]+CORNERS).reshape(-1,3),axis=0)
    maps,F0,V,W,S,D=frozen_maps(x,Vp,np.tile(np.eye(3),(len(x),1,1)),centers,nodes,h)
    return dict(grid=grid,samples=samples,nodes=nodes*h,centers=centers,volume=V,
        particle_volume=Vp,points=x,maps={'center':D,'projected_center':maps,'particle':tuple(S@d for d in D)},
        D=D,weights={'center':V,'projected_center':V,'particle':Vp})


def stiffness(G,V,H):
    """Component-major K, assembled from the same nine deformation derivatives."""
    gram={(i,j):(G[i].T@G[j].multiply(V[:,None])).tocsr() for i in range(3) for j in range(3)}
    blocks=[]
    for a in range(3):
        row=[]
        for c in range(3):
            terms=[H[3*a+b,3*c+d]*gram[b,d] for b in range(3) for d in range(3) if H[3*a+b,3*c+d]!=0]
            row.append(sum(terms) if terms else sp.csr_matrix(gram[0,0].shape))
        blocks.append(row)
    return sp.bmat(blocks,format='csr')


def solve_static(nodes,K,reference=False):
    n=len(nodes);grip=(nodes[:,0]<=.25+1e-12)|(nodes[:,0]>=.75-1e-12)
    free=np.flatnonzero(np.tile(~grip,3));u=np.zeros((3,n));u[0,nodes[:,0]>=.75-1e-12]=.005;u=u.ravel()
    rhs=-(K@u)[free];A=K[free][:,free].tocsr();diag=A.diagonal()
    if np.any(diag<=0):raise ValueError('nonpositive free diagonal')
    if reference:
        y=spsolve(A,rhs);info=0;iterations=1
    else:
        inv=1/np.sqrt(diag);scaled=sp.diags(inv)@A@sp.diags(inv);iterations=0
        def count(_):
            nonlocal iterations
            iterations+=1
        z,info=minres(scaled,inv*rhs,rtol=1e-12,maxiter=20000,callback=count)
        y=inv*z
    u[free]=y;force=K@u;residual=float(np.linalg.norm(force[free])/max(np.linalg.norm(rhs),1e-30))
    field=u.reshape(3,n).T;force_nodes=force.reshape(3,n).T
    # Explicit free checkerboard displacement: zero center gradients, no mass term.
    h=float(np.min(np.diff(np.unique(nodes[:,0]))));ijk=np.rint(nodes/h).astype(int)
    witness=np.zeros((n,3));middle=np.unique(ijk[~grip,0])[len(np.unique(ijk[~grip,0]))//2]
    witness[(ijk[:,0]==middle)&~grip,0]=(-1.)**(ijk[(ijk[:,0]==middle)&~grip,1]+ijk[(ijk[:,0]==middle)&~grip,2])
    z=witness.T.ravel();scale=float(np.max(np.asarray(abs(K).sum(axis=1))))
    null_ratio=float(np.linalg.norm(K@z)/(scale*np.linalg.norm(z)))
    result=dict(linear_info=int(info),iterations=iterations,relative_residual=residual,
        solved=bool(info==0 and residual<=1e-8 and np.isfinite(u).all()),
        free_dofs=len(free),has_demonstrated_zero_mode=null_ratio<1e-10,null_witness_relative=null_ratio,
        reaction_N=float(force_nodes[nodes[:,0]>=.75-1e-12,0].sum()),
        resultant_N=float(np.linalg.norm(force_nodes[grip].sum(axis=0))),
        energy_J=float(.5*u@force),max_displacement=float(np.max(np.linalg.norm(field,axis=1))),
        solution_note='unique fully integrated Q1 reference' if reference else 'compatible singular system; MINRES representative, no diagonal stiffness or mass added')
    return field,result


def center_interpolation(points,centers,h):
    q=points/h-.5;base=np.floor(q).astype(int);f=q-base
    size=int(round(1/h));lookup=np.full(size**3,-1,dtype=np.int32)
    lookup[np.ravel_multi_index(centers.T,(size,)*3)]=np.arange(len(centers))
    corners=base[:,None,:]+CORNERS
    ids=lookup[np.ravel_multi_index(corners.reshape(-1,3).T,(size,)*3)].reshape(-1,8)
    weights=np.prod(np.where(CORNERS[None,:,:],f[:,None,:],1-f[:,None,:]),axis=2)
    if np.any(ids[weights>1e-14]<0):raise ValueError('common evaluation support missing')
    valid=weights>0;rows=np.broadcast_to(np.arange(len(points))[:,None],ids.shape)
    return sp.coo_matrix((weights[valid],(rows[valid],ids[valid])),shape=(len(points),len(centers))).tocsr()


def field_metrics(L,Lref,H,weights):
    P=(L.reshape(-1,9)@H.T).reshape(-1,3,3);Pref=(Lref.reshape(-1,9)@H.T).reshape(-1,3,3)
    norm=lambda a:float(np.sqrt(np.sum(weights[:,None,None]*a*a)))
    return dict(F_minus_I_relative=norm(L-Lref)/max(norm(Lref),1e-30),
        P_relative=norm(P-Pref)/max(norm(Pref),1e-30),min_det_F=float(np.linalg.det(np.eye(3)+L).min()),
        sampled_linear_energy_J=float(.5*np.sum(weights[:,None,None]*L*P)))


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve protocol')
    write_json(out/'protocol.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),source_sha256=source_hashes(),
        type='independent small-strain static spatial audit; zero time-integration error',
        body=[LO.tolist(),HI.tolist()],grips=[.25,.75],displacement=.005,mu=10.,lam=20.,directions=CASES,
        grids=[9,17,25],samples_base=2,sampling_sweep={'grid':17,'samples':[3,4]},modes=MODES,
        reference_grids=[33,49],reference='same-law Hessian at F=I, full Gauss Q1 on physical body',
        common_norm_grid=97,common_norm_gauss=2,
        common_norm_note='partition 1/96 resolves all grid/center gradient breaks; Gauss2 exactly integrates squared piecewise trilinear linearized fields',
        beam_grids=[17,33],force_relative_gate=.05,field_relative_gate=.05,
        reference_reaction_gate=.01,reference_field_gate=.02,linear_residual_gate=1e-8,
        singular_policy='report null witness; no mass/epsilonI; compatible MINRES representative is not a unique displacement solution',
        finite_strain_and_dynamic_accuracy='not certified by static tests'))


class StaticSpaceTests(unittest.TestCase):
    def test_affine_reproduction_and_material_energy(self):
        g=lite_geometry(9,2);A=np.array([[.01,.004,0.],[-.002,-.003,.001],[0.,.002,.005]])
        H=reference_hessian(direction=(1.,1.,0.));u=g['nodes']@A.T
        exact=.5*.046875*A.ravel()@H@A.ravel()
        for mode in MODES:
            G=g['maps'][mode];V=g['weights'][mode]
            L=np.stack([d@u for d in G],axis=2)
            np.testing.assert_allclose(L,np.broadcast_to(A,L.shape),rtol=0,atol=1e-14)
            K=stiffness(G,V,H)
            self.assertAlmostEqual(.5*u.T.ravel()@K@u.T.ravel(),exact,delta=1e-14)

    def test_full_q1_quadrature_and_massless_null_mode(self):
        H=reference_hessian(direction=(1.,1.,0.));h=.125
        K=element_stiffness(h,H,True);z,w=np.polynomial.legendre.leggauss(3);K3=np.zeros_like(K)
        for ix in itertools.product(range(3),repeat=3):
            q=(z[list(ix)]+1)/2;g=shape_gradients(q,h);mapping=np.zeros((9,24))
            for n in range(8):
                for a in range(3):mapping[3*a:3*a+3,3*n+a]=g[n]
            K3+=h**3*np.prod(w[list(ix)]/2)*(mapping.T@H@mapping)
        np.testing.assert_allclose(K,K3,rtol=1e-13,atol=1e-13)
        geo=lite_geometry(9,2)
        for mode in MODES:
            _,result=solve_static(geo['nodes'],stiffness(geo['maps'][mode],geo['weights'][mode],H))
            self.assertTrue(result['solved'],result)
            self.assertTrue(result['has_demonstrated_zero_mode'],result)

    def test_assembly_matches_actual_warp_tangent_without_mass(self):
        import warp as wp
        from engine.types import vec3
        from engine.sp_grid import B
        from demos.aniso import Config,Scene
        wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
        geo=lite_geometry(9,2);H=reference_hessian(direction=(1.,1.,0.))
        rng=np.random.default_rng(29);u=rng.normal(size=geo['nodes'].shape)
        u[(geo['nodes'][:,0]<=.25)|(geo['nodes'][:,0]>=.75)]=0.
        for mode in MODES:
            scene=Scene(Config('tensile',9,.005,45.,quadrature='particle' if mode=='particle' else 'center',
                              history_consistency='projected_center' if mode=='projected_center' else 'standard'),'cpu')
            s=scene.solver;s.step(max_iters=0,print_every=0);self.assertTrue(s.evaluate_residual())
            n=int(s.n_active_nodes.numpy()[0]);addr=s.ndof2bijk[:n].numpy();blocks=s.block_xyz_by_id.numpy()
            actual=blocks[addr[:,0]]*B+np.column_stack((addr[:,1]//(B*B),(addr[:,1]//B)%B,addr[:,1]%B))
            lookup={tuple(np.rint(x/s.dx).astype(int)):i for i,x in enumerate(geo['nodes'])}
            permutation=np.array([lookup[tuple(x)] for x in actual])
            p=wp.zeros_like(s.node_residual);out=wp.zeros_like(p)
            wp.copy(p,wp.array(u[permutation],dtype=vec3,device='cpu'),count=n);s.apply_tangent(p,out)
            mass=s.grid_m[:s.bcn].numpy()[addr[:,0],addr[:,1]//(B*B),(addr[:,1]//B)%B,addr[:,1]%B]
            actual_K=(out[:n].numpy()-mass[:,None]*u[permutation])/s.dt**2
            expected=(stiffness(geo['maps'][mode],geo['weights'][mode],H)@u.T.ravel()).reshape(3,-1).T[permutation]
            expected[(actual[:,0]*s.dx<=.25)|(actual[:,0]*s.dx>=.75)]=0.
            self.assertLess(np.linalg.norm(actual_K-expected)/np.linalg.norm(expected),1e-10)


def tests(out):
    with (out/'tests.log').open('x') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(StaticSpaceTests))
    write_json(out/'tests.json',dict(passed=result.wasSuccessful(),run=result.testsRun,failures=len(result.failures),errors=len(result.errors)))
    return result.wasSuccessful()


def run(out,p):
    if not json.loads((out/'tests.json').read_text())['passed']:raise RuntimeError('tests must pass')
    dest=out/'cases';dest.mkdir(exist_ok=False)
    X,Vcommon=rule(p['common_norm_grid'],p['common_norm_gauss'],True)
    references={};reference_records={};records=[]
    for label,(kf,angle) in CASES.items():
        a=np.deg2rad(angle);H=reference_hessian(kf=kf,direction=(np.cos(a),np.sin(a),0.))
        fields=[]
        for grid in p['reference_grids']:
            start=time.monotonic();x,V=rule(grid,2,True);nodes,G=q1_maps(x,grid)
            u,r=solve_static(nodes,stiffness(G,V,H),True)
            _,Geval=q1_maps(X,grid);L=np.stack([g@u for g in Geval],axis=2);fields.append(L)
            r.update(case=label,grid=grid,mode='full_q1',seconds=time.monotonic()-start)
            energy=float(.5*np.sum(Vcommon[:,None,None]*L*(L.reshape(-1,9)@H.T).reshape(-1,3,3)))
            r['independent_energy_relative']=abs(energy-r['energy_J'])/abs(r['energy_J'])
            write_json(dest/f'reference-{label}-g{grid}.json',r)
            np.savez_compressed(dest/f'reference-{label}-g{grid}.npz',nodes=nodes,u=u)
            reference_records[label,grid]=r;print('reference',label,grid,r['relative_residual'],flush=True)
        references[label]=fields[-1]
        metrics=field_metrics(fields[0],fields[1],H,Vcommon)
        rr0,rr1=[reference_records[label,g] for g in p['reference_grids']]
        metrics.update(reaction_relative=abs(rr0['reaction_N']-rr1['reaction_N'])/abs(rr1['reaction_N']),
                       energy_relative=abs(rr0['energy_J']-rr1['energy_J'])/abs(rr1['energy_J']))
        metrics['passed']=metrics['reaction_relative']<=p['reference_reaction_gate'] and max(metrics['F_minus_I_relative'],metrics['P_relative'])<=p['reference_field_gate']
        write_json(dest/f'reference-gap-{label}.json',metrics)
    choices=[(grid,2) for grid in p['grids']]+[(17,s) for s in (3,4)]
    for grid,samples in choices:
        geo=lite_geometry(grid,samples);interp=center_interpolation(X,geo['centers'],1/(grid-1))
        for label,(kf,angle) in CASES.items():
            a=np.deg2rad(angle);H=reference_hessian(kf=kf,direction=(np.cos(a),np.sin(a),0.))
            for mode in MODES:
                start=time.monotonic();u,r=solve_static(geo['nodes'],stiffness(geo['maps'][mode],geo['weights'][mode],H))
                Lc=np.stack([d@u for d in geo['D']],axis=2)
                L=np.asarray(interp@Lc.reshape(-1,9)).reshape(-1,3,3)
                metric=field_metrics(L,references[label],H,Vcommon);ref=reference_records[label,p['reference_grids'][-1]]
                metric.update(reaction_relative=abs(r['reaction_N']-ref['reaction_N'])/abs(ref['reaction_N']),
                              energy_relative=abs(r['energy_J']-ref['energy_J'])/abs(ref['energy_J']))
                r.update(case=label,grid=grid,samples=samples,mode=mode,particles=len(geo['points']),
                    metrics=metric,seconds=time.monotonic()-start)
                name=f'{label}-g{grid}-p{samples}-{mode}'
                write_json(dest/(name+'.json'),r);np.savez_compressed(dest/(name+'.npz'),nodes=geo['nodes'],u=u)
                records.append(r);print(name,'R',r['reaction_N'],'residual',r['relative_residual'],'null',r['has_demonstrated_zero_mode'],flush=True)
    beam=[]
    for grid in p['beam_grids']:
        for label,(kf,angle) in CASES.items():
            a=np.deg2rad(angle);H=reference_hessian(kf=kf,direction=(np.cos(a),np.sin(a),0.))
            nodes,centers,free,tip,force,matrices=beam_matrices(grid,kf,H)
            r=dict(grid=grid,case=label,free_dofs=len(free),mass_included=False)
            for name,K in zip(('center','full'),matrices):
                vals=np.linalg.eigvalsh(K[free][:,free].toarray());tol=1e-9*max(vals[-1],1.)
                r[name]=dict(min_eigenvalue=float(vals[0]),max_eigenvalue=float(vals[-1]),zero_or_soft_modes=int(np.sum(vals<=tol)))
                if np.all(vals>tol):
                    u=np.zeros(len(force));u[free]=spsolve(K[free][:,free],force[free]);r[name]['tip_displacement']=float(np.mean(u[3*tip+1]))
            beam.append(r);print('beam',grid,label,r['center']['zero_or_soft_modes'],r['full']['zero_or_soft_modes'],flush=True)
    write_json(out/'beam.json',beam)
    write_json(out/'results.json',dict(records=records,completed=len(records)==60,common_evaluation_points=len(X),
        reference_gaps={label:json.loads((dest/f'reference-gap-{label}.json').read_text()) for label in CASES},
        references=[v for v in reference_records.values()],beam=beam))
    return True


def analyze(out,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data=json.loads((out/'results.json').read_text());rows=data['records']
    summary=dict(completed=data['completed'],cases=len(rows),all_linear_checks_passed=all(r['solved'] for r in rows),
        references_solved=all(r['solved'] for r in data['references']),
        max_reference_quadrature_energy_gap=max(r['independent_energy_relative'] for r in data['references']),
        reference_resolution_passed=all(r['passed'] for r in data['reference_gaps'].values()),
        reference_gaps=data['reference_gaps'],zero_mode_cases=sum(r['has_demonstrated_zero_mode'] for r in rows),
        unique_static_displacement_verified=False,finite_strain_accuracy_verified=False,
        beam_center_has_zero_modes=all(r['center']['zero_or_soft_modes']>0 for r in data['beam']),
        beam_full_has_no_zero_modes=all(r['full']['zero_or_soft_modes']==0 for r in data['beam']))
    summary['five_percent_screen_passed']=all(max(r['metrics'][k] for k in ('reaction_relative','F_minus_I_relative','P_relative','energy_relative'))<=.05 for r in rows if r['grid']==25)
    write_json(out/'summary.json',summary)
    fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True)
    for ax,label in zip(axes.flat,CASES):
        for mode in MODES:
            selected=sorted([r for r in rows if r['case']==label and r['mode']==mode and r['samples']==2],key=lambda r:r['grid'])
            ax.plot([1/(r['grid']-1) for r in selected],[100*r['metrics']['reaction_relative'] for r in selected],marker='o',label=mode)
        ax.set(title=label,xlabel='grid spacing (m)',ylabel='reaction difference to fine Q1 (%)');ax.grid(alpha=.25);ax.legend()
    fig.savefig(out/'spatial-reactions.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))
    return bool(summary['all_linear_checks_passed'] and summary['references_solved'] and summary['beam_full_has_no_zero_modes'])


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('freeze','tests','run','analyze'))
    parser.add_argument('--output',type=Path,default=DEFAULT);args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    if args.action=='freeze':freeze(out);return
    p=json.loads((out/'protocol.json').read_text())
    if p['source_sha256']!=source_hashes():raise RuntimeError('frozen spatial source changed')
    ok=tests(out) if args.action=='tests' else run(out,p) if args.action=='run' else analyze(out,p)
    raise SystemExit(0 if ok else 2)


if __name__=='__main__':main()
