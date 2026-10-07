"""Separate material-rule changes from grid-dependent history reconstruction.

Material points carry positive reference volume under exact bend/rigid maps.
This is a controlled history audit, not a deformed-domain clipping algorithm.
"""
import argparse,time
from pathlib import Path
import numpy as np
from engine.aniso_phase1.aligned_quadrature import box_rule
from engine.aniso_phase1.transfer_compatibility import history_read
from engine.aniso_phase1.stabilization_probe import bent_beam_state
from engine.aniso_phase1.diagnostics import energy_density
from engine.aniso_phase1 import AnisotropicMaterialParams
from benchmarks.aniso_material_snapshot import make_snapshot
from benchmarks.aniso_rotation_dissipation import guard,table
from benchmarks.aniso_quadrature_validation import save


def run(grid,ppc,out):
    h=1/(grid-1);s=make_snapshot(grid=grid,ppc_axis=ppc,field='smooth');params=AnisotropicMaterialParams(10,20,200)
    records=[];reference=None
    for angle,shift in [(0,(0,0,0)),(0,(.35*h,.2*h,0)),(0,(2*h,0,0)),(45,(0,0,0)),(90,(0,0,0))]:
        guard();theta=np.deg2rad(angle);Q=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
        x=(s.x-.5)@Q.T+.5+shift;F=Q@s.F;readings={}
        for order in (3,4,5):
            rule=box_rule(h,order);xq,Fq=bent_beam_state(rule.points,amplitude=.01)
            xq=(xq-.5)@Q.T+.5+shift;Fq=Q@Fq
            # Fibers are material-reference directions paired with each Xq.
            a=(rule.points[:,1]-.4375)/.125*np.pi/2
            axis=np.column_stack([np.cos(a),np.sin(a),np.zeros(len(a))]);A=np.einsum('qi,qj->qij',axis,axis)
            rebuilt,stats=history_read(x,s.X,F,s.volume,xq,h)
            exact=float(rule.weights@energy_density(Fq,A,params));U=float(rule.weights@energy_density(rebuilt,A,params))
            readings[str(order)]=dict(exact_energy=exact,rebuilt_energy=U,history_delta=U-exact,
                relative_history=U/exact-1,F_relative=float(np.linalg.norm(rebuilt-Fq)/np.linalg.norm(Fq)),
                min_det=float(np.linalg.det(rebuilt).min()),samples=len(rule.weights),**stats)
        if reference is None:reference=readings
        record=dict(grid=grid,ppc_axis=ppc,angle=angle,shift=list(shift),readings=readings,
            exact_objectivity=float(readings['3']['exact_energy']/reference['3']['exact_energy']-1),
            rebuilt_pose_change=float(readings['3']['rebuilt_energy']/reference['3']['rebuilt_energy']-1),
            same_state_rule_delta=readings['4']['rebuilt_energy']-readings['3']['rebuilt_energy'],
            same_state_rule_relative=readings['4']['rebuilt_energy']/readings['3']['rebuilt_energy']-1,
            high_rule_history_convergence=readings['5']['rebuilt_energy']/readings['4']['rebuilt_energy']-1)
        records.append(record);print('REBUILD',grid,ppc,angle,shift,record['rebuilt_pose_change'],record['same_state_rule_relative'],flush=True)
    result=dict(grid=grid,ppc_axis=ppc,records=records,
        scope='rebuild spatial Hermite fits after exact motion, compare carried material reference rules; no Eulerian cut-domain rule claimed')
    save(out/f'g{grid}-ppc{ppc}.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grids',nargs='+',type=int,default=[17,33]);p.add_argument('--ppc',type=int,default=2)
    p.add_argument('--out',type=Path,default=Path('docs/results/quadrature-validation/rebuild-v1'));args=p.parse_args()
    guard();args.out.mkdir(parents=True,exist_ok=True);results=[]
    for grid in args.grids:
        results.append(run(grid,args.ppc,args.out));save(args.out/'results.json',results)


if __name__=='__main__':main()
