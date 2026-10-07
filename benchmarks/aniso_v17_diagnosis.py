"""Run immutable modal attribution and independent h/particle-density scans."""
import argparse,json
import numpy as np
import scipy.linalg as la
from benchmarks.aniso_v17_time import OUT,load,write,sources
from benchmarks.aniso_v17_modes import ModalModel,snapshot_model,controlled_case
from benchmarks.aniso_compatible_diagnosis import zones


def common_rest_null():
    s,e,m,h,meta=controlled_case();a=ModalModel(s,e,m,h,meta['particle_reference'],meta['carrier_reference'])
    lift=la.block_diag(a.Q,a.Q,a.Q);Z=lift@a.null_basis
    return la.orth(Z),a


def annotate_overlap(model,record,Z):
    lift=la.block_diag(model.Q,model.Q,model.Q);phi=lift@model.phi
    overlap=np.sum((Z.T@phi)**2,axis=0)/np.sum(phi**2,axis=0)
    for i,r in enumerate(record['modes']):r['reference_rest_zero_inertia_displacement_overlap']=float(overlap[i])
    high=overlap>.9
    record['reference_rest_null_overlap_gt90']=dict(modes=int(high.sum()),stress_self_power_fraction=float(model.score[high].sum()/model.score.sum()),
        omega_range_rad_s=[float(model.omega[high].min()),float(model.omega[high].max())] if high.any() else [])


def snapshots():
    assert not (OUT/'modal-snapshots.json').exists();Z,rest=common_rest_null();records=[]
    for label,t in [('early',1.1),('late',1.6)]:
        a=snapshot_model(t);r=a.record();r.update(label=label,time=t);annotate_overlap(a,r,Z);records.append(r)
        # Continuous-time traces resolve the largest finite frequency with >13
        # samples per period. All tensor comparisons later use exact coordinates.
        times=np.linspace(0,.05,50001);q=a.coordinates(times);masks=zones(a.particle_reference)
        rows=[];names=[]
        for name,mask in [('global',np.ones(len(a.e.V),bool)),*masks.items()]:
            if not mask.any():continue
            v=a.e.V[mask];v=v/v.sum();coef=np.einsum('ip,p->i',a.D[:,mask,0,0],v)
            rows.append(float(v@a.P0[mask,0,0])+coef@q);names.append(name+'_mean_Pxx')
        ip=int(np.argmax(np.sum(a.D[a.order[0]]**2,axis=(1,2))))
        rows.append(a.P0[ip,0,0]+a.D[:,ip,0,0]@q);names.append(f'particle_{ip}_Pxx')
        top=a.order[:20]
        np.savez_compressed(OUT/f'modal-{label}.npz',omega=a.omega,eq=a.eq,v0=a.v0,phi=a.phi,D=a.D,P0=a.P0,gram=a.gram,score=a.score,
            reference_x=a.particle_reference,reference_Y=a.carrier_reference,time=times,trace=np.array(rows),trace_names=np.array(names),
            top_indices=top,top_q=q[top],null_overlap=np.array([v['reference_rest_zero_inertia_displacement_overlap'] for v in r['modes']]))
        print('snapshot',label, 'top',r['modes'][r['stress_ranking'][0]],flush=True)
    write(OUT/'modal-snapshots.json',dict(completed=True,records=records,rest_reference_zero_inertia_modes=rest.null_count,
        note='Same physical snapshots; ordering by tensor stress contribution, not maximum frequency. Modal stiffness and zero-inertia overlap are local discrete diagnostics.'))


def scans():
    assert not (OUT/'spatial-mode-scan.json').exists();records=[]
    for h in (1/8,1/10,1/12):
        for ns in (4,6,8):
            for amp in (0.,.0001):
                s,e,m,_,meta=controlled_case(h,ns,amp);a=ModalModel(s,e,m,h,meta['particle_reference'],meta['carrier_reference']);r=a.record()
                r.update(h=h,ns=ns,amplitude=amp,particles=len(s.x),carriers=len(s.Y));records.append(r)
                print('scan',h,ns,amp,'null',a.null_count,'top_omega',a.omega[a.order[0]],flush=True)
    # Same grid/sampling, only the deformation amplitude changes; separates
    # geometry-induced tiny inertia from physical resolution changes.
    Z,rest=common_rest_null();ladder=[]
    for amp in (.001,.0003,.0001,.00003):
        s,e,m,h,meta=controlled_case(amplitude=amp);a=ModalModel(s,e,m,h,meta['particle_reference'],meta['carrier_reference']);r=a.record();annotate_overlap(a,r,Z)
        r.update(amplitude=amp);ladder.append(r);print('amplitude',amp,a.omega[-1],flush=True)
    write(OUT/'spatial-mode-scan.json',dict(completed=True,records=records,amplitude_ladder=ladder,
        geometry=dict(lo=[.125,.375,.375],hi=[.875,.625,.625],clamps=[.25,.75]),
        physical_volume=.046875,material=dict(mu=10.,lam=20.,kf=200.,direction='F45'),
        scope='Controlled common analytic displacement/velocity field, same physical bounds/BC/material. Separate grid size and sampling scans, not resampled unload histories or a continuum stress reference.',
        singular_analysis='Strict kinetic nullspace retained as algebraic constraint via exact stiffness Schur complement; no artificial mass, no regularization. Finite modes only are stress ranked.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['snapshots','scan']);a=p.parse_args()
    assert sources()==load(OUT/'protocol.json')['source_sha256']
    if a.action=='snapshots':snapshots()
    else:scans()
if __name__=='__main__':main()
