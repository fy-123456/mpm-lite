"""Stress-shape MAC matching across independent grids and particle samplings.

Matches indicate shape similarity, not a converged eigenfrequency or identical
excitation history. Degenerate modes may rotate; the individual best match is
reported together with alternatives rather than assumed to be a tracked mode.
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from benchmarks.aniso_v17_modes import controlled_case,ModalModel,snapshot_model
from benchmarks.aniso_v17_time import ROOT,OUT,sha,write


def main():
    assert not (OUT/'stress-mode-matching.json').exists()
    write(OUT/'stress-mode-matching-protocol.json',dict(source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'benchmarks/aniso_v17_mode_match.py',ROOT/'benchmarks/aniso_v17_modes.py']},
        targets='late snapshot dominant stress mode 139 and dominant time-error mode 211',common_points='original 192 reference particle centers',metric='volume weighted full Piola stress-shape MAC'))
    a=snapshot_model(1.6);xp=a.particle_reference;V=a.e.V/a.e.V.sum();targets=[int(a.order[0]),211];records=[]
    for h in (1/8,1/10,1/12):
        for ns in (4,6,8):
            s,e,m,_,meta=controlled_case(h,ns,0);model=ModalModel(s,e,m,h,meta['particle_reference'],meta['carrier_reference'])
            axes=[np.unique(meta['particle_reference'][:,k]) for k in range(3)]
            values=model.D.transpose(1,0,2,3).reshape(*(len(v) for v in axes),-1)
            D=RegularGridInterpolator(axes,values,bounds_error=True)(xp).reshape(len(xp),len(model.omega),3,3).transpose(1,0,2,3)
            norm=np.einsum('ipab,ipab,p->i',D,D,V);rows=[]
            for target in targets:
                ref=a.D[target];rn=np.einsum('pab,pab,p->',ref,ref,V);dot=np.einsum('ipab,pab,p->i',D,ref,V)
                mac=dot*dot/(norm*rn);mac[norm<1e-16*norm.max()]=0.;order=np.argsort(-mac)
                candidates=[]
                for j in order[:3]:
                    p=model.phi[:,j];ks=float(p@model.Ks@p)/model.omega[j]**2
                    candidates.append(dict(mode=int(j),stress_shape_MAC=float(mac[j]),omega_rad_s=float(model.omega[j]),stabilization_fraction=ks))
                rows.append(dict(target_mode=target,target_omega_rad_s=float(a.omega[target]),candidates=candidates))
            records.append(dict(h=h,ns=ns,matching=rows));print(h,ns,rows,flush=True)
    write(OUT/'stress-mode-matching.json',dict(completed=True,records=records,
        interpretation='Common physical stress-shape comparison with three best finite-mode candidates. Reference is the same late snapshot local linearization; scan states are undeformed controlled states, so frequency differences include tangent/history and discretization differences. This does not establish continuum spatial convergence.',
        no_frequency_prefilter=True,no_interpolation_extrapolation=True))
if __name__=='__main__':main()
