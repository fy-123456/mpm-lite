"""Independent constitutive/patch/APIC terminal energy and force-work audit."""
import json
import numpy as np
from benchmarks.aniso_v17_time import ROOT,OUT,load,write
from benchmarks.aniso_v17_analysis import arrays
from benchmarks.aniso_compatible_diagnosis import maps,gradient
from benchmarks.aniso_dynamic_check import pk1
from engine.aniso_phase1.diagnostics import energy_density
from engine.aniso_phase1.types import AnisotropicMaterialParams


def main():
    assert not (OUT/'independent-physical-audit.json').exists();protocol=load(OUT/'protocol.json');records=[];params=AnisotropicMaterialParams(10.,20.,200.)
    for name,spec in protocol['cases'].items():
        source=arrays(ROOT/protocol['inputs'][spec['label']]['path']);state=arrays(OUT/'cases'/name/'terminal.npz');series=arrays(OUT/'cases'/name/'series.npz')
        row=json.loads((OUT/'cases'/name/'steps.jsonl').read_text().splitlines()[-1]);h=.125
        X=source['patch_X'];_,G0=maps(source['particle_reference_x'],np.rint(X/h).astype(int),h)
        R=np.linalg.solve(gradient(G0,source['marker_after']),source['particle_F_after'])
        B=[sum(G0[j].multiply(R[:,j,k,None]) for j in range(3)) for k in range(3)]
        V=source['particle_volume'];A=source['particle_A0'];m=source['particle_mass'];ids=source['patch_ids'];P=source['patch_P'];weights=source['patch_weight']
        def elastic(Y):
            F=gradient(G0,Y)@R;Pm=pk1(F,A,200.);Um=float(V@energy_density(F,A,params))
            residual=np.einsum('cij,cja->cia',P,Y[ids]);Us=.5*float(np.sum(weights[:,None,None]*residual**2))
            force=np.zeros_like(Y);local=weights[:,None,None]*np.einsum('cji,cja->cia',P,residual)
            np.add.at(force,ids.ravel(),local.reshape(-1,3));force+=sum(b.T@(V[:,None]*Pm[:,:,k]) for k,b in enumerate(B))
            return F,Pm,Um,Us,force
        frac=state['x']/h-.5;frac-=np.floor(frac);D=h*h*(frac*(1-frac)+.25)
        def kinetic(v,C):return .5*float(np.sum(m[:,None]*v*v)+np.sum(m[:,None,None]*D[:,None,:]*C*C))
        F,stress,Um,Us,_=elastic(state['Y']);_,_,Um0,Us0,_=elastic(state['Y_before'])
        dy=state['Y']-state['Y_before'];avg=sum(.5*elastic(state['Y_before']+a*dy)[-1] for a in (.5-np.sqrt(3)/6,.5+np.sqrt(3)/6));work=float(np.sum(avg*dy))
        K=kinetic(state['v'],state['C']);K0=kinetic(state['v_before'],state['C_before']);defect=K-K0+work;quad=Um+Us-Um0-Us0-work
        errors=dict(F=float(np.max(abs(F-state['F']))),stress_Pa=float(np.max(abs(stress-series['P'][-1]))),material_J=abs(Um-row['material_J']),
            stabilization_J=abs(Us-row['stabilization_J']),kinetic_J=abs(K-row['kinetic_J']),
            work_defect_J=abs(defect-row['kinetic_force_work_defect_J']),quadrature_J=abs(quad-row['potential_quadrature_error_J']))
        assert errors['F']<1e-11 and errors['stress_Pa']<1e-9 and max(v for k,v in errors.items() if k.endswith('_J'))<1e-12,(name,errors)
        records.append(dict(case=name,errors=errors,independent_work_defect_J=defect,independent_path_error_J=quad))
    write(OUT/'independent-physical-audit.json',dict(passed=True,records=records,
        method='Rebuild reference G0/R from original source, independent Piola stress and strain energy, direct patch row energy/forces, direct APIC v/C kinetic moment formula. No CarrierEnergy.evaluate or step diagnostics used to reconstruct values.'))
    print('independent physical audits',len(records),flush=True)
if __name__=='__main__':main()
