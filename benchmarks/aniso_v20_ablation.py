"""Two-by-two same-dt full cycles, keeping material/patch potential unchanged."""
import numpy as np
from benchmarks.aniso_v20_common import *
from benchmarks.aniso_v20_analysis import rows,compare,STAGES,rms
from benchmarks.aniso_v19_analysis import frequency

def main():
    manifest=load(OUT/'completed-case-paths.json');assert manifest['completed'];paths={n:ROOT/v['path'] for n,v in manifest['cases'].items()};folders={'sampled_original':BASE/'v19/cases/cycle-L2','gauss3_original':paths['gauss3-original-ablation'],'sampled_condensed':paths['sampled-condensed-ablation'],'gauss3_condensed':paths['gauss3-condensed-L2']};_,e,_,_,_=controlled_case();cases={}
    for name,folder in folders.items():
        rr=rows(folder);assert len(rr)==12800 and rr[0]['dt']==.000125;cases[name]=dict(final_hold_spectrum=frequency(rr),constraint_loss_J=sum(r['constraint_kinetic_loss_J'] for r in rr),terminal_stress_rms_Pa=rr[-1]['stress_rms_Pa'],terminal_energy_J=rr[-1]['total_J'],phase_reaction_rms_N={phase:rms([r['reaction_N'] for r in rr[round(lo/.000125):round(hi/.000125)]]) for phase,(lo,hi) in STAGES.items()},raw_reaction_peak_N=max(abs(r['reaction_N']) for r in rr))
    pairs={}
    for name,a,b in [('inertia_without_condensation','sampled_original','gauss3_original'),('inertia_with_condensation','sampled_condensed','gauss3_condensed'),('condensation_with_sampled_inertia','sampled_original','sampled_condensed'),('condensation_with_gauss3_inertia','gauss3_original','gauss3_condensed')]:pairs[name]=dict(first=a,second=b,comparison=compare(folders[a],folders[b],e.V,True))
    write(OUT/'full-cycle-ablation.json',dict(completed=True,dt=.000125,cases=cases,pairs=pairs,scope='Full cycles from the same stress-free zero-velocity state. Each edge changes one model factor. Between-model differences do not alone certify continuum accuracy or time convergence; only the four-level candidate has its own convergence test.'))
if __name__=='__main__':main()
