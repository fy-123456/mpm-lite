"""Read-only reduction of short physical evidence into scoped comparisons."""
import argparse,json
from pathlib import Path
import numpy as np
from .common import write

def assess(run):
    root=run/'S4';read=lambda p:json.loads(p.read_text())
    names=('closed-compressed','closed-full','drained-compressed','dry-compressed','closed-compressed-half-dt')
    rows={n:read(root/n/'metrics.json') for n in names if (root/n/'summary.json').exists()}
    report=dict(cases={})
    for n,rr in rows.items():
        summary=read(root/n/'summary.json');b=summary['budget']
        report['cases'][n]=dict(steps=len(rr),seconds=b['seconds'],components=b['component_seconds'],material_calls=b['counts'].get('material',0),
            minJ=min(x['minJ'] for x in rr),jv_calls=summary['jv_calls'],peak_rss_bytes=summary['peak_rss_bytes'],
            max_mass_defect=max(float(np.linalg.norm(x['mass_defect'])) for x in rr),max_energy_defect=max(abs(x['energy_defect']) for x in rr),
            min_dissipation=min(x['darcy_dissipation'] for x in rr),max_free_residual_ratio=max(x['free_residual_norm']/x['free_only_tolerance_diagnostic'] for x in rr),
            max_rule_error=max((x['material_comparison']['max_error'] if x['material_comparison'] else 0) for x in rr),
            final_reaction=rr[-1]['reaction_right_x'],final_pressure=rr[-1]['pressure'])
    def rel(a,b,floor=1e-8):return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b),floor))
    c=rows['closed-compressed'][0]
    if 'closed-full' in rows:
        f=rows['closed-full'][0];gaps={}
        with np.load(root/'closed-compressed/first-checkpoint.npz') as a,np.load(root/'closed-full/checkpoint.npz') as b:
            for k in ('q','velocity','p'):
                if k in a:gaps[k]=dict(max_absolute=float(np.max(abs(a[k]-b[k]))),relative=rel(a[k],b[k]))
        # From start to accept includes guard; cost excludes later field/checkpoint IO.
        report['material_comparison']=dict(reaction_relative=rel(c['reaction_right_x'],f['reaction_right_x']),pressure_relative=rel(c['pressure'],f['pressure']),states=gaps,
            compressed_first_accept_seconds=c['shared_budget']['seconds'],full_first_accept_seconds=f['shared_budget']['seconds'],
            compressed_solver_seconds=c['seconds'],full_solver_seconds=f['seconds'],scope='one matched step; q5 cost includes q7 endpoint guard, timings are single samples')
    if 'drained-compressed' in rows:
        d=rows['drained-compressed'][0]
        report['mechanical_comparison']=dict(eta_reaction=rel(d['reaction_right_x'],c['reaction_right_x']),closed_pressure=c['pressure'],drained_pressure=d['pressure'],
            stress_relative=rel(d['mean_PK1_by_slab'],c['mean_PK1_by_slab'],1e-5),scope='same dt/target, step-average grip reaction')
    if 'closed-compressed-half-dt' in rows:
        rr=rows['closed-compressed-half-dt'];dt=read(root/'closed-compressed/summary.json')['dt'];ref=sum(x['reaction_right_x']*dt/2 for x in rr);base=c['reaction_right_x']*dt
        with np.load(root/'closed-compressed/first-checkpoint.npz') as a,np.load(root/'closed-compressed-half-dt/checkpoint.npz') as b:
            fields={k:dict(max_absolute=float(np.max(abs(a[k]-b[k]))),relative=rel(a[k],b[k])) for k in ('q','velocity','p') if k in a}
        with np.load(root/'closed-compressed/frames.npz') as a,np.load(root/'closed-compressed-half-dt/frames.npz') as b:
            du=a['x'][1]-b['x'][-1];u=b['x'][-1]-b['X']
            probes=dict(max_displacement_gap_m=float(np.linalg.norm(du,axis=1).max()),relative_displacement_L2=float(np.linalg.norm(du)/max(np.linalg.norm(u),1e-12)),max_F_gap=float(abs(a['F'][1]-b['F'][-1]).max()))
        report['time_comparison']=dict(probes=probes,impulse_relative=rel(base,ref,1e-12),coarse_impulse=base,half_step_impulse=ref,fields=fields,
            scope='same endpoint time and target; compare integrated reaction, never compare different midpoint reactions as same-time values')
    report['limitations']=['single45deg material','two material pressure volumes','constant reference hydraulic network','no independent spatial reference','no production transfer or GPU','no temporal accuracy certificate']
    write(run/'S7/assessment.json',report);return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);print(json.dumps(assess(p.parse_args().run),indent=2))
