"""Summarize archived production/legacy comparisons without new simulations."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('docs/results/corotated');old=Path('docs/results/rotation-dissipation')
    def read(folder,name):return np.genfromtxt(folder/(name+'.csv'),delimiter=',',names=True)
    comparisons={}
    new={k:read(root,f'tensile-{k}-corotated')[1:] for k in ('center','particle')}
    comparisons['center_particle_force_curve_relative']=float(np.linalg.norm(new['center']['right_force']-new['particle']['right_force'])/np.linalg.norm(new['particle']['right_force']))
    for kind in ('center','particle'):
        a=new[kind];b=read(old,f'tensile-{kind}-supplemental-T0.16')[1:]
        dwork=a['right_force']*np.diff(np.r_[0,a['displacement']])
        comparisons[kind]=dict(force_relative_to_supplemental=float(np.linalg.norm(a['right_force']-b['right_force'])/np.linalg.norm(b['right_force'])),
            loading_work=[float(dwork[i:i+32].sum()) for i in (0,64)],
            repeat_force_relative=float(np.linalg.norm(a['right_force'][64:]-a['right_force'][:64])/np.linalg.norm(a['right_force'][:64])))
    comparisons['loading_work_relative_center_particle']=[a/b-1 for a,b in zip(comparisons['center']['loading_work'],comparisons['particle']['loading_work'])]
    (root/'comparison.json').write_text(json.dumps(comparisons,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,axis in zip(axes[0],('y','z')):
        suffix='-y' if axis=='y' else ''
        for folder,mode,grid,style in ((old,'supplemental',17,'--'),(root,'corotated',17,'-'),(root,'corotated',33,':')):
            name=f'rotation-g{grid}-dt0.01-{mode}'+suffix
            a=read(folder,name);E0=read(folder,name+'-budget')['elastic'][0]
            ax.plot(a['stabilization_sampling_angle_degrees'],100*((a['center_material_energy']+a['stabilization_energy'])/E0-1),style,label=f'{mode}, grid {grid}',lw=2)
        ax.set(title=f'Rigid rotation about {axis}',xlabel='Step-start sampling angle (deg)',ylabel='Total elastic change (%)');ax.legend(fontsize=8);ax.grid(alpha=.3)
    recon=json.loads((root/'reference-reconstruction.json').read_text());ax=axes[1,0]
    for j,key in enumerate(('particle_reconstructed_stabilization_energy','analytic_reference_stabilization_energy')):
        values=[]
        for grid in (17,33):
            a,b=[r for r in recon if r['grid']==grid];values.append(100*(b[key]/a[key]-1))
        ax.bar(np.arange(2)+(j-.5)*.35,values,.35,label='particle history' if j==0 else 'analytic reference')
    ax.set_xticks(range(2),['grid 17','grid 33']);ax.set(title='Remaining fixed-grid error: z=45 deg',ylabel='Stabilization energy change (%)');ax.legend(fontsize=8)
    ax=axes[1,1]
    for mode,folder,name,style in [('supplemental',old,'release-supplemental-dt0.0005-flip0.9','--'),('corotated',root,'beam-corotated-dt0.0005','-')]:
        a=read(folder,name);ax.plot(a['time'],100*(a['mechanical']/a['mechanical'][0]-1),style,label=mode,lw=2)
    ax.set(title='Small-bend release, dt=.0005',xlabel='Time',ylabel='Mechanical energy change (%)');ax.legend(fontsize=8);ax.grid(alpha=.3)
    fig.savefig(root/'summary.png',dpi=160);fig.savefig(root/'summary.svg');plt.close(fig)


if __name__=='__main__':main()
