"""Small actual 2D Biot block/work/mass audit; no claim of C--E 3D coupling."""
from pathlib import Path
import argparse
import numpy as np
from .provenance import read,write,register,serial_lock,utc,PARENT_RELEASE_SHA
from benchmarks.research_e.stage2.experiments import make_model
from engine.aniso_phase1.research_sequential.solver import GeneralBiot,solve_mixed
from engine.aniso_phase1.research_e.stage2.transaction import ChildAdapter
from engine.aniso_phase1.research_d.common_state import CommonState,StateTransaction


def study(run):
    run=Path(run)
    register(run,'S5/coupling-protocol.json',dict(grid=[4,4],displacement='existing Q2',pressure='existing P0',flux='existing RT0',
        dtype='float64',device='cpu',dt=[.01,.015,.005,.01],solver='general direct',
        checks=['pressure work cancellation','local/global fluid content','nonnegative Darcy loss','mixed rank and residual','whole-state rollback'],
        physical_3D_coupling=False,C_E_force_flux_exchange=False,solid_inertia=False))
    b=make_model(4);solver=GeneralBiot(b.solid,b.flow,b.storage);s,g=solver.solid,solver.flow.grid
    adapter=ChildAdapter(solver,PARENT_RELEASE_SHA)
    child=adapter.initial();initial=CommonState(np.zeros((len(s.nodes),3)),np.zeros((len(s.nodes),3)),child_states={'E':child})
    def validate(state):
        adapter.validate_parent(state)
        if not np.array_equal(state.q[:,:2].ravel(),np.asarray(state.child_states['E']['u'])):raise ValueError('parent u differs from actual coupled child')
    transaction=StateTransaction(initial,validator=validate)
    boundary={(0,0):('flux',0.),(0,1):('pressure',0.),(1,0):('flux',0.),(1,1):('flux',0.)}
    load=s.traction(0,1,[-.05,0.]);src=.001;records=[]
    def prepare(trial,dt):
        old=trial.state.child_states['E']
        proposed=adapter.prepare(old,dt=dt,load=load,boundary=boundary,source=src)
        trial.state.child_states['E']=proposed
        trial.state.q[:,:2]=np.asarray(proposed['u']).reshape(-1,2)
        trial.state.velocity[:,:2]=(trial.state.q[:,:2]-np.asarray(old['u']).reshape(-1,2))/dt
        trial.state.time+=dt;trial.state.step+=1
        return old,proposed
    trial=transaction.begin_trial();before=transaction.snapshot().digest();prepare(trial,.01);transaction.rollback(trial)
    assert transaction.snapshot().digest()==before
    for dt in [.01,.015,.005,.01]:
        trial=transaction.begin_trial();old,new=prepare(trial,dt)
        du=np.asarray(new['u'])-np.asarray(old['u']);p=np.asarray(new['p']);dp=p-np.asarray(old['p']);z=np.asarray(new['flux'])
        wm=-float(du@(s.G.T@p));wp=float(p@(s.G@du))
        local=s.G@du+solver.C@dp+dt*solver.flow.B@z-dt*g.volume*src
        m=new['ledger'][-1]
        record=dict(time_s=new['time'],dt=dt,mechanical_pressure_work_J=wm,fluid_pressure_work_J=wp,
                    pressure_work_defect_J=wm+wp,local_mass_max=float(np.max(abs(local))),global_mass_abs=float(abs(local.sum())),
                    Darcy_dissipation_J=float(dt*z@(solver.flow.H@z)),energy_residual_J=m['energy_residual'],
                    mixed_residual=m['true_residual'])
        assert abs(wm+wp)<1e-10 and record['local_mass_max']<1e-8 and record['Darcy_dissipation_J']>=-1e-12
        assert abs(record['energy_residual_J'])<1e-8 and record['mixed_residual']<1e-8
        transaction.commit(trial);records.append(record)
    old=solver.initial();matrix,rhs,*_=solver._system(old,.01,load,boundary,src)
    dense=matrix.toarray();rank=int(np.linalg.matrix_rank(dense));n=len(rhs)
    assert rank==n
    try:solve_mixed(matrix,rhs,method='cg')
    except ValueError:cg_rejected=True
    else:cg_rejected=False
    assert cg_rejected
    final=transaction.snapshot();np.savez_compressed(run/'S5/coupled-2d-small.npz',u=final.q,p=np.array(final.child_states['E']['p']),flux=np.array(final.child_states['E']['flux']))
    write(run/'S5/coupling-check.json',dict(status='passed_scoped',steps=4,records=records,
        actual_mixed_matrix_shape=list(matrix.shape),rank=rank,cg_rejected=True,SPD_assumed=False,
        same_G_in_pressure_work_and_content=True,whole_transaction_rollback=True,
        physical_2D_coupling=True,physical_3D_coupling=False,C_E_force_flux_exchange=False,
        scope='existing shared-grid 2D Q2/P0/RT0 equations; closure only, no new accuracy certificate'))
    print('COUPLING_CLOSURE',n,rank,max(abs(x['pressure_work_defect_J']) for x in records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    with serial_lock(a.run):study(a.run)
