"""One registered unused state validates the diagnostic; zero dynamic advances."""
import argparse
import numpy as np
import scipy.linalg as la
from .provenance import *
from .fixture import inputs
from .selected import selected_setup
from .runtime import update
from benchmarks.research_sequential_next.checkpoint import GenerationStore
from benchmarks.research_sequential_next.compare import metric
from engine.aniso_phase1.research_coupled_reference_next.reference import affine_operator,exact

def main(run):
    run=Path(run);mutable(run)
    register(run,'S1/holdout-protocol.json',dict(status='registered',step=16,time_s=50e-6,training_steps=[8,12,18],new_dynamic_steps=0,reason='one unused existing state checks snapshot overfitting; keep practical engineering budgets',pressure_atol_Pa=.001,displacement_atol_m=5e-5,velocity_atol_m_s=1e-4,rtol=.05))
    c,m,cfg,ident=selected_setup(run);h=inputs();s=h[16]['state'];z=dict(np.load(run/'S1/linear-blocks.npz'));U=z['U'];r=U.shape[1];G=z['G']@U;K=z['Ksolid']+z['Kpressure']
    A,drift=affine_operator(U.T@K,G,z['C'],z['L'],z['D'],U.T@z['M']@z['acc'],z['pdot'])
    scale=np.r_[np.maximum(abs(U.T@z['M']@s.q[m.free].ravel()),1e-10),np.maximum(abs(U.T@z['M']@s.velocity[m.free].ravel()),1e-5),np.full(len(z['C']),.2)]
    y=exact(A,drift,[s.time],scale=scale)[0];pp=z['p0']+y[2*r:];q=np.zeros_like(s.q);q[m.free]=(U@y[:r]).reshape(-1,3);v=np.zeros_like(s.velocity);v[m.free]=(U@y[r:2*r]).reshape(-1,3)
    axes=tuple(np.linspace(e[0],e[-1],9) for e in m.parent.edges)
    def fields(a):return m.parent._sample(m.parent.nodes(m.reduction.velocity(a)),axes)[0]
    actual_p=np.array(s.child_states['fluid']['pressure_Pa']);checks=dict(pressure=metric(pp,actual_p,.001,.05),displacement=metric(fields(q),fields(s.q),5e-5,.05),velocity=metric(fields(v),fields(s.velocity),1e-4,.05))
    geo=c.geometry.evaluate(s.q);GG=geo['gradient'][:,m.free].reshape(len(z['C']),-1);flux=la.cho_solve(la.cho_factor(geo['H']),z['B'].T@actual_p-c.core.gb)
    x=U.T@z['M']@s.q[m.free].ravel();vv=U.T@z['M']@s.velocity[m.free].ravel()
    true_pd=(-.8*GG@s.velocity[m.free].ravel()-z['B']@flux)/z['C'];lin_pd=z['pdot']+(-.8*G@vv-z['D']@x-z['L']@(actual_p-z['p0']))/z['C']
    mf=la.cho_factor(z['M']);f=m.evaluate(s.q)['force'][m.free].ravel();true_a=la.cho_solve(mf,-f+.8*GG.T@actual_p);lin_a=z['acc']+la.cho_solve(mf,-K@x+.8*z['G'].T@(actual_p-z['p0']))
    passed=all(x['passed'] for x in checks.values())
    write(run/'S1/holdout-review.json',dict(status='passed_scoped' if passed else 'limited',step=16,time_s=s.time,input_digest=s.digest(),source=str(h[16]['folder']),checks=checks,maximum_pressure_difference_Pa=float(np.max(abs(pp-actual_p))),pressure_derivative_remainder_over12_5us_Pa=float(np.max(abs(true_pd-lin_pd))*12.5e-6),acceleration_remainder_mass_norm=float(np.sqrt(max(0,(true_a-lin_a)@z['M']@(true_a-lin_a)))),scope='one unused existing state, engineering local check, no full-space/continuum certificate',new_dynamic_steps=0))
    folder=run/'S3/continuous' if read(run/'S3/backend-decision.json')['selected'] else run/'S2/bridge';prior=read(folder/'identity.json');check(ROOT,prior['numerical_sources']);store=GenerationStore(folder,prior);store.history();rec=store.load(validator=c.validate);c.restore(rec['state'])
    write(run/'S5/selected-load-check.json',dict(status='passed_scoped',fresh_process=True,zero_steps=True,actual_backend=type(c.geometry).__name__,interface='benchmarks.research_coupled_reference_next.selected.selected_setup',source=str(folder),source_identity_sha256=sha(folder/'identity.json'),state_digest=c.state.digest(),step=c.state.step,execution_identity=ident))
    update(f'S1追加单一预登记留出状态：未参与参考构造的第16步/50微秒工程检查 {"通过" if passed else "受限"}；最大压力差{np.max(abs(pp-actual_p)):.3g}Pa。新增动态步0。新进程实际加载{type(c.geometry).__name__}第{c.state.step}步。')
    print('HOLDOUT_LOAD',passed,type(c.geometry).__name__,c.state.step,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
