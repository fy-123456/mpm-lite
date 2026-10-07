"""Real parent value-transaction interoperability, without a C dynamics claim."""
import copy
import json
import numpy as np
from engine.aniso_phase1.research_d.common_state import CommonState, StateTransaction
from engine.aniso_phase1.research_e.stage2.transaction import ChildAdapter, digest_value
from engine.aniso_phase1.research_e.stage2 import handoff
from .experiments import make_model
from benchmarks.research_e.experiments import bc_all


def transaction_audit(out,protocol,parent_folder):
    b=make_model(4);adapter=ChildAdapter(b,protocol['parent_bundle_sha256'])
    with np.load(parent_folder/'initial-state.npz',allow_pickle=False) as z:
        initial=CommonState(z['q'].copy(),z['velocity'].copy(),float(z['time']),int(z['step']))
    initial.child_states['E']=adapter.initial()
    tx=StateTransaction(initial,adapter.validate_parent)
    bc=bc_all(2);load=b.solid.traction(0,1,[-.1,0]);kwargs=dict(dt=.01,load=load,boundary=bc)
    def candidate(transaction,**changes):
        token=transaction.begin_trial()
        token.state.child_states['E']=adapter.prepare(token.state.child_states['E'],**dict(kwargs,**changes))
        token.state.step+=1;token.state.time+=.01
        return token
    rows=[]
    def expect_unchanged(name,fn):
        before=tx.snapshot().digest();revision=tx.revision
        try:fn();rejected=False
        except (ValueError,RuntimeError,KeyError):rejected=True
        rows.append(dict(case=name,rejected=rejected,unchanged=before==tx.snapshot().digest() and revision==tx.revision))
    expect_unchanged('solver_exception_invalid_dt',lambda:adapter.prepare(tx.snapshot().child_states['E'],**dict(kwargs,dt=-.01)))
    expect_unchanged('unconverged_fixed_stress',lambda:adapter.prepare(tx.snapshot().child_states['E'],method='once',**kwargs))
    bad=candidate(tx);bad.state.child_states['E']['p'][0]=float('nan')
    expect_unchanged('validation_exception',lambda:tx.commit(bad))
    expect_unchanged('failed_token_reuse',lambda:tx.commit(bad))
    bad=candidate(tx);bad.state.step+=1
    expect_unchanged('parent_rejection',lambda:tx.commit(bad))
    token=candidate(tx);before=tx.snapshot().digest();tx.rollback(token)
    rows.append(dict(case='rollback',rejected=True,unchanged=tx.snapshot().digest()==before))
    expect_unchanged('rolled_back_token',lambda:tx.commit(token))
    one=candidate(tx);sibling=candidate(tx);tx.commit(one)
    expect_unchanged('stale_sibling',lambda:tx.commit(sibling));expect_unchanged('double_commit',lambda:tx.commit(one))
    external={'count':0}
    def callback_failure(state):
        adapter.validate_parent(state)
        if state.step:raise RuntimeError('external validator failed before any publication')
    other=StateTransaction(initial,callback_failure);before=other.snapshot().digest();token=candidate(other)
    try:other.commit(token);rejected=False
    except RuntimeError:rejected=True
    rows.append(dict(case='external_callback_error',rejected=rejected,unchanged=before==other.snapshot().digest() and external['count']==0))
    checkpoint=adapter.checkpoint(tx.snapshot().child_states['E']);handoff.save(out/'E-child-checkpoint.json',checkpoint)
    restarted=tx.snapshot();restarted.child_states['E']=adapter.restore(json.loads((out/'E-child-checkpoint.json').read_text()))
    rt=StateTransaction(restarted,adapter.validate_parent)
    for _ in range(3):tx.commit(candidate(tx));rt.commit(candidate(rt))
    equal=tx.snapshot().digest()==rt.snapshot().digest()
    damaged=copy.deepcopy(checkpoint);damaged['payload']['config']['storage']=.2;damaged['sha256']=digest_value(damaged['payload'])
    expect_unchanged('restart_wrong_parameters',lambda:adapter.restore(damaged))
    escaped=tx.snapshot();escaped.child_states['E']['u'][0]=99.
    isolated=tx.snapshot().child_states['E']['u'][0]!=99.
    return dict(passed=bool(all(r['rejected'] and r['unchanged'] for r in rows) and equal and isolated and tx.revision==4),
        cases=rows,restart_identical=equal,owned_snapshots=isolated,successful_commits=tx.revision,
        parent_q_shape=list(initial.q.shape),E_u_shape=[b.solid.ndof],parent_A_displacement_used_as_E=False,
        status='actual frozen StateTransaction + E owned child values tested; no C stage2 time integrator or dynamic coupling certified')


def handoff_audit(out,protocol,extension_sha,protocol_sha,evidence):
    b=make_model(4);old=b.initial();load=b.solid.traction(0,1,[-.1,0]);bc=bc_all(2)
    m,signature=handoff.export(out/'handoff',b,old,.01,load,bc,protocol['parent_bundle_sha256'],extension_sha,protocol_sha,evidence)
    expected={k:m[k] for k in ('parent_bundle_sha256','space_sha256','boundary_sha256','dt','units','extension_source_sha256','protocol_sha256')}
    _,matrix,v,residual=handoff.load(out/'handoff',expected_manifest_sha256=signature,expected=expected)
    rejection={}
    for k,value in [('space_sha256','0'*64),('parent_bundle_sha256','0'*64),('boundary_sha256','0'*64),('dt',.02),('units',{'u':'cm'})]:
        bad=dict(expected);bad[k]=value
        try:handoff.load(out/'handoff',expected_manifest_sha256=signature,expected=bad);rejection[k]=False
        except ValueError:rejection[k]=True
    # Wrong SPD declaration is tested in a separate copy, preserving original package.
    import tempfile,shutil
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        folder=__import__('pathlib').Path(tmp)/'wrong';shutil.copytree(out/'handoff',folder)
        bad=copy.deepcopy(m);bad['positive_definite']=True;handoff.save(folder/'operator-package.json',bad)
        try:handoff.load(folder,expected_manifest_sha256=handoff.sha(folder/'operator-package.json'),expected=expected);rejection['false_SPD']=False
        except ValueError:rejection['false_SPD']=True
    return dict(passed=bool(residual['maximum']<=1e-8 and all(rejection.values())),manifest_sha256=signature,
        expected=expected,true_residual_blocks=residual,rejection_controls=rejection,
        D_public_consumer='see separately bound D-contract-consumer.json; original three-block package retained and D two-block interface uses exact flux elimination',
        raw_three_block_D_contract=False,private_consumer_certified=True)
