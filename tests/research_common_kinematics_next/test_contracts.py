import copy,tempfile,unittest
from pathlib import Path
import numpy as np
from engine.aniso_phase1.research_common_kinematics_next.mapping import basis,push
from engine.aniso_phase1.research_common_kinematics_next.model import CommonBridge,CommonOperator,Space
from engine.aniso_phase1.research_common_kinematics_next.checkpoint import save,load,config
from engine.aniso_phase1.research_eulerian_poro_next.materials import make_operator,validate_moments

class Contracts(unittest.TestCase):
    def test_composed_nonaffine_map(self):
        s=Space();rng=np.random.default_rng(24)
        X=rng.uniform([.06,.02,.02],[.94,.23,.23],(20,3));X[:,0]=np.where(abs(X[:,0]-.5)<.03,X[:,0]+.06,X[:,0])
        increments=[rng.normal(size=(10,3))*.003 for _ in range(3)]
        def evolve(X):
            x=X.copy();F=np.tile(np.eye(3),(len(X),1,1))
            for d in increments:x,F=push(s,X,x,F,d)
            return x,F
        x,F=evolve(X)
        for eps in (1e-5,1e-6,1e-7):
            fd=np.zeros_like(F)
            for a in range(3):
                delta=np.eye(3)[a]*eps;fd[:,:,a]=(evolve(X+delta)[0]-evolve(X-delta)[0])/(2*eps)
            self.assertLess(np.max(abs(fd-F)),2e-7)
        # Exact clamping for carrier + material bubbles at x=0.
        Q=X.copy();Q[:,0]=0.;N,D=basis(s,Q,Q,np.tile(np.eye(3),(len(Q),1,1)))
        self.assertLess(abs(N).max(),1e-14)

    def test_energy_volume_and_full_mass(self):
        b=CommonBridge();op,v,_=b.prepare();rng=np.random.default_rng(25)
        d=rng.normal(size=v.shape)*1e-4;z=rng.normal(size=v.shape)*.1
        _,force,_=op.material(d);analytic=float(np.sum(force*z));G=op.geometry(d,False)[1]
        for eps in (1e-4,1e-5,1e-6):
            numerical=(op.material(d+eps*z)[0]-op.material(d-eps*z)[0])/(2*eps)
            self.assertLess(abs(numerical-analytic),1e-6)
            fd=(op.geometry(d+eps*z,False)[0]-op.geometry(d-eps*z,False)[0])/(2*eps)
            np.testing.assert_allclose(fd,np.einsum('cni,ni->c',G,z),atol=1e-8)
        V0=op.geometry(np.zeros_like(d),False)[0];V1=op.geometry(d,False)[0]
        np.testing.assert_allclose(V1-V0,np.einsum('cni,ni->c',op.discrete_G(np.zeros_like(d),d),d),atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(op.M).min(),0.)
        self.assertGreater(np.linalg.norm(op.M[:8,8:]),1e-4)

    def test_frozen_operator_isolation_and_roundtrip(self):
        b=CommonBridge();op,v,_=b.prepare();d=np.full_like(v,1e-4);energy=op.material(d)[0]
        b.state.particles.X[:]=99;b.state.particles.A[:]=0;b.state.qF[:]=0
        self.assertEqual(op.material(d)[0],energy)
        with tempfile.TemporaryDirectory() as temp:
            f=Path(temp)/'frozen.npz';op.save(f);other=CommonOperator.load_package(f)
            self.assertEqual(op.identity,other.identity);self.assertEqual(energy,other.material(d)[0])
        self.assertFalse(op.F0.flags.writeable);self.assertFalse(op.D.flags.writeable)

    def test_failure_rolls_back_whole_state(self):
        b=CommonBridge();digest=b.state.digest()
        for stage in ('prepare','solve','load','commit'):
            with self.assertRaisesRegex(RuntimeError,'injected'):b.step(fault=stage)
            self.assertEqual(b.state.digest(),digest)
        reference=CommonBridge();reference.step();b.step()
        self.assertEqual(b.state.digest(),reference.state.digest())

    def test_checkpoint_and_foreign_identity(self):
        b=CommonBridge();b.step()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.npz';save(b,path);c=load(path,config(b))
            self.assertEqual(c.state.digest(),b.state.digest());b.step();c.step()
            self.assertEqual(c.state.digest(),b.state.digest())
            foreign=config(b);foreign['storage']=.0002
            with self.assertRaisesRegex(ValueError,'foreign'):load(path,foreign)

    def test_direction_distribution_and_fallback_limits(self):
        b=CommonBridge();p=b.state.particles;op,info=make_operator(b.space,p)
        # Two equally weighted, orthogonal directions require a true A4.
        a=np.diag([1.,0.,0.]);c=np.diag([0.,1.,0.]);A2=(a+c)/2
        A4=(np.outer(a.ravel(),a.ravel())+np.outer(c.ravel(),c.ravel()))/2
        validate_moments(A2[None],A4[None])
        F=np.diag([1.1,.9,1.]);delta=(F.T@F-np.eye(3)).ravel()
        correct=delta@A4@delta;wrong=delta@np.outer(A2.ravel(),A2.ravel())@delta
        self.assertGreater(abs(correct-wrong),.01)
        with self.assertRaisesRegex(ValueError,'rank-one'):
            q=p.copy();q.A[:]=A2;make_operator(b.space,q)
        p.A[:]=np.where((p.X[:,0]<.5)[:,None,None],a,c)
        _,info=make_operator(b.space,p,'director-or-positive')
        self.assertTrue(info['fallback'])
        p.X[0]+=[.001,0,0]
        with self.assertRaises(ValueError):make_operator(b.space,p,'director-or-positive')

    def test_zero_equilibrium_and_closed_fluid(self):
        from engine.aniso_phase1.research_unified_lite_poro.solve import advance
        b=CommonBridge();op,v,_=b.prepare();zero=np.zeros_like(v)
        d,v1,p,z,m=advance(op,zero,zero,np.zeros(2),.00125,0.,amplitude=0.)
        np.testing.assert_array_equal(d,zero);np.testing.assert_array_equal(v1,zero)
        base=op.top
        class Closed:
            cells=base.cells;V0=base.V0;B=base.B[:,base.internal]
            def assemble(self,*args):
                H,J=base.assemble(*args);ids=base.internal
                return H[np.ix_(ids,ids)],J
        op.top=Closed();op.gb=np.zeros(len(base.internal))
        p0=np.array([.2,.1]);V0=op.geometry(zero,False)[0]
        d,v1,p,z,m=advance(op,zero,zero,p0,.00125,0.,amplitude=0.)
        change=op.capacity*(p-p0)+op.alpha*(op.geometry(d,False)[0]-V0)
        self.assertLess(abs(change.sum()),1e-10);self.assertGreaterEqual(m['darcy_dissipation'],0.)
        self.assertEqual(float(np.sum(op.top.B@z)),0.)

    def test_reject_bad_history_and_mass(self):
        b=CommonBridge();b.state.particles.F[0]*=-1
        with self.assertRaises(ValueError):b.prepare()
        b=CommonBridge();b.state.particles.weight[0]=-1
        with self.assertRaises(ValueError):b.prepare()

if __name__=='__main__':unittest.main()
