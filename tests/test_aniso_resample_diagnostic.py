import unittest
import numpy as np
from pathlib import Path
from demos.aniso import Config
from benchmarks.aniso_resample_diagnostic import decompose,replay
from benchmarks.aniso_affine_consistency import DEFAULT
import json


class ResampleDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads((DEFAULT/'cases/incremental-finest/config.json').read_text())
        with np.load(DEFAULT/'cases/incremental-finest/audit-01000.npz') as z:cls.z={k:z[k].copy() for k in z.files}

    def test_real_snapshot_closure_and_actual_repeated_rebuild(self):
        r,a=decompose(self.z,self.cfg['dt'],Config(**self.cfg).params)
        self.assertLess(max(r['checks'].values()),1e-12)
        self.assertGreater(r['terms']['gradient']['F_rms'],1e-8)
        q=replay(self.z,self.cfg,a)
        self.assertLess(max(q.values()),1e-12)
        self.assertEqual(q['particle_mutation_max'],0.)
        self.assertEqual(q['repeated_F_max'],0.)

    def test_uniform_affine_update_has_no_resampling_jump(self):
        z={k:v.copy() for k,v in self.z.items()};dt=self.cfg['dt'];n=len(z['particle_F_before']);nc=len(z['coords'])
        F=np.diag([1.01,.997,1.]);L=np.array([[.02,.01,0.],[-.005,.004,0.],[0.,0.,-.002]])
        z['particle_x_after']=z['particle_x_before'].copy()
        z['particle_F_before']=np.tile(F,(n,1,1));z['particle_F_after']=np.tile((np.eye(3)+dt*L)@F,(n,1,1))
        z['particle_L_after']=np.tile(L,(n,1,1));z['center_G']=np.tile(L,(nc,1,1))
        z['center_F_resampled']=np.tile(F,(nc,1,1));z['center_F_committed']=np.tile((np.eye(3)+dt*L)@F,(nc,1,1))
        r,_=decompose(z,dt,Config(**self.cfg).params)
        self.assertLess(max(r['checks'].values()),1e-12)
        self.assertLess(r['total_F_rms'],1e-13)
        self.assertLess(r['nonlinear_stress_averaging_gap_rms_Pa'],1e-11)
        self.assertLess(abs(r['nonlinear_energy_averaging_gap_J']),1e-13)

    def test_invalid_dt_or_changed_support_is_rejected(self):
        params=Config(**self.cfg).params
        for dt in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):decompose(self.z,dt,params)
        z={k:v.copy() for k,v in self.z.items()};z['particle_x_after'][:,0]+=.125
        with self.assertRaisesRegex(ValueError,'support'):decompose(z,self.cfg['dt'],params)


if __name__=='__main__':unittest.main()
