import unittest
import numpy as np
import scipy.linalg as la
from engine.aniso_phase1.research_a.stage2.cases import CaseSpec, supports
from engine.aniso_phase1.research_a.stage2.problem import Problem
from engine.aniso_phase1.research_a.stage2.adaptive import enrich
from engine.aniso_phase1.high_order_space import BoxElastic
from benchmarks.research_a.stage2.protocol import matrix


class AdaptiveTests(unittest.TestCase):
    def test_small_actual_six_round_curve_and_exact_quadratic_tangent(self):
        case=CaseSpec(angle=30.,name='construction-angle30');p=Problem.from_archive(case,2,'construction')
        events=[]
        result=enrich(p,supports(case,'v22-overlap'),rounds=6,patches_per_round=4,budget=72,
                      callback=lambda r,f,raw,t:events.append(r))
        self.assertEqual([r['scalar_local_dofs'] for r in events],[12,24,36,48,60,72])
        self.assertTrue(np.all(np.diff([r['equilibrium']['energy_J'] for r in events])<=1e-10))
        np.testing.assert_allclose(result['raw']@result['transform'],result['W'],rtol=1e-6,atol=1e-8)
        field,r,K,g,coef=p.equilibrate(result['W'],return_matrix=True)
        rng=np.random.default_rng(220930);v=rng.normal(size=coef.shape);v/=la.norm(v)
        op=BoxElastic(p.edges,p.degree,case.H)
        u=p.A@(p.Q@v[:p.Q.shape[1]])+result['W']@v[p.Q.shape[1]:]
        nodal=op.apply(u.T.ravel()).reshape(-1,p.n).T
        expected=np.vstack((p.Q.T@(p.A.T@nodal+p.Ks@(p.Q@v[:p.Q.shape[1]])),result['W'].T@nodal))
        np.testing.assert_allclose((K@v.T.ravel()).reshape(3,-1).T,expected,rtol=2e-6,atol=1e-8)
        self.assertEqual(r['mass_included'],False)

    def test_case_matrix_preserves_all_eight_and_two_modes(self):
        specs=matrix();self.assertEqual([s['budget'] for s in specs[:3]],[72,144,288])
        hidden={s['case'] for s in specs if s['case']!='F45'};self.assertEqual(len(hidden),8)
        self.assertEqual(sum(s['mode']=='reuse-F45' for s in specs),7)
        self.assertEqual(len({s['name'] for s in specs}),len(specs))

if __name__=='__main__':unittest.main()
