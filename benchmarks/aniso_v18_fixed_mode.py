"""Same-displacement energy audit isolates quadrature from modal rematching."""
import numpy as np
from benchmarks.aniso_v18_runs import ROOT,OUT,load,write,sha
from benchmarks.aniso_v18_space import assemble
from benchmarks.aniso_v17_modes import controlled_case
from benchmarks.aniso_boundary_reference import hessian
from engine.aniso_phase1.carrier_joint import Geometry,gradient

def main():
    assert not (OUT/'fixed-mode-quadrature.json').exists()
    with np.load(OUT/'space-mode-targets.npz') as z:fields=z['fields'];labels=z['labels'].tolist();nodes=z['nodes']
    s,e,m,h,meta=controlled_case();g=Geometry(s,e,m,h);H=hessian('F45');records=[]
    def record(label,scheme,km,ks,mt,mc):
        assert min(km,ks,mt,mc)>0
        return dict(label=label,scheme=scheme,material_stiffness=km,stabilization_stiffness=ks,translation_inertia=mt,affine_inertia=mc,rayleigh_rad_s=float(np.sqrt((km+ks)/(mt+mc))),stabilization_fraction=float(ks/(km+ks)))
    np.testing.assert_array_equal(nodes,meta['carrier_reference'])
    for label,field in zip(labels,fields):
        grad=gradient(e.B,field).reshape(-1,9);km=float(np.einsum('pa,ab,pb,p->',grad,H,grad,e.V));ks=float(np.sum(field*(e.Ks@field)));z=g.J@field;q=g.metric
        mt=float(np.sum(q[:len(m),None]*z[:len(m)]**2));mc=float(np.sum(q[len(m):,None]*z[len(m):]**2));records.append(record(label,'sampled192',km,ks,mt,mc))
    for order in (2,3,4):
        a=assemble(h,order);np.testing.assert_allclose(a['nodes'],nodes,atol=1e-14)
        for label,field in zip(labels,fields):
            w=field[a['free']].T.ravel();ks=float(w@a['Ks']@w);km=float(w@a['K']@w)-ks
            z=a['J']@field[a['free']];np_=len(a['X']);mt=float(np.sum(z[:np_]**2));mc=float(np.sum(z[np_:]**2));records.append(record(label,f'gauss{order}',km,ks,mt,mc))
    write(OUT/'fixed-mode-quadrature.json',dict(completed=True,source_sha256=sha(ROOT/'benchmarks/aniso_v18_fixed_mode.py'),records=records,
        scope='Rayleigh quotients of identical carrier displacements at h=1/8, undeformed state. No modal rematching; not continuum natural frequencies.'))
    for r in records:print(r,flush=True)
if __name__=='__main__':main()
