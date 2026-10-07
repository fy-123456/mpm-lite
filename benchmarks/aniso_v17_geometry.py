"""Supplement: break transverse symmetry to test small-inertia frequency scaling."""
import numpy as np
from benchmarks.aniso_v17_modes import controlled_case,ModalModel
from benchmarks.aniso_v17_time import ROOT,OUT,sha,write
from benchmarks.aniso_v17_diagnosis import common_rest_null,annotate_overlap
from engine.aniso_phase1.carrier_joint import Geometry,gradient
from engine.aniso_phase1.history_increment import material_tangent
import scipy.linalg as la


def field(X,a):
    q=(X[:,0]-.25)/.5;inside=(q>0)&(q<1);q=np.clip(q,0,1);f=np.sin(np.pi*q)**2;df=2*np.pi*np.sin(2*np.pi*q)
    y=(X[:,1]-.5)/.25;z=(X[:,2]-.5)/.25
    b=np.column_stack((y*z+.3*y*y+.2*z*z,y*y*z+.2*z**3,y*z*z+.3*y**3))
    by=np.column_stack((z+.6*y,2*y*z,z*z+.9*y*y))/.25
    bz=np.column_stack((y+.4*z,y*y+.6*z*z,2*y*z))/.25
    u=a*f[:,None]*b;C=np.stack((a*df[:,None]*b,a*f[:,None]*by,a*f[:,None]*bz),axis=2)
    u[~inside]=0;C[~inside]=0
    return u,C


def main():
    assert not (OUT/'generic-geometry-scan.json').exists()
    sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'benchmarks/aniso_v17_geometry.py',ROOT/'benchmarks/aniso_v17_modes.py',ROOT/'benchmarks/aniso_v17_diagnosis.py']}
    write(OUT/'generic-geometry-protocol-v2.json',dict(source_sha256=sources,
        purpose='The original common analytic field preserves transverse symmetries and leaves exact kinetic nulls. This additional 3D field tests symmetry-breaking geometry without changing material, mass, patch energy or BC.'))
    # Analytic gradient cross-check independent of the carrier discretization.
    rng=np.random.default_rng(1701);x=np.array([.4,.5,.5])+rng.uniform(-.02,.02,(20,3));_,C=field(x,.001)
    for k in range(3):
        dx=np.eye(3)[k]*1e-6;fd=(field(x+dx,.001)[0]-field(x-dx,.001)[0])/(2e-6)
        np.testing.assert_allclose(fd,C[:,:,k],rtol=1e-7,atol=1e-11)
    Z,_=common_rest_null();records=[]
    specs=[(h,ns,.0001) for h in (1/8,1/10,1/12) for ns in (4,6,8)]
    specs += [(1/8,4,a) for a in (.001,.0003,.00003)]
    for h,ns,a in specs:
        s,e,m,_,meta=controlled_case(h,ns,0);xp=meta['particle_reference'];X=meta['carrier_reference']
        s.x=xp+field(xp,a)[0];s.Y=X+field(X,a)[0];s.v,s.C=field(xp,.001)
        rest_s,rest_e,rest_m,_,_=controlled_case(h,ns,0)
        rest=ModalModel(rest_s,rest_e,rest_m,h,xp,X);g=Geometry(s,e,m,h);J=g.J@g.Q
        singular=la.svdvals(np.sqrt(g.metric)[:,None]*J)
        F=gradient(e.B,s.Y);direction_records=[]
        for p in rest.null_basis.T:
            w=rest.Q@p.reshape(3,rest.Q.shape[1]).T
            w=g.Q@(g.Q.T@w);w/=la.norm(w)
            mapped=g.J@w;inertia=float(np.sum(g.metric[:,None]*mapped*mapped))
            dF=gradient(e.B,w);dP=material_tangent(F,e.A,dF,e.params)
            km=float(np.sum(e.V[:,None,None]*dF*dP))
            rr=np.einsum('cij,cja->cia',e.P,w[e.ids]);ks=float(np.sum(e.weights[:,None,None]*rr*rr))
            direction_records.append(dict(effective_inertia=inertia,material_stiffness=km,stabilization_stiffness=ks,
                rayleigh_omega_rad_s=float(np.sqrt((km+ks)/inertia)) if inertia>1e-30 else None))
        r=dict(h=h,ns=ns,amplitude=a,smallest_kinetic_singular=float(singular[-1]),kinetic_singular_ratio=float(singular[-1]/singular[0]),
            numerical_null_modes={str(tol):3*int(np.sum(singular<=tol*singular[0])) for tol in (1e-14,1e-12,1e-10,1e-8)},
            reference_rest_null_directions=direction_records,
            method='Rayleigh quotients on projected original rest-null directions, evaluated directly through weighted J; these are not eigenfrequencies.')
        records.append(r);print(h,ns,a,'sigma min',singular[-1],flush=True)
    write(OUT/'generic-geometry-scan.json',dict(completed=True,records=records,field_derivative_check_passed=True,
        scope='Same controlled 3D field across grids/densities. Direct weighted-J singular values and Rayleigh quotients on former null directions; not eigenmode matching or a continuum accuracy reference.',
        rejected_attempt='The first generic full-eigenmode decomposition failed its modal energy identity in very ill-conditioned coordinates; source/log/protocol retained. Direct J norms avoid forming the normal mass matrix for this supplement.'))
    for n,d in sources.items():assert sha(ROOT/n)==d
if __name__=='__main__':main()
