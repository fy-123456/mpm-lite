"""Same-state remap energy: sealed v13 histories, independent formula and Warp."""
import gc,hashlib,json
import numpy as np
from demos.aniso import Config,Scene
from benchmarks.aniso_material_history import ROOT,BASE,write,load
from benchmarks.aniso_apic_frequency import Oracle
from engine.aniso_phase1.selective_patch import scalar_matrix
OUT=BASE/'v14-exploration'

def main():
    records=[];cfgs=load(BASE/'v13/protocol.json')['configs']
    for folder in sorted((BASE/'v13/cases').glob('null-*')):
        cfg=cfgs[folder.name];h=1/(cfg['grid']-1)
        for path in sorted(folder.glob('audit-*.npz')):
            with np.load(path) as f:z={k:f[k].copy() for k in f.files}
            Y=z['patch_origin']+cfg['dt']*z['native_velocity'];P=z['patch_P'];ids=z['patch_ids'];w=z['patch_weight']
            energy=lambda y:float(.5*np.sum(w[:,None,None]*np.einsum('cij,cja->cia',P,y[ids])**2))
            old=energy(Y);o=Oracle(z['particle_x_after'],z['particle_mass'],h);T=o.S@o.H
            inverse=np.linalg.inv(z['particle_F_after']);relative=o.xn[:,None,:]-z['particle_x_after'][None,:,:]
            Xparticle=z['particle_reference_x'][None,:,:]+np.einsum('pij,npj->npi',inverse,relative)
            X=np.einsum('pn,p,npi->ni',T,z['particle_mass'],Xparticle)/o.mn[:,None]
            S,_,_=scalar_matrix(o.xn,o.c,o.S.T@z['particle_volume'],h,X)
            rebuilt=float(.5*np.sum(o.xn*(S@o.xn)))
            r=dict(case=folder.name,snapshot=path.name,old_J=old,reconstructed_J=rebuilt,rebuild_jump_J=rebuilt-old,carried_J=energy(Y),carried_jump_J=energy(Y)-old)
            t=load(path.with_suffix('.json'))['time']
            if abs(t-.5)<1e-9 or abs(t-1.2)<1e-9:
                scene=Scene(Config(**{**cfg,'stabilization':'material_patch'}),'cpu');s=scene.solver
                for a,b in [('x','x'),('F','F'),('v','velocity'),('C','C')]:getattr(s,'ptc_'+a).assign(z['particle_'+b+'_after'])
                s.ptc_reference_x.assign(z['particle_reference_x']);e=s.enhancements
                e.restart_state=dict(Y=Y,X=z['patch_X'],P=P,ids=ids,weights=w)
                assert not s.step(max_iters=0,print_every=0)
                e.correction(False);E=float(e.hg_energy.numpy()[0]);r['actual_Warp_carry_error_J']=abs(E-old)
                for _ in range(3):e.resample();e.prepare();e.correction(False);assert abs(float(e.hg_energy.numpy()[0])-E)<1e-18
                assert r['actual_Warp_carry_error_J']<1e-14
                del scene,s,e;gc.collect()
            records.append(r)
    write(OUT/'same-state-rebuild.json',dict(passed=True,records=records,actual_kernel_cases=sum('actual_Warp_carry_error_J' in r for r in records),source_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),scope='same particle state plus identical incoming stabilization history; compare reconstruction with carrying history; no energy offset'))
    print('same-state records',len(records),'max old jump',max(abs(r['rebuild_jump_J']) for r in records),'max new jump',max(abs(r['carried_jump_J']) for r in records))

if __name__=='__main__':main()
