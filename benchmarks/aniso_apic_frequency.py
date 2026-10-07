"""Frozen actual-kernel APIC transfer ablations, with an independent NumPy oracle."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import itertools
import json
import numpy as np
import warp as wp
from demos.aniso import Config, Scene
from engine.sp_grid import B
from engine.kernel.d3.kernel_lite import lite_c2g, lite_g2c_kernel, lite_c2p_kernel
from engine.aniso_phase1.diagnostics import particle_kinetic
from benchmarks import aniso_flip_time as prior
from benchmarks.aniso_mainline import write_json
from utils.resource_guard import prepare_warp_cache

ROOT=prior.ROOT
DEFAULT=ROOT/'docs/results/lite-aniso-mainline/v6'
SOURCE=ROOT/'docs/results/lite-aniso-mainline/v4/cases/baseline-finest/audit-01000.npz'
CORNERS=np.array(list(itertools.product((0,1),repeat=3)))


def norm(a,m):
    return float(np.sqrt(np.sum(m.reshape((-1,)+(1,)*(a.ndim-1))*a*a)/m.sum()))


class Oracle:
    def __init__(self,x,m,dx):
        self.x=x.copy();self.m=m.copy();self.dx=dx
        q=x/dx-.5;base=np.floor(q).astype(int);f=q-base
        self.c=np.unique((base[:,None,:]+CORNERS).reshape(-1,3),axis=0)
        lookup={tuple(c):i for i,c in enumerate(self.c)}
        self.S=np.zeros((len(x),len(self.c)))
        for p in range(len(x)):
            for corner in CORNERS:
                self.S[p,lookup[tuple(base[p]+corner)]]=np.prod(np.where(corner,f[p],1-f[p]))
        self.mc=self.S.T@m
        if np.any(self.mc<=0):raise ValueError('probe needs strictly occupied complete center support')
        self.W=self.S.T*m[None,:]/self.mc[:,None]
        self.xc=(self.c+.5)*dx
        self.nodes=np.unique((self.c[:,None,:]+CORNERS).reshape(-1,3),axis=0)
        lookup={tuple(n):i for i,n in enumerate(self.nodes)}
        self.H=np.zeros((len(self.c),len(self.nodes)))
        self.D=np.zeros((len(self.c),len(self.nodes),3))
        for c in range(len(self.c)):
            for corner in CORNERS:
                n=lookup[tuple(self.c[c]+corner)]
                self.H[c,n]=.125;self.D[c,n]=(2*corner-1)/(4*dx)
        self.mn=self.H.T@self.mc
        self.xn=self.nodes*dx

    def apply(self,v,G,beta):
        cv=self.W@v+np.einsum('cp,pij,cpj->ci',self.W,G,self.xc[:,None,:]-self.x[None,:,:])
        cG=np.einsum('cp,pij->cij',self.W,G)
        nv=(self.H.T@(self.mc[:,None]*cv)+np.einsum('cn,c,cij,cnj->ni',self.H,self.mc,cG,self.xn[None,:,:]-self.xc[:,None,:]))/self.mn[:,None]
        rv=self.H@nv;rG=np.einsum('ni,cnj->cij',nv,self.D)
        pic=self.S@rv;pG=np.einsum('pc,cij->pij',self.S,rG)
        return dict(p2c_v=cv,p2c_G=cG,grid_v=nv,g2c_v=rv,g2c_G=rG,pic=pic,v=beta*v+(1-beta)*pic,G=pG)


def initial_field(x,field):
    v=np.zeros_like(x);G=np.zeros((len(x),3,3))
    if field=='constant':v[:]=[.013,-.007,.003]
    elif field=='affine':
        A=np.array([[.02,.013,-.005],[-.011,.007,.004],[.003,-.008,-.009]])
        v=(x-.5)@A.T+np.array([.013,-.007,.003]);G[:]=A
    else:
        waves=2 if field=='sine_x2' else 1
        direction=np.array([1.,1.,0.])/np.sqrt(2) if field=='sine_45' else np.array([1.,0.,0.])
        k=2*np.pi*waves/.75*direction;phase=(x-.5)@k
        v=.01*np.sin(phase)[:,None]*direction
        G=.01*np.cos(phase)[:,None,None]*np.outer(direction,k)[None,:,:]
    return v,G


class Probe:
    def __init__(self,x,F):
        self.scene=Scene(Config('tensile',9,.001,45.,boundary_impulse_transfer=True),'cpu')
        self.s=self.scene.solver;s=self.s
        s.ptc_x.assign(x);s.ptc_F.assign(F);s.bc_type.zero_();s.energy_ledger=None
        s.activate_sparse_grid();self.x=x.copy();self.F=F.copy();self.m=s.ptc_m.numpy().copy()
        self.o=Oracle(x,self.m,s.dx)
        bmap=s.block2bid.numpy()
        def ids(coords):
            b=coords//B;bid=bmap[b[:,0],b[:,1],b[:,2]]
            if np.any(bid<0):raise ValueError('incomplete sparse support')
            local=coords%B
            return bid*B**3+local[:,0]*B**2+local[:,1]*B+local[:,2]
        self.cids=ids(self.o.c);self.nids=ids(self.o.nodes)

    def compact(self,a,ids):
        return a[:self.s.bcn].numpy().reshape((-1,)+a.dtype._shape_)[ids].copy()

    def step(self,beta,mode='coupled',audit=False):
        s=self.s;o=self.o;v=s.ptc_v.numpy().copy();G=s.ptc_G.numpy().copy()
        expected=o.apply(v,G,beta) if audit else None
        errors={};stages={}
        def capture(key,a,ids):
            value=self.compact(a,ids);stages[key]=value
            errors[key]=float(np.max(abs(value-expected[key])))
        s.reset_grid()
        if not s._transfer_to_centers():raise RuntimeError('history resample failed')
        if audit:
            capture('p2c_v',s.center_v,self.cids);capture('p2c_G',s.center_G,self.cids)
        lite_c2g(s.block_count,s.block2bid,s.block_xyz_by_id,s.bc_block2bid,s.bc_type,s.bc_norm,s.bc_velo,
            s.hf_bc_p,s.hf_bc_n,s.hf_bc_v,s.hf_bc_type,s.num_hf,s.center_m,s.center_v,s.center_G,s.center_vol,
            s.center_tau,s.grid_m,s.grid_v,s.grid_v_new,s.grid_size,s.center_size,0.,s.dx,0.,s.n_psi,s.device,
            explicit_force=False,enable_apic=True)
        if audit:capture('grid_v',s.grid_v,self.nids)
        wp.copy(s.grid_v_new,s.grid_v)
        wp.launch(lite_g2c_kernel,dim=(s.bcn,B,B,B),inputs=[s.block_count,s.block2bid,s.block_xyz_by_id,
            s.grid_v,s.grid_v_new,s.center_v,s.center_dv,s.center_G,s.center_size,s.dx],device='cpu')
        if audit:
            capture('g2c_v',s.center_v,self.cids);capture('g2c_G',s.center_G,self.cids)
        wp.launch(lite_c2p_kernel,dim=s.n_ptc,inputs=[s.block2bid,s.ptc_x,s.ptc_v,s.ptc_k,s.ptc_F,s.ptc_G,
            s.ptc_dlogJ,s.center_m,s.center_v,s.center_dv,s.center_G,s.psi_params,s.center_size,s.dx,0.,beta],device='cpu')
        candidate_v=s.ptc_v.numpy().copy();candidate_G=s.ptc_G.numpy().copy()
        if audit:
            errors['v']=float(np.max(abs(candidate_v-expected['v'])))
            errors['G']=float(np.max(abs(candidate_G-expected['G'])))
        # Experimental holds are applied AFTER measuring the untouched actual kernel result.
        if mode=='velocity_only':s.ptc_G.assign(G)
        elif mode=='gradient_only':s.ptc_v.assign(v)
        elif mode!='coupled':raise ValueError(mode)
        record=dict(candidate_momentum_error=float(np.max(abs(self.m@(candidate_v-v)))))
        if audit:
            p=self.m@v
            record.update(oracle_errors=errors,
                stage_momentum_error=max(float(np.max(abs(m@stages[k]-p))) for m,k in
                    [(o.mc,'p2c_v'),(o.mn,'grid_v'),(o.mc,'g2c_v')]),
                center_roundtrip_velocity_rms=norm(stages['g2c_v']-stages['p2c_v'],o.mc),
                center_roundtrip_gradient_rms=norm(stages['g2c_G']-stages['p2c_G'],o.mc),
                particle_pic_gap_rms=norm(expected['pic']-v,self.m),
                candidate_gradient_increment_rms=norm(candidate_G-G,self.m))
        return record


def run_case(probe,field,mode,beta,steps):
    s=probe.s;m=probe.m;v0,G0=initial_field(probe.x,field)
    s.ptc_v.assign(v0);s.ptc_G.assign(G0)
    p0=m@v0;vmean=p0/m.sum();nv=norm(v0-vmean,m);ng=norm(G0,m)
    k0=sum(particle_kinetic(probe.x,v0,G0,m,s.dx,True));rows=[];states={}
    for step in range(1,steps+1):
        row=probe.step(beta,mode,audit=step in (1,20,40,80))
        v=s.ptc_v.numpy().copy();G=s.ptc_G.numpy().copy()
        row.update(step=step,velocity_norm_gain=norm(v-(m@v)/m.sum(),m)/nv if nv>1e-14 else 1.,
            velocity_relative_change=norm(v-v0,m)/max(norm(v0,m),1e-30),
            gradient_norm_gain=norm(G,m)/ng if ng>1e-14 else 1.,
            gradient_relative_change=norm(G-G0,m)/max(ng,1e-30) if ng>1e-14 else norm(G-G0,m),
            kinetic_gain=sum(particle_kinetic(probe.x,v,G,m,s.dx,True))/k0,
            momentum_error=float(np.max(abs(m@v-p0))),
            x_error=float(np.max(abs(s.ptc_x.numpy()-probe.x))),F_error=float(np.max(abs(s.ptc_F.numpy()-probe.F))),
            held_error=float(np.max(abs(G-G0))) if mode=='velocity_only' else float(np.max(abs(v-v0))) if mode=='gradient_only' else 0.)
        rows.append(row)
        if step in (1,20,40,80):states['v_'+str(step)]=v;states['G_'+str(step)]=G
    return rows,states


def cases():
    result={}
    for field in ('sine_x1','sine_x2','sine_45'):
        for mode in ('coupled','velocity_only','gradient_only'):
            schedules=[('fixed',.9,80)]
            if mode!='gradient_only':schedules += [('calibrated_fine',prior.scaled_flip(.0005),40),('calibrated_finest',prior.scaled_flip(.00025),80)]
            if mode=='coupled':schedules += [('pure_flip',1.,80)]
            for label,beta,steps in schedules:result[f'{field}-{mode}-{label}']=dict(field=field,mode=mode,beta=beta,steps=steps)
    for field in ('constant','affine'):result[field+'-control']=dict(field=field,mode='coupled',beta=.9,steps=80)
    return result


def hashes():
    h=prior.hashes()
    for f in ('benchmarks/aniso_apic_frequency.py','tests/test_aniso_apic_frequency.py'):
        h[f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    return h


def references():
    count=0
    for version in range(1,6):
        path=ROOT/f'docs/results/lite-aniso-mainline/v{version}/artifact-sha256.json'
        manifest=json.loads(path.read_text())
        for name,digest in manifest.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('changed reference '+name)
            count+=1
    return count


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('freeze','run'))
    parser.add_argument('--output',type=Path,default=DEFAULT);args=parser.parse_args();out=args.output
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache');out.mkdir(parents=True,exist_ok=True)
    if args.action=='freeze':
        if (out/'protocol.json').exists():raise RuntimeError('preserve existing protocol')
        write_json(out/'protocol.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),source_sha256=hashes(),
            input_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),input=str(SOURCE.relative_to(ROOT)),
            references_verified=references(),cases=cases(),physical_interval_label=.02,counts=[20,40,80],
            nominal_dt=[.001,.0005,.00025],kernel_dt=0.,device='cpu',precision='float64',
            fixed='x,F,mass,volume,sparse support; no force, boundary, advection, or Newton solve',
            ablation='hold original G for velocity_only; hold original v for gradient_only; keep APIC enabled',
            acceptance=dict(oracle_max=1e-12,momentum_max=1e-14,frozen_state_max=0.,held_state_max=0.,affine_max=1e-11),
            scope='frozen manufactured fields on deformed F45 geometry; no new full-loading accuracy claim'))
        return
    p=json.loads((out/'protocol.json').read_text())
    if hashes()!=p['source_sha256']:raise RuntimeError('frozen source changed')
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=p['input_sha256']:raise RuntimeError('input changed')
    references()
    if (out/'cases').exists():raise RuntimeError('preserve existing run')
    (out/'cases').mkdir()
    with np.load(SOURCE) as z:probe=Probe(z['particle_x_before'],z['particle_F_before'])
    summary={}
    for name,c in p['cases'].items():
        rows,states=run_case(probe,**c)
        write_json(out/'cases'/f'{name}.json',rows)
        np.savez_compressed(out/'cases'/f'{name}.npz',**states)
        summary[name]={str(r['step']):r for r in rows if r['step'] in (1,20,40,80)}
        print(name,'complete',flush=True)
    allrows=[r for records in summary.values() for r in records.values()]
    checks=dict(oracle_max=max(max(r['oracle_errors'].values()) for r in allrows),
        momentum_max=max(max(r['momentum_error'],r['candidate_momentum_error'],r['stage_momentum_error']) for r in allrows),
        frozen_state_max=max(max(r['x_error'],r['F_error']) for r in allrows),held_state_max=max(r['held_error'] for r in allrows),
        affine_max=max(max(r['velocity_relative_change'],r['gradient_relative_change']) for name,records in summary.items() if name.endswith('-control') for r in records.values()))
    checks['passed']=all(checks[k]<=v for k,v in p['acceptance'].items())
    # Reproduce the earlier frozen probe, and show pure FLIP leaves the G-only recurrence intact.
    old=json.loads((ROOT/'docs/results/lite-aniso-mainline/v4/frozen-velocity-results.json').read_text())
    anchors=[];gradient_anchors=[]
    for r in old:
        label=r['mode'];count=r['transfers']
        variant={'fixed':'fixed','pure_flip':'pure_flip','calibrated':'calibrated_fine' if count==40 else 'calibrated_finest'}[label]
        row=summary['sine_x1-coupled-'+variant][str(count)]
        anchors.append(abs(row['velocity_norm_gain']-r['nonuniform_velocity_norm_gain']))
    for field in ('sine_x1','sine_x2','sine_45'):
        with np.load(out/'cases'/f'{field}-coupled-pure_flip.npz') as a,np.load(out/'cases'/f'{field}-gradient_only-fixed.npz') as b:
            gradient_anchors.extend(float(np.max(abs(a[k]-b[k]))) for k in a.files)
    checks['v4_anchor_max']=max(anchors);checks['pure_flip_gradient_only_anchor_max']=max(gradient_anchors)
    checks['passed']=checks['passed'] and max(anchors)<1e-12 and max(gradient_anchors)==0.
    write_json(out/'summary.json',dict(completed=True,checks=checks,cases=summary,production_changed=False,full_loading_retested=False))
    print(json.dumps(checks,indent=2));references()
    raise SystemExit(0 if checks['passed'] else 2)


if __name__=='__main__':main()
