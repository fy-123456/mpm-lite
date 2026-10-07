"""APIC C/L separation and incremental affine transfer: frozen/full F45 acceptance."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import argparse
import hashlib
import json
import os
import subprocess
import sys
import numpy as np
import warp as wp
from benchmarks import aniso_mainline as base
from benchmarks import aniso_f45_refinement as refine
from benchmarks import aniso_flip_time as flip
from benchmarks import aniso_apic_frequency as frequency
from benchmarks.aniso_gradient_control import GradientControlLedger
from demos.aniso import Scene,Config
from engine.aniso_phase1.affine_transfer import prepare_incremental,finish_transfer
from engine.aniso_phase1.tensile import grid_values
from engine.aniso_phase1.transfer_audit import tensor_rms
from engine.kernel.d3.kernel_lite import lite_c2g,lite_g2c_kernel,lite_c2p_kernel
from engine.sp_grid import B
from engine.types import vec3
from utils.resource_guard import prepare_warp_cache,inspect_storage

ROOT=base.ROOT;DEFAULT=ROOT/'docs/results/lite-aniso-mainline/v7'
TESTS=flip.TESTS+['tests.test_aniso_apic_frequency','tests.test_aniso_affine_transfer']


def hashes():
    h=frequency.hashes()
    for f in ('benchmarks/aniso_affine_consistency.py','tests/test_aniso_affine_transfer.py'):
        h[f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    return h


def references():
    count=frequency.references()
    for name,h in json.loads((frequency.DEFAULT/'artifact-sha256.json').read_text()).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise RuntimeError('changed reference '+name)
        count+=1
    return count


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve existing protocol')
    old=json.loads((flip.DEFAULT/'protocol.json').read_text());configs={}
    for level in refine.LEVELS:
        c=old['configs']['calibrated-'+level].copy();c['apic_transfer']='incremental';configs['incremental-'+level]=c
    c=old['configs']['calibrated-coarse'].copy();c['apic_transfer']='overwrite';configs['anchor-coarse']=c
    base.write_json(out/'protocol.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),source_sha256=hashes(),
        configs=configs,device='cpu',precision='float64',duration=.5,endpoint_displacement_m=.005,
        beta_rule='0.9**(dt/0.001)',snapshot_times=[.05,.1,.25,.5],tests=TESTS,
        run_count=4,total_steps=4000,reference_files_verified=references(),
        acceptance=dict(relative_force_max=.05,force_floor_N=.001,interval=[.05,.5],
                        decreasing_adjacent_absolute_difference=True,force_error_N=1e-7,momentum_gap=1e-14,
                        oracle_max=1e-12,anchor_max=1e-12,all_tests_no_skips=True),
        cuda={'status':'not_run','reason':'CPU float64 controlled experiment'},
        resource={'max_cpu_processes':2,'storage':asdict(inspect_storage())},
        frozen_probe=dict(fields=['sine_x1','sine_x2','sine_45'],counts=[20,40,80],
                          beta=[flip.scaled_flip(t) for t in refine.LEVELS.values()],kernel_dt=0.,
                          purpose='test complete C/v update frequency; F and x frozen; not a loading solution'),
        scope='only APIC C return changes; physical L, F update, center force/tangent, position and velocity formulas retain original form'))


def verify(p):
    if hashes()!=p['source_sha256']:raise RuntimeError('frozen source changed')


class AffineLedger(GradientControlLedger):
    def begin(self,s):
        super().begin(s)
        if self.audit_next:
            self.old_C=s.ptc_C.numpy().copy();self.old_v=s.ptc_v.numpy().copy()

    def finish(self,s):
        super().finish(s)
        C=s.ptc_C.numpy();L=s.ptc_L.numpy()
        error=float(np.max(abs(C-L-s.flip_ratio*s._apic_difference.numpy()))) if s.apic_transfer=='incremental' else float(np.max(abs(C-L)))
        self.rows[-1].update(apic_material_gradient_gap_rms=float(np.sqrt(np.mean(np.sum((C-L)**2,axis=(1,2))))),
                             apic_increment_identity_max=error)
        if self.audit_next:
            self.audit.update(apic_transfer=s.apic_transfer,apic_increment_identity_max=error)
            self.snapshot.update(particle_C_before=self.old_C,particle_velocity_before=self.old_v,
                particle_C_after=C.copy(),particle_L_after=L.copy(),particle_velocity_after=s.ptc_v.numpy().copy(),
                grid_nodes=self.nodes.copy(),grid_mass=self.node_m.copy(),grid_velocity_raw=self.raw_v.copy(),
                grid_velocity_new=grid_values(s,s.grid_v_new,self.nodes),particle_mass=s.ptc_m.numpy().copy())


def worker(out,p,name):
    dest=out/'cases'/name
    class AuditedScene(Scene):
        def __init__(self,config,device):
            super().__init__(config,device);self.solver.energy_ledger=AffineLedger('baseline')
            self.audit_steps={round(t/config.dt) for t in p['snapshot_times']}
        def step(self):
            ledger=self.solver.energy_ledger;ledger.audit_next=self.solver.sim_steps+1 in self.audit_steps
            ok=super().step()
            if ok and ledger.audit_next:
                step=self.solver.sim_steps
                np.savez_compressed(dest/f'audit-{step:05d}.npz',**ledger.snapshot)
                base.write_json(dest/f'audit-{step:05d}.json',ledger.audit)
            return ok
    base.Scene=AuditedScene
    return base.run_case(out,name,p['configs'][name],p)


def run(out,p,jobs):
    if not json.loads((out/'tests.json').read_text())['required_checks_passed']:raise RuntimeError('tests must pass')
    if not json.loads((out/'frozen-summary.json').read_text())['passed']:raise RuntimeError('frozen probe must pass')
    references()
    def launch(name):
        log=out/(name+'.log')
        if log.exists() or (out/'cases'/name).exists():raise RuntimeError('preserve '+name)
        with log.open('w') as f:
            r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_affine_consistency','worker','--output',str(out),'--case',name],
                cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        print(name,'exit',r.returncode,flush=True);return dict(case=name,exit_code=r.returncode)
    # Anchor is independent; enqueue finest early to balance the two worker durations.
    names=['incremental-finest','anchor-coarse','incremental-coarse','incremental-fine']
    with ThreadPoolExecutor(max_workers=jobs) as pool:records=list(pool.map(launch,names))
    base.write_json(out/'batch.json',records)
    return all(r['exit_code']==0 for r in records)


def frozen(out,p):
    if (out/'frozen-summary.json').exists():raise RuntimeError('preserve frozen probes')
    with np.load(frequency.SOURCE) as z:probe=frequency.Probe(z['particle_x_before'],z['particle_F_before'])
    s=probe.s;s.apic_transfer='incremental';s.grid_v_raw=wp.zeros((s.bcn,B,B,B),dtype=vec3,device='cpu')
    results=[];worst=0.;momentum=0.;states={}
    for field in p['frozen_probe']['fields']:
        for count,beta in zip(p['frozen_probe']['counts'],p['frozen_probe']['beta']):
            v0,C0=frequency.initial_field(probe.x,field);vref=v0.copy();Cref=C0.copy()
            s.flip_ratio=beta
            s.ptc_v.assign(v0);s.ptc_C.assign(C0)
            for step in range(count):
                oracle=probe.o.apply(vref,Cref,0.)
                vref=beta*vref+(1-beta)*oracle['v'];Cref=beta*Cref+(1-beta)*oracle['G']
                s.reset_grid();assert s._transfer_to_centers()
                lite_c2g(s.block_count,s.block2bid,s.block_xyz_by_id,s.bc_block2bid,s.bc_type,s.bc_norm,s.bc_velo,
                    s.hf_bc_p,s.hf_bc_n,s.hf_bc_v,s.hf_bc_type,s.num_hf,s.center_m,s.center_v,s.center_G,s.center_vol,
                    s.center_tau,s.grid_m,s.grid_v,s.grid_v_new,s.grid_size,s.center_size,0.,s.dx,0.,s.n_psi,s.device,
                    explicit_force=False,enable_apic=True,grid_v_raw=s.grid_v_raw)
                wp.copy(s.grid_v_new,s.grid_v);prepare_incremental(s)
                wp.launch(lite_g2c_kernel,dim=(s.bcn,B,B,B),inputs=[s.block_count,s.block2bid,s.block_xyz_by_id,s.grid_v_raw,s.grid_v_new,
                    s.center_v,s.center_dv,s.center_G,s.center_size,s.dx],device='cpu')
                wp.launch(lite_c2p_kernel,dim=s.n_ptc,inputs=[s.block2bid,s.ptc_x,s.ptc_v,s.ptc_k,s.ptc_F,s.ptc_G,s.ptc_dlogJ,
                    s.center_m,s.center_v,s.center_dv,s.center_G,s.psi_params,s.center_size,s.dx,0.,beta],device='cpu')
                finish_transfer(s)
                v=s.ptc_v.numpy();C=s.ptc_C.numpy()
                worst=max(worst,float(np.max(abs(v-vref))),float(np.max(abs(C-Cref))),float(np.max(abs(s.ptc_L.numpy()-oracle['G']))))
                momentum=max(momentum,float(np.max(abs(probe.m@(v-v0)))))
                assert np.array_equal(s.ptc_x.numpy(),probe.x) and np.array_equal(s.ptc_F.numpy(),probe.F)
            mean=probe.m@v0/probe.m.sum()
            r=dict(field=field,count=count,beta=beta,velocity_norm_gain=frequency.norm(v-mean,probe.m)/frequency.norm(v0-mean,probe.m),
                affine_norm_gain=frequency.norm(C,probe.m)/frequency.norm(C0,probe.m),
                material_gradient_norm_gain=frequency.norm(s.ptc_L.numpy(),probe.m)/frequency.norm(C0,probe.m))
            results.append(r);states[field+'_'+str(count)+'_v']=v.copy();states[field+'_'+str(count)+'_C']=C.copy()
            print('frozen',field,count,'complete',flush=True)
    pairs={}
    for field in p['frozen_probe']['fields']:
        pairs[field]={}
        for key in ('v','C'):
            vals=[states[f'{field}_{n}_{key}'] for n in (20,40,80)]
            pairs[field][key]=dict(coarse_fine=frequency.norm(vals[1]-vals[0],probe.m),fine_finest=frequency.norm(vals[2]-vals[1],probe.m))
    passed=worst<=1e-12 and momentum<=1e-14
    base.write_json(out/'frozen-summary.json',dict(passed=passed,oracle_max=worst,momentum_max=momentum,rows=results,adjacent=pairs,passes=420))
    np.savez_compressed(out/'frozen-states.npz',**states)
    return passed


def analyze(out,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data,checks,audits=refine.read_cases(out,p['configs'])
    ref,_,_=refine.read_cases(flip.DEFAULT,['calibrated-'+l for l in refine.LEVELS])
    old={'overwrite-'+k.split('-')[-1]:v for k,v in ref.items()}
    trends=dict(overwrite=refine.refinement(old,'overwrite'),incremental=refine.refinement(data,'incremental'))
    anchor={}
    if 'anchor-coarse' in data:
        anchor['reaction_max']=max(abs(a['right_force']-b['right_force']) for a,b in zip(data['anchor-coarse'],old['overwrite-coarse']))
        with np.load(out/'cases/anchor-coarse/frames.npz') as a,np.load(flip.DEFAULT/'cases/calibrated-coarse/frames.npz') as b:
            for key in ('x','F'):anchor[key+'_max']=float(np.max(abs(a[key]-b[key])))
        anchor['passed']=max(anchor.values())<=1e-12
    diagnostics=len(audits)==4 and all(len(rows)==4 and all(r['frozen_decomposition_max_error']<=1e-12 and r['apic_increment_identity_max']<=1e-12 and r['pic_advection_oracle_max_error']<=1e-12 for r in rows) for rows in audits.values())
    complete=len(data)==4;physical=complete and all(c['passed'] for c in checks.values())
    tr=trends['incremental'];time_ok=bool(tr['complete'] and tr['components']['right_force']['passes_5_percent'] and tr['components']['right_force']['decreasing'])
    summary=dict(experiment_completed=complete,checks=checks,refinement=trends,anchor=anchor,audits=audits,
        diagnostics_passed=diagnostics,physical_checks_passed=physical,time_acceptance_passed=time_ok,
        required_checks_passed=json.loads((out/'tests.json').read_text())['required_checks_passed'],
        production_default_changed=False,cuda='not_run',spatial_accuracy='not_verified')
    summary['all_acceptance_passed']=bool(complete and physical and time_ok and diagnostics and anchor.get('passed') and summary['required_checks_passed'])
    base.write_json(out/'summary.json',summary)
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for i,(mode,group) in enumerate((('overwrite',old),('incremental',data))):
        for level in refine.LEVELS:
            rows=group.get(mode+'-'+level,[])
            for j,key in enumerate(('right_force','right_elastic_force','right_inertial_force')):
                axes[i,j].plot([r['displacement'] for r in rows],[r[key] for r in rows],label=level)
        for j,title in enumerate(('total','elastic','inertial')):
            axes[i,j].set(title=mode+' / '+title,xlabel='command displacement (m)',ylabel='reaction (N)');axes[i,j].grid(alpha=.25);axes[i,j].legend()
    fig.savefig(out/'affine-reaction.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(refinement=trends,physical=physical,diagnostics=diagnostics,anchor=anchor),indent=2))
    return summary['all_acceptance_passed']


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('action',choices=('freeze','tests','frozen','run','worker','analyze'))
    parser.add_argument('--output',type=Path,default=DEFAULT);parser.add_argument('--case');parser.add_argument('--jobs',type=int,choices=(1,2),default=2)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache')
    if args.action=='freeze':freeze(out);return
    p=json.loads((out/'protocol.json').read_text());verify(p)
    if args.action=='tests':ok=base.tests(out,p)
    elif args.action=='frozen':ok=frozen(out,p)
    elif args.action=='run':ok=run(out,p,args.jobs)
    elif args.action=='worker':ok=worker(out,p,args.case)
    else:ok=analyze(out,p)
    raise SystemExit(0 if ok else 2)


if __name__=='__main__':main()
