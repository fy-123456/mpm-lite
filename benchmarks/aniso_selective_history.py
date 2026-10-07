"""v11 selective stabilization: frozen gates, static studies and four-step slow loads."""
import argparse,gc,hashlib,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import warp as wp
from benchmarks import aniso_mainline as base
from benchmarks import aniso_residual_history as v10
from benchmarks.aniso_boundary_reference import CASES,hessian
from benchmarks.aniso_residual_gate import geometry,matrix,hourglass_matrix,rank_gate,static_solve
from benchmarks.aniso_static_space import center_interpolation
from benchmarks.aniso_projected_history import HistoryLedger,read_case
from engine.aniso_phase1.selective_patch import scalar_matrix,polynomial
from engine.aniso_phase1.diagnostics import ParticleEnergyLedger
from engine.aniso_phase1.tensile import grid_values
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/results/lite-aniso-mainline/v11'
TESTS=v10.TESTS+['tests.test_aniso_selective_patch']
MODES=('matrix_only','selective_patch')
LEVELS={'coarse':.001,'fine':.0005,'finest':.00025,'fourth':.000125}


def hashes():
    paths=set(v10.hashes())|{'engine/aniso_phase1/selective_patch.py','benchmarks/aniso_selective_history.py','benchmarks/aniso_selective_check.py','tests/test_aniso_selective_patch.py'}
    return {f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sorted(paths)}


def freeze(out):
    if (out/'protocol.json').exists():raise RuntimeError('preserve protocol')
    old=json.loads((ROOT/'docs/results/lite-aniso-mainline/v9/protocol.json').read_text())['configs']['projected-coarse'];configs={}
    for label,(kf,angle) in CASES.items():
        for level,dt in LEVELS.items():configs[label+'-'+level]={**old,'kf':kf,'fiber_angle':angle,'dt':dt,'flip_ratio':.9**(dt/.001),'history_consistency':'residual_center','stabilization':'selective_patch'}
    base.write_json(out/'protocol.json',dict(source_sha256=hashes(),tests=TESTS,configs=configs,device='cpu',precision='float64',
        modes=MODES,static_grids=[9,17,33],beam_grids=[17,33],duration=.5,snapshot_times=[.05,.1,.25,.5],dynamic_cases=16,dynamic_steps=30000,
        candidate_gate='90 regression checks plus all 12 massless tensile and 8 beam ranks; F45 grid33 reaction gap < 5% and below v10; all solves pass',
        time_gate=dict(R_F_P_U_relative=.02,reaction_trend_min_order=.5),coefficient='mu=10 Pa; fixed 1; no lambda/fiber stiffness in patch energy',
        support='complete local 4x4x4 active-node patches; reject missing support; reference-volume weighting',
        default_changed=False,accuracy_certified=False,cuda='not_run'))


def static(out,p):
    assert json.loads((out/'tests.json').read_text())['required_checks_passed']
    dest=out/'static';dest.mkdir(exist_ok=False);records=[];beams=[]
    for label in CASES:
        X,V,Lref,Pref,reference=v10.evaluation(label);H=hessian(label)
        masks=dict(whole=np.ones(len(X),bool),near_grip=np.minimum(abs(X[:,0]-.25),abs(X[:,0]-.75))<.0625,interior=(X[:,0]>.3125)&(X[:,0]<.6875),deep_interior=(X[:,0]>.375)&(X[:,0]<.625))
        norm=lambda a,m:float(np.sqrt(np.sum(V[m,None,None]*a[m]**2)))
        for grid in p['static_grids']:
            g=geometry(grid);h=1/(grid-1);S,ids,P=scalar_matrix(g['nodes'],g['centers'],g['volume'],h)
            I=center_interpolation(X,g['centers'],h);Km=matrix(g,H,'residual_center')
            for mode in MODES:
                start=time.monotonic();Ks=hourglass_matrix(g,hessian('ISO')) if mode=='matrix_only' else sp.block_diag([S]*3,format='csr');K=Km+Ks
                gate=rank_gate(g,K);u,r=static_solve(g,K,gate['passed']);Lc=np.stack([D@u for D in g['D']],axis=2);L=(I@Lc.reshape(-1,9)).reshape(-1,3,3);Pfield=(L.reshape(-1,9)@H.T).reshape(-1,3,3)
                energy=float(.5*u.T.ravel()@(Ks@u.T.ravel()));r.update(case=label,grid=grid,mode=mode,gate=gate,reaction_reference_N=reference['reaction_N'],reaction_relative=abs(r['reaction_N']-reference['reaction_N'])/abs(reference['reaction_N']),reaction_signed_relative=(r['reaction_N']-reference['reaction_N'])/reference['reaction_N'],stabilization_energy_J=energy,stabilization_energy_fraction=energy/r['energy_J'],regions={key:dict(F_relative=norm(L-Lref,m)/norm(Lref,m),P_relative=norm(Pfield-Pref,m)/norm(Pref,m)) for key,m in masks.items()},seconds=time.monotonic()-start)
                if mode=='selective_patch':
                    residual=np.einsum('cij,cja->cia',P,u[ids]);Ec=.5*10*g['volume']/h**2/ids.shape[1]*np.sum(residual**2,axis=(1,2));xc=(g['centers']+.5)*h
                    outside=np.any((xc<v10.LO)|(xc>v10.HI),axis=1);near=np.minimum(abs(xc[:,0]-.25),abs(xc[:,0]-.75))<.0625
                    ng=g['nodes'];ghost=np.any((ng<v10.LO)|(ng>v10.HI),axis=1)
                    r['support_diagnostic']=dict(outside_center_volume_fraction=float(g['volume'][outside].sum()/g['volume'].sum()),outside_center_stabilization_fraction=float(Ec[outside].sum()/Ec.sum()),near_grip_stabilization_fraction=float(Ec[near].sum()/Ec.sum()),outside_nodes=int(ghost.sum()),total_nodes=len(ng),constant_linear_quadratic_residual_max=float(np.max(abs(P@polynomial(ng)[ids]))),stabilization_energy_independent_error=abs(float(Ec.sum())-energy))
                name=f'{label}-g{grid}-{mode}';records.append(r);base.write_json(dest/(name+'.json'),r);np.savez_compressed(dest/(name+'.npz'),u=u,nodes=g['nodes']);print(name,'rank',gate['passed'],'Rgap',r['reaction_signed_relative'],'internalP',r['regions']['interior']['P_relative'],flush=True)
            del I;gc.collect()
    for grid in p['beam_grids']:
        g=geometry(grid,True);S,_,_=scalar_matrix(g['nodes'],g['centers'],g['volume'],1/(grid-1))
        for label in CASES:
            for mode in MODES:
                K=matrix(g,hessian(label),'residual_center')+(hourglass_matrix(g,hessian('ISO')) if mode=='matrix_only' else sp.block_diag([S]*3,format='csr'))
                gate=rank_gate(g,K,True);r=dict(grid=grid,case=label,mode=mode,gate=gate)
                if gate['passed']:_,sol=static_solve(g,K,True);r.update(sol)
                beams.append(r);print('beam',grid,label,mode,gate['passed'],flush=True)
    gates={m:all(r['gate']['passed'] and r.get('solved',False) for r in records+beams if r['mode']==m) for m in MODES}
    old=json.loads((ROOT/'docs/results/lite-aniso-mainline/v10/static/F45-g33-residual_corotated.json').read_text());r=next(r for r in records if r['case']=='F45' and r['grid']==33 and r['mode']=='selective_patch')
    improved=r['reaction_relative']<min(.05,old['reaction_relative'])
    base.write_json(out/'static-summary.json',dict(completed=True,records=records,beams=beams,candidate_static_gate=gates,F45_improved=improved,loading_allowed=gates['selective_patch'] and improved,reference='same v10 references for before/after; v11 local reference assessed separately'))
    return gates['selective_patch'] and improved


class PatchLedger(HistoryLedger,ParticleEnergyLedger):
    def finish(self,s):
        super().finish(s)
        if self.audit_next:
            e=s.enhancements;self.snapshot.update(patch_ids=e.ids.numpy(),patch_P=e.P.numpy(),patch_weight=e.weights.numpy(),patch_origin=e.origin.numpy(),patch_X=e.reference_X,particle_reference_x=s.ptc_reference_x.numpy(),native_nodes=s.mapped_nodes,native_velocity=grid_values(s,s.grid_v_new,s.mapped_nodes))


def worker(out,p,name):
    v10.ResidualLedger=PatchLedger
    return v10.worker(out,p,name)


def run(out,p,jobs):
    assert json.loads((out/'tests.json').read_text())['required_checks_passed']
    assert json.loads((out/'static-summary.json').read_text())['loading_allowed']
    def launch(name):
        with (out/(name+'.log')).open('x') as log:
            r=subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_selective_history','worker','--case',name,'--output',str(out)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        print(name,r.returncode,flush=True);return dict(case=name,exit_code=r.returncode)
    # Start finest cases first, maintaining four independent solver processes.
    names=[label+'-'+level for level in reversed(LEVELS) for label in ('F45','ISO','F0','F90')]
    with ThreadPoolExecutor(max_workers=jobs) as pool:r=list(pool.map(launch,names))
    base.write_json(out/'batch.json',r);return all(x['exit_code']==0 for x in r)


def analyze(out,p):
    from benchmarks.aniso_dynamic_space import relative,stress
    data={};checks={};frames={};audits=[]
    for name,cfg in p['configs'].items():
        data[name],checks[name]=read_case(out/'cases'/name)
        with np.load(out/'cases'/name/'frames.npz') as z:frames[name]=z['F'].copy()
        audits.extend(json.loads(f.read_text())|{'case':name} for f in (out/'cases'/name).glob('audit-*.json'))
    rows=[]
    for label in CASES:
        pairs=[];t=np.array([r['time'] for r in data[label+'-coarse']]);t=t[t>=.05-1e-12]
        rms=lambda a:float(np.sqrt(np.trapezoid(a*a,t)/(t[-1]-t[0])))
        for l0,l1 in zip(list(LEVELS)[:-1],list(LEVELS)[1:]):
            a,b=label+'-'+l0,label+'-'+l1;ra=np.interp(t,[r['time'] for r in data[a]],[r['right_force'] for r in data[a]]);rb=np.interp(t,[r['time'] for r in data[b]],[r['right_force'] for r in data[b]])
            Fa,Fb=frames[a][-1],frames[b][-1];absR=rms(ra-rb);row=dict(pair=[l0,l1],reaction_absolute_rms_N=absR,reaction_relative=absR/max(rms(rb),.001),F_relative=relative(Fa-np.eye(3),Fb-np.eye(3)),P_relative=relative(stress(Fa,p['configs'][a]),stress(Fb,p['configs'][b])),energy_relative=abs(data[a][-1]['elastic']-data[b][-1]['elastic'])/max(abs(data[b][-1]['elastic']),1e-30))
            if pairs:row['rho']=absR/pairs[-1]['reaction_absolute_rms_N'];row['observed_order']=-float(np.log2(row['rho']))
            row['threshold_passed']=max(row[k] for k in ('reaction_relative','F_relative','P_relative','energy_relative'))<=.02;pairs.append(row)
        rows.append(dict(case=label,pairs=pairs,time_threshold_passed=pairs[-1]['threshold_passed'],reliable_reaction_trend=all(r['observed_order']>=.5 for r in pairs[1:])))
    summary=dict(completed=all(c['completed'] for c in checks.values()),physical_checks_passed=all(c['passed'] for c in checks.values()),checks=checks,refinement=rows,history_closure_max=max(r['frozen_F_max'] for r in audits),particle_trial_commit_max=max(r['particle_F_update_max'] for r in audits),snapshot_count=len(audits),time_threshold_passed=all(r['time_threshold_passed'] for r in rows),reliable_reaction_trend=all(r['reliable_reaction_trend'] for r in rows),default_changed=False,accuracy_certified=False)
    base.write_json(out/'summary.json',summary);print({k:v for k,v in summary.items() if k not in ('checks','refinement')},flush=True);return summary['physical_checks_passed']


def main():
    wp.config.kernel_cache_dir='/tmp/mpm-lite-warp-cache'
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=('freeze','tests','static','run','worker','analyze'));p.add_argument('--output',type=Path,default=OUT);p.add_argument('--case');p.add_argument('--jobs',type=int,choices=(1,2,4),default=4);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    if a.action=='freeze':freeze(a.output);return
    protocol=json.loads((a.output/'protocol.json').read_text());assert protocol['source_sha256']==hashes(),'frozen source changed'
    ok=base.tests(a.output,protocol) if a.action=='tests' else static(a.output,protocol) if a.action=='static' else run(a.output,protocol,a.jobs) if a.action=='run' else worker(a.output,protocol,a.case) if a.action=='worker' else analyze(a.output,protocol)
    raise SystemExit(0 if ok else 2)


if __name__=='__main__':main()
