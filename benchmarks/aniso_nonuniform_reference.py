"""Equal-cost local Y refinement, with retained X/Z controls and a finer Y probe."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path
import numpy as np

from engine.aniso_phase1.directional_reference import directional_geometry,cartesian_geometry
from engine.aniso_phase1.convergence_reference import knots
from . import aniso_axis_reference as previous
from .aniso_directional_reference import initial,fields,initial_representation,endpoint,difference_norms,location
from .aniso_axis_reference import EDGES,KEYS,TIME_TARGET,SPACE_TARGET,FINE,concentration,effect_time
from .aniso_reference_scale import candidate_fields,scaled_differences
from .aniso_spatial_energy import run as run_path,cached_compare
from .aniso_refinement import field_axes
from .aniso_convergence_reference import save,fingerprint
from .aniso_history_increment import guard,DATA_ROOT

OUT=Path('docs/results/nonuniform-reference/v1')
BAND=(.453125,.46875)
CASES=dict(previous.CASES,localY=(88,24,22),fineY=(88,48,22))
NEW=('localY','fineY')
PAIRS=dict(local_base=('localY','base'),uniform_base=('Y','base'),local_uniform=('localY','Y'),
           fine_base=('fineY','base'),local_fine=('localY','fineY'),uniform_fine=('Y','fineY'),
           X_base=('X','base'),Z_base=('Z','base'))
CONTROL_PAIRS=dict(uniform_base='Y',X_base='X',Z_base='Z')
_meta_cache={}


@lru_cache(None)
def source(case):
    if case!='localY':return directional_geometry(CASES[case])
    axes=list(knots(directional_geometry(CASES['base'])))
    y=axes[1];ids=np.flatnonzero((y[:-1]>=BAND[0])&(y[1:]<=BAND[1]))
    if len(ids)!=2:raise ValueError('predeclared band must contain two complete base cells')
    axes[1]=np.sort(np.r_[y,(y[ids]+y[ids+1])/2])
    return cartesian_geometry(axes)


def label(case):return 'x'.join(map(str,CASES[case]))+('-localY' if case=='localY' else '')


def read_meta(path):
    key=(path.stat().st_mtime_ns,path.stat().st_size)
    if path not in _meta_cache or _meta_cache[path][0]!=key:
        m=json.loads(path.read_text());m['steps']=len(m.pop('rows'))
        _meta_cache[path]=(key,m)
    return _meta_cache[path][1]


def catalog(out,case):
    roots=[out]
    if case in previous.CASES:
        roots=([previous.PREVIOUS,previous.OUT] if case=='base' else [previous.OUT])+roots
    result={};expected=knots(source(case));signature=fingerprint(initial())
    for root in roots:
        for p in (root/'original').glob(f'g{label(case)}-dt*.json'):
            m=read_meta(p);c=m['config'];axes=c.get('reference_axes',[])
            if (c['initial']!=signature or c['order']!=5 or not all(m['checks'].values()) or
                    len(axes)!=3 or any(not np.array_equal(a,b) for a,b in zip(axes,expected))):
                raise ValueError(f'incompatible history/axes/integration/checks: {p}')
            if c['duration']==.003 and p.with_suffix('.npz').exists():result[c['dt']]=(p,m)
    return result


def prepare(out):
    out.mkdir(parents=True,exist_ok=True);guard()
    sources={c:source(c) for c in CASES}
    check=initial_representation(out,CASES,sources=sources);rows={}
    old=knots(directional_geometry((8,2,2)))
    for c,s in sources.items():
        nested=all(all(np.min(abs(axis-x))<1e-14 for x in history) for axis,history in zip(knots(s),old))
        rows[c]=dict(cells=s.counts.tolist(),cell_count=int(np.prod(s.counts)),free_components=int(3*sum(s.free)),
                     material_points=int(np.prod(s.counts)*125),axes=[a.tolist() for a in knots(s)],
                     min_widths=[float(np.diff(a).min()) for a in knots(s)],old_interfaces_retained=bool(nested))
    extra=np.setdiff1d(sources['localY'].axes[1],sources['base'].axes[1])
    same=all(rows['localY'][k]==rows['Y'][k] for k in ('cell_count','free_components','material_points'))
    if not same or len(extra)!=2 or not np.all((extra>BAND[0])&(extra<BAND[1])) or not all(v['old_interfaces_retained'] for v in rows.values()):
        raise RuntimeError('equal-cost/old-interface preflight failed')
    r=dict(initial=check['initial'],initial_passed=check['passed'],cases=rows,added_Y=extra.tolist(),band=list(BAND),
           equal_cost_passed=same,time_target=TIME_TARGET,space_target=SPACE_TARGET,fine_dt=FINE,
           fineY_is_converged_reference=False,storage=guard())
    save(out/'protocol.json',r);return r


def initial_audit(out):
    result={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={c:pool.submit(endpoint,out,'initial-'+c,initial(),source=source(c)) for c in NEW}
        for c,job in jobs.items():
            result[c]=job.result();save(out/'initial-quadrature-progress.json',result)
    report=dict(cases=result,passed=all(v['passed'] for v in result.values()),initial=fingerprint(initial()),storage=guard())
    save(out/'initial-quadrature.json',report)
    if not report['passed']:raise RuntimeError('initial nonuniform quadrature failed')
    return report


def retained(out):
    times={c:time_check(out,c,FINE) for c in previous.CASES}
    pairs={n:pair(out,n,FINE) for n in CONTROL_PAIRS}
    report=dict(time=times,pairs=pairs,dt=FINE,new_trajectories_complete=False,storage=guard())
    save(out/'retained-controls.json',report);return report


def run(out,case,dt,device):
    import warp as wp
    from utils.resource_guard import prepare_warp_cache
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)
    return run_path(out,'original',label(case),dt,device,initial=initial(),source=source(case))


def time_check(out,case,dt):
    items=catalog(out,case)
    if dt not in items or 2*dt not in items:raise RuntimeError('missing time pair')
    cache_root=previous.OUT if case in previous.CASES else out
    comparison=scaled_differences(cache_root,fields(items[2*dt]),fields(items[dt]),candidate_fields(previous.CANDIDATE_ROOT))
    r=dict(case=case,dt=dt,comparison=comparison,
           passed=all(v['ratios'][k]<TIME_TARGET for v in comparison.values() for k in ('F','P','v')))
    save(out/f'time-{case}-dt{dt:.13f}.json',r)
    print('TIME',case,{g:v['ratios'] for g,v in comparison.items()},r['passed'],flush=True);return r


def pair(out,name,dt):
    ca,cb=PAIRS[name];a=fields(catalog(out,ca)[dt]);b=fields(catalog(out,cb)[dt])
    if name in CONTROL_PAIRS:
        old=previous.OUT/f'space-{CONTROL_PAIRS[name]}-dt{dt:.13f}.json'
        if old.exists():
            p=json.loads(old.read_text())
            signature=p['location']['signature']
            if signature['a']==fingerprint(a) and signature['b']==fingerprint(b) and signature.get('order')==3 and signature.get('edges')==[e.tolist() for e in EDGES]:
                r=dict(pair=[ca,cb],dt=dt,location=p['location'],comparison=p['comparison'],
                       concentration=p['concentration'],reused_from=str(old))
                save(out/f'pair-{name}-dt{dt:.13f}.json',r);return r
    loc=location(out,a,b,f'{name}-dt{dt:.13f}',edges=EDGES)
    r=dict(pair=[ca,cb],dt=dt,location=loc,
           comparison=scaled_differences(out,a,b,candidate_fields(previous.CANDIDATE_ROOT)),concentration=concentration(loc))
    save(out/f'pair-{name}-dt{dt:.13f}.json',r);return r


def contrast_time(out,name,dt):
    a,b=PAIRS[name]
    states=[fields(catalog(out,c)[step]) for step in (dt,2*dt) for c in (a,b)]
    return effect_time(out,name,dt,states=states)


def quality(local,uniform):
    """Errors relative to a declared probe; never assert that probe is truth."""
    result={}
    for k in KEYS:
        l,u=local['location']['rms'][k],uniform['location']['rms'][k]
        sl=local['concentration']['predeclared_Y_band']['squared_share'][k]
        su=uniform['concentration']['predeclared_Y_band']['squared_share'][k]
        regions={}
        for name,volume,cl,cu in (('band',.125,sl,su),('outside',.875,1-sl,1-su)):
            el=l*np.sqrt(max(cl,0)/volume);eu=u*np.sqrt(max(cu,0)/volume)
            regions[name]=dict(local=float(el),uniform=float(eu),ratio=float(el/eu) if eu>0 else None)
        result[k]=dict(local=l,uniform=u,ratio=l/u if u>0 else None,gain=u-l,regions=regions)
    return result


def check_case(out,case,dt):
    t=time_check(out,case,dt);items=catalog(out,case);f=fields(items[dt]);coarse=fields(items[2*dt])
    e=endpoint(out,case,f,source=source(case));difference_norms(out,coarse,f,5)
    names=('local_base','local_uniform') if case=='localY' else ('fine_base','local_fine','uniform_fine') if case=='fineY' else ()
    result=dict(time=t,endpoint=e,pairs={},effect_time={})
    for name in names:
        result['pairs'][name]=pair(out,name,dt)
        result['effect_time'][name]=contrast_time(out,name,dt)
        ca,cb=PAIRS[name];difference_norms(out,fields(catalog(out,ca)[dt]),fields(catalog(out,cb)[dt]),5)
    save(out/f'check-{case}-dt{dt:.13f}.json',result);return result


def analyze(out,dt):
    times={c:time_check(out,c,dt) for c in CASES};pairs={};effects={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={name:pool.submit(pair,out,name,dt) for name in PAIRS}
        for name,job in jobs.items():pairs[name]=job.result()
    for name in ('local_base','local_uniform','fine_base','local_fine','uniform_fine'):
        effects[name]=contrast_time(out,name,dt)
    q=quality(pairs['local_fine'],pairs['uniform_fine'])
    for k,v in q.items():
        coarse_gain=effects['uniform_fine']['coarse_rms'][k]-effects['local_fine']['coarse_rms'][k]
        v['coarse_gain']=coarse_gain
        v['gain_time_relative']=abs(v['gain']-coarse_gain)/abs(v['gain']) if v['gain']!=0 else None
    records=[m for c in CASES for p,m in catalog(out,c).values() if p.is_relative_to(out)]
    r=dict(dt=dt,time=times,time_passed={c:v['passed'] for c,v in times.items()},pairs=pairs,effect_time=effects,
           quality_against_fineY=q,full_reference_ready=False,fineY_is_converged_reference=False,
           joint_nonuniform_XZ_tested=False,runs=len(records),steps=sum(m['steps'] for m in records),
           numerical_passed=all(all(m['checks'].values()) for m in records),storage=guard())
    save(out/'results.json',r);return r


def audit(out):
    r=json.loads((out/'results.json').read_text());dt=r['dt'];states={c:fields(catalog(out,c)[dt]) for c in CASES}
    report=dict(endpoints={},time={},space={},gap={})
    def case_check(c):
        ep=endpoint(out,c,states[c],source=source(c))
        norms=difference_norms(out,fields(catalog(out,c)[2*dt]),states[c],5)
        primary=r['time'][c]['comparison']['17']['difference']['all']
        return ep,{k:abs(primary[k+'_rms']/v-1) for k,v in norms.items()}
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={c:pool.submit(case_check,c) for c in CASES}
        for c,j in jobs.items():report['endpoints'][c],report['time'][c]=j.result();save(out/'audit-progress.json',report)
        jobs={name:pool.submit(difference_norms,out,states[ca],states[cb],5) for name,(ca,cb) in PAIRS.items()}
        for name,j in jobs.items():
            independent=j.result();p=r['pairs'][name]
            report['space'][name]={kind:{k:abs((p['location']['rms'][k] if kind=='profiles' else p['comparison']['17']['difference']['all'][k+'_rms'])/v-1) for k,v in independent.items()} for kind in ('profiles','comparison')}
    for g,candidates in candidate_fields(previous.CANDIDATE_ROOT).items():
        independent=cached_compare(out,*candidates,field_axes(*candidates),7)['all']
        report['gap'][g]={name:{k:abs(p['comparison'][g]['scheme_gap']['all'][k+'_rms']/independent[k+'_rms']-1) for k in KEYS} for name,p in r['pairs'].items()}
    def worst(x):return max(worst(v) if isinstance(v,dict) else float(v) for v in x.values())
    report['passed']=all(v['passed'] for v in report['endpoints'].values()) and all(worst(report[k])<1e-6 for k in ('space','time','gap'))
    report['initial_passed']=json.loads((out/'initial-representation.json').read_text())['passed']
    report['passed']=report['passed'] and report['initial_passed'];report['storage']=guard();save(out/'audit.json',report)
    if not report['passed']:raise RuntimeError('independent quadrature audit failed')
    return report


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((out/'results.json').read_text())
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,k in zip(axs,('F','P','v')):
        q=r['quality_against_fineY'][k]
        vals=[q['ratio'],q['regions']['band']['ratio'],q['regions']['outside']['ratio']]
        ax.bar(range(3),vals);ax.axhline(1,color='r',ls='--');ax.set_xticks(range(3),['whole beam','Y band','outside band'])
        ax.set_ylabel(k+' discrepancy: local / uniform');ax.set_ylim(bottom=0)
    fig.suptitle('Relative to (88,48,22) Y probe; not a converged exact solution')
    fig.savefig(out/'probe-comparison.png',dpi=180);fig.savefig(out/'probe-comparison.pdf');plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(14,10),sharey='row',layout='constrained')
    for i,k in enumerate(('F','P','v')):
        for d in range(3):
            ax=axs[i,d]
            for name,title in (('local_fine','local Y24 - Y48'),('uniform_fine','uniform Y24 - Y48')):
                p=r['pairs'][name]['location']['profiles'][k][d];ax.plot(p['coordinate'],p['rms'],label=title)
            ax.set_xlabel('material '+'XYZ'[d]);ax.set_ylabel(k+' discrepancy RMS');ax.legend()
            if d==1:ax.axvspan(*BAND,color='blue',alpha=.08)
            if d==0:ax.axvspan(.4375,.5625,color='orange',alpha=.08)
    for row in axs:row[0].set_ylim(bottom=0)
    fig.savefig(out/'profiles.png',dpi=180);fig.savefig(out/'profiles.pdf');plt.close(fig)


def progress(out,once=False):
    import time,re
    from datetime import datetime,timezone
    while True:
        s=guard();status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        lines=['# 非均匀 Y 参考实验进度\n\n',datetime.now(timezone.utc).isoformat()+'\n\n',
            f'系统／数据盘剩余 {s["system_free_bytes"]/2**30:.2f}/{s["data_free_bytes"]/2**30:.2f} GiB。\n\n',
            status.get('message','实验准备／运行中')+'\n\n',
            '| 网格 | dt (μs) | 步数 | 状态 |\n|---|---:|---:|---|\n']
        paths=set(out.glob('run-*.log'))|{out/f'run-{c}-dt{dt:.13f}.log' for c in NEW for dt in (FINE,2*FINE)}
        for f in sorted(paths):
            match=re.fullmatch(r'run-(\w+)-dt([0-9.]+)\.log',f.name)
            if not match:continue
            c=match[1];dt=float(match[2]);rows=re.findall(r': (\d+)/(\d+), residual=',f.read_text() if f.exists() else '')
            n,total=map(int,rows[-1]) if rows else (0,round(.003/dt));meta=out/'original'/f'g{label(c)}-dt{dt:.13f}.json'
            run_status='已保存' if meta.exists() else '未启动：CUDA 不可用' if status.get('state')=='blocked_cuda' else '请查看运行日志' if f.exists() else '待启动'
            lines.append(f'| {c} | {.003/total*1e6:.9f} | {n}/{total} | {run_status} |\n')
        lines.append('\n基线、均匀 Y、X/Z 对照复用并验证前轮数据。步数完成不等于精度通过。\n')
        for c in CASES:
            files=sorted(out.glob(f'time-{c}-dt*.json'))
            if files:
                v=json.loads(files[0].read_text());lines.append(f'\n{c} 时间门槛通过：{v["passed"]}。\n')
        if (out/'audit.json').exists():lines.append(f'\n独立积分审计通过：{json.loads((out/"audit.json").read_text())["passed"]}。\n')
        (out/'PROGRESS.md').write_text(''.join(lines))
        if once or (out/'audit.json').exists() or (out/'stop-progress').exists() or status.get('state')=='blocked_cuda':return
        for _ in range(10):
            time.sleep(30)
            if (out/'audit.json').exists() or (out/'stop-progress').exists():break


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('prepare','run','time','check','analyze','audit','plot','progress','initial-audit','retained'),required=True)
    p.add_argument('--out',type=Path,default=OUT);p.add_argument('--case',choices=CASES,default='localY');p.add_argument('--dt',type=float,default=FINE)
    p.add_argument('--device',default='cuda:0');p.add_argument('--once',action='store_true');a=p.parse_args();guard();a.out.mkdir(parents=True,exist_ok=True)
    if a.stage=='run':run(a.out,a.case,a.dt,a.device)
    elif a.stage=='time':time_check(a.out,a.case,a.dt)
    elif a.stage=='check':check_case(a.out,a.case,a.dt)
    elif a.stage=='analyze':analyze(a.out,a.dt)
    elif a.stage=='progress':progress(a.out,a.once)
    else:globals()[a.stage.replace('-','_')](a.out)


if __name__=='__main__':main()
