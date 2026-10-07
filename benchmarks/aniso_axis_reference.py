"""Equal-cell-count X/Y/Z reference refinement from (88,22,22)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np

from engine.aniso_phase1.directional_reference import directional_geometry
from engine.aniso_phase1.convergence_reference import tensor_rule,knots
from .aniso_history_increment import guard,DATA_ROOT
from .aniso_convergence_reference import save,fingerprint
from .aniso_spatial_energy import run as run_path,cached_compare
from .aniso_refinement import field_axes
from .aniso_reference_scale import candidate_fields,scaled_differences
from .aniso_directional_reference import (
    OUT as PREVIOUS,PREVIOUS as CANDIDATE_ROOT,initial,fields,location,evaluated,
    initial_representation,endpoint,difference_norms)

OUT=Path('docs/results/axis-reference/v1')
CASES=dict(base=(88,22,22),X=(96,22,22),Y=(88,24,22),Z=(88,22,24))
FINE=6.103515625e-8
TIME_TARGET=.05
SPACE_TARGET=.1
EDGES=[np.linspace(.25,.75,33),np.linspace(.4375,.5625,33),np.linspace(.4375,.5625,33)]
KEYS=('x','F','P','v')


def label(case):return 'x'.join(map(str,CASES[case]))


def catalog(out,case):
    signature=fingerprint(initial());axes=knots(directional_geometry(CASES[case]));result={}
    for root in ([PREVIOUS,out] if case=='base' else [out]):
        for path in (root/'original').glob(f'g{label(case)}-dt*.json'):
            m=json.loads(path.read_text());c=m['config']
            if c['initial']!=signature or c['order']!=5 or not all(m['checks'].values()):
                raise ValueError(f'initial/integration/acceptance mismatch: {path}')
            if any(not np.array_equal(a,b) for a,b in zip(axes,c.get('reference_axes',[]))) or len(c.get('reference_axes',[]))!=3:
                raise ValueError(f'axis mismatch: {path}')
            if c['duration']==.003 and path.with_suffix('.npz').exists():result[c['dt']]=(path,m)
    return result


def run(out,case,dt,device):
    import warp as wp
    from utils.resource_guard import prepare_warp_cache
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)
    return run_path(out,'original',label(case),dt,device,initial=initial(),source=directional_geometry(CASES[case]))


def prepare(out):
    report=initial_representation(out,CASES)
    rows={}
    old=knots(directional_geometry((8,2,2)))
    for case,counts in CASES.items():
        src=directional_geometry(counts)
        nested=all(all(np.min(abs(axis-p))<1e-14 for p in history) for axis,history in zip(src.axes,old))
        rows[case]=dict(cells=list(counts),cell_count=int(np.prod(counts)),free_components=int(3*sum(src.free)),
                       material_points=int(np.prod(counts)*125),old_interfaces_retained=bool(nested))
    if len({rows[c]['cell_count'] for c in ('X','Y','Z')})!=1 or not all(r['old_interfaces_retained'] for r in rows.values()):
        raise RuntimeError('equal-cost/history-interface preflight failed')
    result=dict(initial=report['initial'],representation_passed=report['passed'],cases=rows,time_target=TIME_TARGET,
                marginal_space_target=SPACE_TARGET,initial_fine_dt=FINE,storage=guard())
    save(out/'protocol.json',result);return result


def time_check(out,case):
    items=catalog(out,case);candidates=candidate_fields(CANDIDATE_ROOT);pairs=[]
    for dt in sorted(items,reverse=True):
        if 2*dt not in items:continue
        comparison=scaled_differences(out,fields(items[2*dt]),fields(items[dt]),candidates)
        passed=all(row['ratios'][k]<TIME_TARGET for row in comparison.values() for k in ('F','P','v'))
        pairs.append(dict(dt_fine=dt,comparison=comparison,passed=passed))
        print('TIME',case,dt,{g:v['ratios'] for g,v in comparison.items()},passed,flush=True)
    result=dict(case=case,target=TIME_TARGET,pairs=pairs);save(out/f'time-{case}.json',result);return result


def concentration(report):
    """Predeclared Y band, plus exploratory contiguous 1/8-width peak bands."""
    def band(d,start,end):
        coords=np.asarray(report['profiles']['F'][d]['coordinate']);mask=(coords>=start)&(coords<end)
        return dict(bounds=[float(start),float(end)],volume_fraction=float(np.mean(mask)),
                    squared_share={k:float(np.asarray(report['profiles'][k][d]['squared_share'])[mask].sum()) for k in KEYS})
    result=dict(predeclared_Y_band=band(1,.453125,.46875),exploratory_peak_bands={})
    for d in (0,1,2):
        if report['rms']['F']==0:
            result['exploratory_peak_bands']['XYZ'[d]]=None
            continue
        p=report['profiles']['F'][d];coords=np.asarray(p['coordinate']);width=coords[1]-coords[0]
        count=len(coords)//8;shares=np.asarray(p['squared_share'])
        i=int(np.argmax(np.convolve(shares,np.ones(count),mode='valid')))
        result['exploratory_peak_bands']['XYZ'[d]]=band(d,coords[i]-width/2,coords[i+count-1]+width/2)
    return result


def spatial(out,case,dt):
    a=fields(catalog(out,case)[dt]);b=fields(catalog(out,'base')[dt])
    # Pairwise common cuts plus identical profile boundaries: norms are
    # comparable without constructing the much larger all-grid Cartesian union.
    loc=location(out,a,b,f'{case}-base-dt{dt:.13f}',edges=EDGES)
    comparison=scaled_differences(out,a,b,candidate_fields(CANDIDATE_ROOT))
    result=dict(case=case,dt=dt,location=loc,comparison=comparison,concentration=concentration(loc))
    result['marginal_passed']=all(v['ratios'][k]<SPACE_TARGET for v in comparison.values() for k in ('F','P','v'))
    save(out/f'space-{case}-dt{dt:.13f}.json',result);return result


def effect_time(out,case,dt,states=None):
    if states is None:
        c,b=catalog(out,case),catalog(out,'base')
        states=[fields(c[dt]),fields(b[dt]),fields(c[2*dt]),fields(b[2*dt])]
    signature=dict(fields=[fingerprint(f) for f in states],dt=dt,order=3)
    path=out/f'effect-time-{case}-dt{dt:.13f}.json'
    if path.exists():
        result=json.loads(path.read_text())
        if result['signature']==signature:return result
    X,w=tensor_rule(field_axes(*states),3)
    size={k:0. for k in KEYS};change=size.copy();coarse=size.copy()
    for start in range(0,len(X),32768):
        guard();p=X[start:start+32768];weights=w[start:start+32768]
        a,b,c,d=[evaluated(f,p) for f in states]
        for k in KEYS:
            delta=a[k]-b[k];previous=c[k]-d[k]
            for total,value in ((size,delta),(coarse,previous),(change,delta-previous)):
                total[k]+=float(weights@(value.reshape(len(p),-1)**2).sum(axis=1))
        if start//32768%64==0:print('EFFECT TIME',case,start+len(p),'/',len(X),flush=True)
    ratios={k:float(np.sqrt(change[k]/max(size[k],1e-300))) for k in KEYS}
    result=dict(signature=signature,relative=ratios,fine_rms={k:np.sqrt(v/w.sum()) for k,v in size.items()},
                coarse_rms={k:np.sqrt(v/w.sum()) for k,v in coarse.items()},
                passed=all(ratios[k]<TIME_TARGET for k in ('F','P','v')))
    save(path,result);return result


def analyze(out):
    items={c:catalog(out,c) for c in CASES};common=set.intersection(*(set(v) for v in items.values()))
    if not common:raise RuntimeError('no common time step')
    dt=min(common);times={c:time_check(out,c) for c in CASES}
    passed={c:any(p['dt_fine']==dt and p['passed'] for p in t['pairs']) for c,t in times.items()}
    result=dict(dt=dt,time=times,time_passed=passed,spatial={},effect_time={},joint_reference_tested=False)
    save(out/'results-progress.json',result)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={c:pool.submit(spatial,out,c,dt) for c in ('X','Y','Z')}
        for c,job in jobs.items():result['spatial'][c]=job.result();save(out/'results-progress.json',result)
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs={c:pool.submit(effect_time,out,c,dt) for c in ('X','Y','Z')}
        result['effect_time']={c:job.result() for c,job in jobs.items()}
    result['marginal_space_passed']=all(row['marginal_passed'] for row in result['spatial'].values())
    records=[m for values in items.values() for p,m in values.values() if p.is_relative_to(out)]
    result.update(runs=len(records),steps=sum(len(m['rows']) for m in records),
                  numerical_checks_passed=all(all(m['checks'].values()) for m in records),storage=guard())
    save(out/'results.json',result);return result


def check(out,case,dt):
    items=catalog(out,case)
    if dt not in items or 2*dt not in items:raise RuntimeError('missing temporal pair')
    t=time_check(out,case)
    row=next(p for p in t['pairs'] if p['dt_fine']==dt)
    final=fields(items[dt]);coarse=fields(items[2*dt])
    result=dict(case=case,dt=dt,time_passed=row['passed'],
                endpoint=endpoint(out,case,final,counts=CASES[case]))
    difference_norms(out,coarse,final,5)
    if case!='base':
        result['spatial']=spatial(out,case,dt)
        result['effect_time']=effect_time(out,case,dt)
        difference_norms(out,final,fields(catalog(out,'base')[dt]),5)
    save(out/f'check-{case}-dt{dt:.13f}.json',result);return result


def audit(out):
    r=json.loads((out/'results.json').read_text());dt=r['dt'];items={c:catalog(out,c) for c in CASES}
    states={c:fields(v[dt]) for c,v in items.items()};init=json.loads((out/'initial-representation.json').read_text())
    if init['initial']!=fingerprint(initial()) or not init['passed']:raise ValueError('initial representation check failed')
    report=dict(initial=init,endpoints={},space_probes={},time_probes={},gap_probes={})
    def check(case):
        result=dict(endpoint=endpoint(out,case,states[case],counts=CASES[case]))
        independent=difference_norms(out,fields(items[case][2*dt]),states[case],5)
        t=next(p for p in r['time'][case]['pairs'] if p['dt_fine']==dt)['comparison']['17']['difference']['all']
        result['time']={k:abs(t[k+'_rms']/v-1) for k,v in independent.items()}
        if case!='base':
            independent=difference_norms(out,states[case],states['base'],5)
            reference=r['spatial'][case]
            result['space']={kind:{k:abs((reference['location']['rms'][k] if kind=='profiles' else reference['comparison']['17']['difference']['all'][k+'_rms'])/v-1) for k,v in independent.items()} for kind in ('profiles','comparison')}
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs={c:pool.submit(check,c) for c in CASES}
        for case,job in jobs.items():
            v=job.result();report['endpoints'][case]=v['endpoint'];report['time_probes'][case]=v['time']
            if 'space' in v:report['space_probes'][case]=v['space']
            save(out/'audit-progress.json',report)
    for g,pair in candidate_fields(CANDIDATE_ROOT).items():
        independent=cached_compare(out,*pair,field_axes(*pair),7)['all']
        report['gap_probes'][g]={case:{k:abs(row['comparison'][g]['scheme_gap']['all'][k+'_rms']/independent[k+'_rms']-1) for k in KEYS} for case,row in r['spatial'].items()}
    def worst(obj):return max((worst(v) if isinstance(v,dict) else v) for v in obj.values())
    report['passed']=(all(v['passed'] for v in report['endpoints'].values()) and
                      all(worst(report[k])<1e-6 for k in ('space_probes','time_probes','gap_probes')))
    report['storage']=guard();save(out/'audit.json',report)
    if not report['passed']:raise RuntimeError('axis comparison quadrature audit failed')
    return report


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((out/'results.json').read_text());fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,k in zip(axs,('F','P','v')):
        for j,g in enumerate(('17','33')):
            ax.bar(np.arange(3)+(j-.5)*.35,[r['spatial'][c]['comparison'][g]['ratios'][k] for c in ('X','Y','Z')],.35,label=f'candidate {g}')
        ax.set_xticks(range(3),['refine X','refine Y','refine Z']);ax.set_ylabel('Marginal R_'+k);ax.set_yscale('log')
        ax.axhline(SPACE_TARGET,color='r',linestyle='--');ax.legend()
    fig.savefig(out/'directional-ratios.png',dpi=180);fig.savefig(out/'directional-ratios.pdf');plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(14,10),sharey='row',layout='constrained')
    for i,k in enumerate(('F','P','v')):
        for d in range(3):
            ax=axs[i,d]
            for c,row in r['spatial'].items():
                p=row['location']['profiles'][k][d]
                ax.plot(p['coordinate'],p['rms'],label=f'refine {c}')
            ax.set_xlabel('material '+'XYZ'[d]);ax.set_ylabel(k+' change RMS')
            if d==0:
                ax.axvspan(.25,.3125,color='gray',alpha=.1)
                ax.axvspan(.4375,.5625,color='orange',alpha=.1)
            elif d==1:ax.axvspan(.453125,.46875,color='blue',alpha=.08)
            ax.legend()
    for row in axs:row[0].set_ylim(bottom=0)
    fig.savefig(out/'profiles.png',dpi=180);fig.savefig(out/'profiles.pdf');plt.close(fig)


def progress(out):
    import time,re
    from datetime import datetime,timezone
    while True:
        status=guard();lines=['# 分轴参考实验实时进度\n\n',f'更新时间（UTC）：{datetime.now(timezone.utc).isoformat()}\n\n',
            f'系统／数据盘剩余 {status["system_free_bytes"]/2**30:.2f}/{status["data_free_bytes"]/2**30:.2f} GiB。\n\n',
            '| 方向 | dt (μs) | 步数 | 终态文件 |\n|---|---:|---:|---|\n']
        paths=set(out.glob('*.log'))
        paths.update(out/f'{case}-{dt:.13f}.log' for case in ('X','Y','Z') for dt in (FINE,2*FINE))
        for path in sorted(paths):
            match=re.fullmatch(r'(base|X|Y|Z)-([0-9.]+)\.log',path.name)
            if not match:continue
            rows=re.findall(r': (\d+)/(\d+), residual=',path.read_text() if path.exists() else '');case,dt=match[1],float(match[2])
            step,total=rows[-1] if rows else ('0',str(round(.003/dt)))
            meta=out/'original'/f'g{label(case)}-dt{dt:.13f}.json'
            status_text='已保存' if meta.exists() else '运行中' if path.exists() else '待启动'
            lines.append(f'| {case} | {.003/int(total)*1e6:.9f} | {step}/{total} | {status_text} |\n')
        lines.append('\n基线 (88,22,22) 复用前轮。运行进度不代表精度通过。\n')
        if (out/'results.json').exists():
            r=json.loads((out/'results.json').read_text());lines.append(f'\n共同 dt={r["dt"]*1e6:.9f} μs，时间：{r["time_passed"]}；分轴空间门槛：{r["marginal_space_passed"]}。未计算三方向共同加密的交互。\n')
        if (out/'audit.json').exists():lines.append(f'\n独立积分复核：{json.loads((out/"audit.json").read_text())["passed"]}。\n')
        (out/'PROGRESS.md').write_text(''.join(lines))
        if (out/'audit.json').exists() or (out/'stop-progress').exists():return
        for _ in range(10):
            time.sleep(30)
            if (out/'audit.json').exists() or (out/'stop-progress').exists():break


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('prepare','run','time','spatial','check','analyze','audit','plot','progress'),required=True)
    p.add_argument('--case',choices=tuple(CASES),default='base');p.add_argument('--dt',type=float,default=FINE)
    p.add_argument('--device',default='cuda:0');p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args();guard();a.out.mkdir(parents=True,exist_ok=True)
    if a.stage=='prepare':prepare(a.out)
    elif a.stage=='run':run(a.out,a.case,a.dt,a.device)
    elif a.stage=='time':time_check(a.out,a.case)
    elif a.stage=='spatial':spatial(a.out,a.case,a.dt)
    elif a.stage=='check':check(a.out,a.case,a.dt)
    else:globals()[a.stage](a.out)


if __name__=='__main__':main()
