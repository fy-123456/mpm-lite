"""Short full-grid nonuniform-Y comparison with consistent-mass PIC history."""
import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

from .aniso_nonuniform_reference import source, label, BAND
from .aniso_directional_reference import initial, evaluated, endpoint
from .aniso_spatial_energy import run as run_path
from .aniso_refinement import field_axes
from .aniso_convergence_reference import save, fingerprint
from .aniso_history_increment import guard, DATA_ROOT
from engine.aniso_phase1.convergence_reference import load_fields, tensor_rule, union_knots

OUT = Path('docs/results/practical-validation/v1')
CASES = ('localY', 'Y', 'fineY')
DT = 1.953125e-6
DURATION = .000125
TIME_TARGET = .02
SPACE_TARGET = .05
KEYS = ('x', 'F', 'P', 'v')


def progress(out, message):
    out.mkdir(parents=True, exist_ok=True)
    status = dict(time=datetime.now(timezone.utc).isoformat(), message=message, storage=guard())
    save(out/'status.json', status)
    with (out/'PROGRESS.md').open('a') as f:
        f.write(f"- {status['time']}：{message}\n")
    print(message, flush=True)


def device_choice():
    # Honour CUDA_VISIBLE_DEVICES: Warp aliases can differ from physical ids.
    import warp as wp
    wp.init()
    devices = wp.get_cuda_devices()
    if not devices:
        raise RuntimeError('no visible CUDA device')
    rows = []
    for device in devices:
        # Driver free memory includes other processes; mempool usage does not.
        free, total = device.free_memory, device.total_memory
        rows.append((total-free, -free, device.alias))
    return sorted(rows)[0][2]


def paths(out, case, factor):
    name = f'g{label(case)}-dt{DT/factor:.13f}'
    return out/'original'/name


def run(out, case, factor, device):
    import warp as wp
    from utils.resource_guard import prepare_warp_cache
    wp.config.kernel_cache_dir = prepare_warp_cache('/tmp/mpm-lite-warp-cache', DATA_ROOT)
    device = device_choice() if device == 'auto' else device
    progress(out, f'启动 {case} dt={DT/factor:g} s，{device}，consistent_pic')
    callback = lambda step, total, info: progress(out, f'{case} / {factor}: {step}/{total}，残差 {info["scaled_residual_inf"]:.3g}')
    meta, _ = run_path(out, 'original', label(case), DT/factor, device,
                      duration=DURATION, initial=initial(), source=source(case),
                      transfer_mode='consistent_pic', progress_callback=callback, replay=True)
    progress(out, f'{case} / {factor} 完成：{len(meta["rows"])} 步，数值检查 {meta["checks"]}')


def load(out, case, factor):
    path = paths(out, case, factor)
    meta = json.loads(Path(str(path)+'.json').read_text())
    config = meta['config']
    if (config['initial'] != fingerprint(initial()) or config['duration'] != DURATION or
        config['dt'] != DT/factor or config.get('transfer_mode') != 'consistent_pic' or
        config['reference_axes'] != [a.tolist() for a in source(case).axes] or
        config['order'] != 5 or not all(meta['checks'].values())):
        raise ValueError('incompatible run '+str(path))
    return load_fields(Path(str(path)+'.npz')), meta


def metric(out, name, a, b, order=3):
    path = out/f'metric-{name}-q{order}.json'
    signature = dict(a=fingerprint(a), b=fingerprint(b), order=order, band=list(BAND), protocol='practical-v1')
    if path.exists():
        r = json.loads(path.read_text())
        if r['signature'] == signature:
            return r
    # Split on both histories and the band edges; all regions use true volume.
    edges = [np.array([.25,.75]), np.array([.4375,*BAND,.5625]), np.array([.4375,.5625])]
    X, w = tensor_rule(union_knots(field_axes(a,b), edges), order)
    sums = {region: dict(volume=0., error=np.zeros(4), scale=np.zeros(4)) for region in ('all','band','outside')}
    for start in range(0,len(X),32768):
        guard()
        p, weights = X[start:start+32768], w[start:start+32768]
        av, bv = evaluated(a,p), evaluated(b,p)
        errors = np.column_stack([((av[k]-bv[k]).reshape(len(p),-1)**2).sum(axis=1) for k in KEYS])
        scales = np.column_stack([((bv[k]-(p if k=='x' else np.eye(3) if k=='F' else 0)).reshape(len(p),-1)**2).sum(axis=1) for k in KEYS])
        band = (p[:,1]>=BAND[0]) & (p[:,1]<=BAND[1])
        for region, mask in [('all',np.ones(len(p),bool)),('band',band),('outside',~band)]:
            row = sums[region]
            row['volume'] += float(weights[mask].sum())
            row['error'] += weights[mask] @ errors[mask]
            row['scale'] += weights[mask] @ scales[mask]
    result = dict(signature=signature, regions={})
    for region, row in sums.items():
        result['regions'][region] = dict(volume=row['volume'],
            rms={k:float(np.sqrt(v/row['volume'])) for k,v in zip(KEYS,row['error'])},
            relative={k:float(np.sqrt(e/max(s,1e-300))) for k,e,s in zip(KEYS,row['error'],row['scale'])})
    save(path,result)
    progress(out, f'比较 {name} Gauss {order} 完成')
    return result


def analyze(out, factor):
    states = {}; records = {}
    for case in CASES:
        states[case], records[case] = load(out,case,factor)
    times = {case:metric(out,f'time-{case}-{factor}',load(out,case,factor/2)[0],states[case]) for case in CASES}
    pairs = {case:metric(out,f'{case}-fineY-{factor}',states[case],states['fineY']) for case in ('localY','Y')}
    coarse = {case:metric(out,f'{case}-fineY-{factor/2}',load(out,case,factor/2)[0],load(out,'fineY',factor/2)[0]) for case in ('localY','Y')}
    quality = {}
    for region in ('all','band','outside'):
        quality[region] = {}
        for key in KEYS:
            a,b = [pairs[c]['regions'][region]['rms'][key] for c in ('localY','Y')]
            ac,bc = [coarse[c]['regions'][region]['rms'][key] for c in ('localY','Y')]
            gain, coarse_gain = b-a, bc-ac
            quality[region][key] = dict(ratio=a/b if b else None, coarse_ratio=ac/bc if bc else None,
                gain=gain,coarse_gain=coarse_gain,ordering_stable=bool(gain*coarse_gain>0),
                gain_time_relative=abs(gain-coarse_gain)/max(abs(gain),1e-30))
    time_passed = {c:all(times[c]['regions']['all']['relative'][k]<TIME_TARGET for k in ('F','P','v')) for c in CASES}
    spatial = {c:all(pairs[c]['regions']['all']['relative'][k]<SPACE_TARGET for k in ('F','P','v')) for c in pairs}
    result = dict(protocol='short-full-grid-consistent-pic-v1',duration=DURATION,fine_dt=DT/factor,factor=factor,
        initial=fingerprint(initial()),time_target=TIME_TARGET,space_screen_target=SPACE_TARGET,
        time=times,spatial=pairs,quality=quality,time_passed=time_passed,space_screen_passed=spatial,
        full_reference_converged=False,local_enrichment_justified=False,
        numerical={c:r['checks'] for c,r in records.items()},
        max_transfer_relative=max(row['transfer_roundtrip_relative'] for r in records.values() for row in r['rows']),
        max_history_error=max(max(r['history_error'].values()) for r in records.values()),
        consistent_mass={c:r['rank'] for c,r in records.items()},storage=guard())
    save(out/'results.json',result)
    progress(out,f'时间检查 {time_passed}；空间工程筛选 {spatial}，不认证整体参考收敛')
    return result


def audit(out,factor):
    report = dict(endpoints={},metric_quadrature={})
    for case in CASES:
        state,_ = load(out,case,factor)
        report['endpoints'][case] = endpoint(out,case,state,source=source(case))
        save(out/'audit-progress.json',report)
        progress(out, f'{case} 终态 Gauss 5/7：{report["endpoints"][case]}')
    for case in ('localY','Y'):
        a,b=load(out,case,factor)[0],load(out,'fineY',factor)[0]
        low=metric(out,f'{case}-fineY-{factor}',a,b,3)
        high=metric(out,f'{case}-fineY-{factor}',a,b,5)
        report['metric_quadrature'][case]={region:{k:abs(low['regions'][region]['rms'][k]/max(high['regions'][region]['rms'][k],1e-300)-1) for k in KEYS} for region in ('all','band','outside')}
    report['passed']=all(max(r[k] for k in ('energy','force','tangent_action'))<1e-5 for r in report['endpoints'].values()) and all(v<1e-4 for r in report['metric_quadrature'].values() for row in r.values() for v in row.values())
    save(out/'audit.json',report)
    progress(out,f'终态与比较积分复核：{report["passed"]}')
    return report


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=json.loads((out/'results.json').read_text())
    fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,k in zip(axs,('F','P','v')):
        ax.bar(range(3),[r['quality'][zone][k]['ratio'] for zone in ('all','band','outside')])
        ax.axhline(1,color='red',ls='--');ax.set_xticks(range(3),['Whole','Y band','Outside'])
        ax.set_title(k);ax.set_ylabel('Local / uniform discrepancy vs Y48')
    fig.suptitle('0.125 ms; consistent-mass PIC; Y48 is a comparison, not exact truth')
    fig.savefig(out/'comparison.png',dpi=160);fig.savefig(out/'comparison.pdf');plt.close(fig)


def report(out):
    r=json.loads((out/'results.json').read_text())
    a=json.loads((out/'audit.json').read_text())
    whole=r['quality']['all']
    uniform_better=all(whole[k]['ratio']>1 and whole[k]['ordering_stable'] for k in ('F','P','v'))
    recommendation=('本轮粗细时间步排序一致，整梁与目标条带均未显示局部方案优势；优先保留均匀 Y24。'
                    if uniform_better and all(r['quality']['band'][k]['ratio']>1 for k in ('F','P','v'))
                    else '各方案优劣需按分量及区域判断，不宣称局部加密普遍更好。')
    records=[json.loads(p.read_text()) for p in (out/'original').glob('*.json')]
    records=[m for m in records if m['config']['dt'] in (r['fine_dt'],2*r['fine_dt'])]
    lines=['# 短程非均匀参考与一致传递验证结果','',
           f"释放 {DURATION*1e3:g} ms；{len(records)} 条轨迹，共 {sum(len(m['rows']) for m in records)} 步。",
           '本轮采用完整原网格、共同原初态和 consistent_pic；Y48 仅作较细比较，不是精确解。','',
           '## 时间检查','',
           '相邻时间步的体积 RMS 差，以细时间步 F-I / P / v 的 RMS 归一化；目标 2%。','',
           '| 网格 | F | P | v | 通过 |','|---|---:|---:|---:|---|']
    for c in CASES:
        row=r['time'][c]['regions']['all']['relative']
        lines.append(f"| {c} | {row['F']:.4%} | {row['P']:.4%} | {row['v']:.4%} | {r['time_passed'][c]} |")
    lines += ['', '## 与 Y48 的空间差','',
              '| 网格 | F | P | v | 5% 工程筛选 |','|---|---:|---:|---:|---|']
    for c in ('localY','Y'):
        row=r['spatial'][c]['regions']['all']['relative']
        lines.append(f"| {c} | {row['F']:.4%} | {row['P']:.4%} | {row['v']:.4%} | {r['space_screen_passed'][c]} |")
    lines += ['', '局部 / 均匀差异比 G：小于 1 表示局部方案更接近 Y48。','',
              '| 区域 | F | P | v |','|---|---:|---:|---:|']
    for zone in ('all','band','outside'):
        row=r['quality'][zone]
        lines.append(f"| {zone} | {row['F']['ratio']:.5f} | {row['P']['ratio']:.5f} | {row['v']['ratio']:.5f} |")
    lines += ['', '粗细时间步的排序与改善量变化见 results.json 的 quality；不能只看 G 接近 1 判断优劣。',
              '', '## 独立终态积分','', '| 网格 | 能量 | 力 | 切线作用 |','|---|---:|---:|---:|']
    for c,row in a['endpoints'].items():
        lines.append(f"| {c} | {row['energy']:.3e} | {row['force']:.3e} | {row['tangent_action']:.3e} |")
    maximum=max(v for regions in a['metric_quadrature'].values() for row in regions.values() for v in row.values())
    lines += ['',f"Gauss 3/5 比较指标复核最大相对差 {maximum:.3e}；整体积分审计通过：{a['passed']}。",'',
              '## 完整质量、历史与耗散','',
              f"- 细轨迹最大传递往返相对误差：{r['max_transfer_relative']:.3e}。",
              f"- 细轨迹最大材料/粒子 F 及动能读回绝对误差：{r['max_history_error']:.3e}。",
              '- 所有轨迹逐步检查残差、Jacobian、夹持、传递、历史、能量预算和耗散。',
              '- 能量损失仍包含后向欧拉时间积分耗散，不将其归因于材料内禀阻尼。','',
              '## 决策与范围','',
              recommendation, '',
              '本轮提供短程非均匀 Y 比较和完整一致质量传递实现。X/Z 未在本轮共同加密，',
              '因此不宣称整体空间参考收敛；也不足以支持新增局部余量模式，继续保留现有整体模式。',
              '旧 3 ms 实验与原 MLS/Lite 直接混搭接入的否定结论保留。',
              '本实现适用于相容弹性历史、固定材料参考和完整质量 PIC；不包含 Eulerian 重网格、APIC 或接触。','',
              '## CLI','', '```bash',
              '.venv/bin/python -m demos.aniso_practical --port 8087',
              'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m benchmarks.aniso_practical_validation --device auto',
              '```','', '查看器：http://127.0.0.1:8087；支持三方案叠加和相对 Y48 的位置差。','',
              '![差异比](comparison.png)','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    progress(out,f"结果已汇总：时间通过 {all(r['time_passed'].values())}，积分通过 {a['passed']}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('all','run','analyze','audit','plot','report'),default='all')
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--case',choices=CASES,default='localY')
    parser.add_argument('--factor',type=int,default=2)
    parser.add_argument('--device',default='auto')
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.factor<1 or args.factor & (args.factor-1) or (args.stage!='run' and args.factor<2):
        parser.error('factor must be a power of two; comparisons need factor >= 2')
    if args.stage=='run':run(args.out,args.case,args.factor,args.device)
    elif args.stage=='analyze':analyze(args.out,args.factor)
    elif args.stage=='audit':audit(args.out,args.factor)
    elif args.stage=='plot':plot(args.out)
    elif args.stage=='report':report(args.out)
    else:
        save(args.out/'protocol.json',dict(duration=DURATION,coarse_dt=DT/(args.factor/2),fine_dt=DT/args.factor,
            initial=fingerprint(initial()),cases={c:[a.tolist() for a in source(c).axes] for c in CASES},
            time_target=TIME_TARGET,space_target=SPACE_TARGET,transfer_mode='consistent_pic'))
        for case in CASES:
            for factor in (args.factor//2,args.factor):
                # Isolate trajectories to release device/host allocations between runs.
                cmd=[sys.executable,'-m',__spec__.name,'--stage','run','--case',case,'--factor',str(factor),'--out',str(args.out),'--device',args.device]
                subprocess.run(cmd,check=True)
        result=analyze(args.out,args.factor);audited=audit(args.out,args.factor);plot(args.out);report(args.out)
        if not all(result['time_passed'].values()) or not audited['passed']:
            raise RuntimeError('time or quadrature acceptance failed; inspect REPORT.md before refining')


if __name__=='__main__':main()
