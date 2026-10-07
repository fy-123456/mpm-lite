"""Summarize completed nonuniform-reference runs without rerunning physics."""
import argparse
import json
from pathlib import Path
from .aniso_nonuniform_reference import OUT,guard,save


def report(out):
    result=json.loads((out/'results.json').read_text())
    audit=json.loads((out/'audit.json').read_text())
    if not audit['passed'] or not all(result['time_passed'].values()):
        raise RuntimeError('successful time and independent integration checks required')
    runs=[]
    for path in sorted((out/'original').glob('*.json')):
        meta=json.loads(path.read_text());rows=meta['rows']
        runs.append(dict(file=str(path),steps=len(rows),checks=meta['checks'],
            max_residual=max(r['scaled_residual_inf'] for r in rows),
            min_det=min(r['min_det'] for r in rows),max_clamp_speed=max(r['clamp_speed'] for r in rows),
            max_kinetic_readback=max(abs(r['kinetic_readback_delta']) for r in rows),
            max_energy_budget=max(abs(r['energy_budget_residual']) for r in rows),
            history_error=meta['history_error'],elapsed_seconds=meta['elapsed_seconds']))
    summary=dict(runs=runs,steps=sum(r['steps'] for r in runs),passed=bool(runs) and all(all(r['checks'].values()) for r in runs),storage=guard())
    save(out/'numerical-summary.json',summary)
    def worst(value):return max(worst(v) if isinstance(v,dict) else float(v) for v in value.values())
    lines=['# 非均匀 Y 参考加密结果\n\n',
        f'完成 {len(runs)} 条新增轨迹、{summary["steps"]:,} 步。共同细时间步 {result["dt"]*1e6:.11f} μs；六组时间验收及独立积分审计通过。\n\n',
        '局部 Y24 与均匀 Y24 均为 46,464 单元、151,800 个自由位移分量、5,808,000 个材料积分点。局部网格只二分预先指定 Y 条带中的两个原单元。X/Z 对照保留。较细 Y48 是单独预算的比较网格，尚未认证为收敛真解。\n\n',
        '## 与较细 Y48 的差异比\n\n',
        'G = ‖局部 Y24 − Y48‖ / ‖均匀 Y24 − Y48‖，使用体积加权 RMS。小于 1 表示更接近本轮较细 Y 比较网格；不等同于已知精确误差更小。目标条带占体积 12.5%。\n\n',
        '| 场 | 整梁 G | 条带内 G | 条带外 G | 细步改善量 | 改善量的时间相对变化 |\n|---|---:|---:|---:|---:|---:|\n']
    def fmt(v):return '未定义' if v is None else f'{v:.7g}'
    for k in ('x','F','P','v'):
        q=result['quality_against_fineY'][k]
        vals=[q['ratio'],q['regions']['band']['ratio'],q['regions']['outside']['ratio'],q['gain'],q['gain_time_relative']]
        lines.append('| '+k+' | '+' | '.join(map(fmt,vals))+' |\n')
    lines+=['\n改善量为均匀网格差异 RMS 减去局部网格差异 RMS。负值代表局部网格差异更大；时间相对变化在改善量接近零时可能放大，应结合原始差异查看。\n\n',
        '## 时间尺度与保留的 X/Z 检查\n\n',
        'T_f=‖f₂dt−f_dt‖/S_f，S_f 为固定增强／未增强候选的差异。下表取候选 17、33 两种尺度中的较大值。门槛为 F/P/v 全部 <0.05。\n\n',
        '| 网格 | max T_F | max T_P | max T_v |\n|---|---:|---:|---:|\n']
    for c,t in result['time'].items():
        lines.append('| '+c+' | '+' | '.join(fmt(max(v['ratios'][k] for v in t['comparison'].values())) for k in ('F','P','v'))+' |\n')
    lines+=['\n保留的分轴 R_f=‖f_加密−f_基线‖/S_f，空间目标为 <0.1：\n\n',
            '| 方向／候选尺度 | R_F | R_P | R_v |\n|---|---:|---:|---:|\n']
    for name in ('X_base','Z_base'):
        for g,c in result['pairs'][name]['comparison'].items():
            lines.append('| '+name+'/'+g+' | '+' | '.join(fmt(c['ratios'][k]) for k in ('F','P','v'))+' |\n')
    lines+=['\n两档时间步下空间差异向量的相对变化（D）；门槛沿用 0.05：\n\n',
        '| 比较 | D_F | D_P | D_v | 通过 |\n|---|---:|---:|---:|---|\n']
    for name,row in result['effect_time'].items():
        lines.append('| '+name+' | '+' | '.join(fmt(row['relative'][k]) for k in ('F','P','v'))+' | '+str(row['passed'])+' |\n')
    lines+=['\n## 验证与限制\n\n',
        f'全部新增轨迹的数值检查通过：{summary["passed"]}。独立积分复核最大相对变化：空间 {worst(audit["space"]):.3g}，时间 {worst(audit["time"]):.3g}，候选差异尺度 {worst(audit["gap"]):.3g}。\n\n',
        '原初态、旧历史界面、材料、本构、夹具和求解器保持一致。新增网格的终态能量、力、解析切线作用采用 Gauss 5/7 复核，场差异另用高阶公共细分积分复核。\n\n',
        '本轮没有叠加局部 Y 与 X/Z，因此不能判断其交互项；Y48 没有再与更细 Y 网格验证收敛，且仍保持原 X/Z 分辨率。无论 G 的大小如何，都不能据此宣布整体空间参考达标，也不能把参考变化的集中区域直接认定为候选算法错误位置。\n\n',
        f'最终系统／数据盘剩余 {summary["storage"]["system_free_bytes"]/2**30:.2f}/{summary["storage"]["data_free_bytes"]/2**30:.2f} GiB。\n\n',
        '[原始结果](results.json) · [独立审计](audit.json) · [数值汇总](numerical-summary.json)\n\n',
        '![与较细 Y 比较](probe-comparison.png)\n\n![三方向剖面](profiles.png)\n']
    (out/'RESULTS_ZH.md').write_text(''.join(lines))
    if out.resolve()==OUT.resolve():
        doc=out.resolve().parents[2]/'ANISO_NONUNIFORM_REFERENCE_ZH.md'
        text=doc.read_text();start=text.index('状态更新：');end=text.index('前轮见',start)
        ratios='、'.join(k+'='+fmt(result['quality_against_fineY'][k]['ratio']) for k in ('F','P','v'))
        text=text[:start]+f'状态更新：完成 {len(runs)} 条新增轨迹、{summary["steps"]:,} 步，时间门槛和独立积分复核通过。整梁差异比 G 为 {ratios}；较细 Y48 尚未认证为收敛真解。完整表格和剖面见[本轮结果](results/nonuniform-reference/v1/RESULTS_ZH.md)。下文保留协议及运行记录。'+text[end:]
        doc.write_text(text)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT);a=p.parse_args();report(a.out)


if __name__=='__main__':main()
