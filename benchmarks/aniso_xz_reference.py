"""Short joint X/Z refinement against the completed Y reference study."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from functools import lru_cache
from . import aniso_practical_validation as previous
from .aniso_spatial_energy import run as run_path
from .aniso_directional_reference import initial,endpoint
from .aniso_convergence_reference import save,fingerprint
from engine.aniso_phase1.directional_reference import directional_geometry
from engine.aniso_phase1.convergence_reference import load_fields
from .aniso_history_increment import guard,DATA_ROOT

OUT=Path('docs/results/xz-mpm/v1/reference')
COUNTS=(96,48,24)

@lru_cache(None)
def source():return directional_geometry(COUNTS)

def path(out,factor):return out/'original'/f'g96x48x24-dt{previous.DT/factor:.13f}'

def load(out,factor):
    p=path(out,factor);r=json.loads(Path(str(p)+'.json').read_text());c=r['config']
    if (c['initial']!=fingerprint(initial()) or c['reference_axes']!=[a.tolist() for a in source().axes] or
        c['duration']!=previous.DURATION or c['dt']!=previous.DT/factor or c.get('transfer_mode')!='consistent_pic' or
        c['order']!=5 or not all(r['checks'].values())):raise ValueError('incompatible XZ reference')
    return load_fields(Path(str(p)+'.npz')),r

def run(out,factor,device):
    import warp as wp
    from utils.resource_guard import prepare_warp_cache
    wp.config.kernel_cache_dir=prepare_warp_cache('/tmp/mpm-lite-warp-cache',DATA_ROOT)
    device=previous.device_choice() if device=='auto' else device
    previous.progress(out,f'joint XZ {COUNTS}, dt={previous.DT/factor:g}, {device}')
    return run_path(out,'original','96x48x24',previous.DT/factor,device,
        duration=previous.DURATION,initial=initial(),source=source(),transfer_mode='consistent_pic',replay=True,
        progress_callback=lambda step,total,info:previous.progress(out,f'jointXZ/{factor}: {step}/{total}'))

def analyze(out):
    fine,m=load(out,2);coarse,_=load(out,1)
    yfine,_=previous.load(previous.OUT,'fineY',2);ycoarse,_=previous.load(previous.OUT,'fineY',1)
    uniform,_=previous.load(previous.OUT,'Y',2)
    time=previous.metric(out,'time-jointXZ',coarse,fine)
    effect=previous.metric(out,'jointXZ-vs-Y48',fine,yfine)
    coarse_effect=previous.metric(out,'jointXZ-vs-Y48-coarse',coarse,ycoarse)
    overall=previous.metric(out,'Y24-vs-jointXZ',uniform,fine)
    old=json.loads((previous.OUT/'results.json').read_text())['spatial']['Y']
    ratios={k:effect['regions']['all']['rms'][k]/old['regions']['all']['rms'][k] for k in ('F','P','v')}
    time_effect={k:abs(coarse_effect['regions']['all']['rms'][k]/effect['regions']['all']['rms'][k]-1) for k in ('F','P','v')}
    r=dict(counts=COUNTS,previous_counts=(88,48,22),duration=previous.DURATION,time=time,
        time_passed=all(time['regions']['all']['relative'][k]<previous.TIME_TARGET for k in ('F','P','v')),
        joint_effect=effect,uniform_Y24_against_joint=overall,XZ_change_over_previous_Y_gap=ratios,
        effect_amplitude_time_change=time_effect,full_reference_converged=False,numerical=m['checks'],storage=guard())
    save(out/'results.json',r);previous.progress(out,'XZ 时间与空间比较完成');return r

def audit(out):
    r=endpoint(out,'jointXZ',load(out,2)[0],source=source())
    save(out/'audit.json',r);previous.progress(out,f'XZ 终态独立积分：{r}');return r

def report(out):
    r=json.loads((out/'results.json').read_text());a=json.loads((out/'audit.json').read_text())
    lines=['# 短程 X/Z 共同加密','',
        'Y48 基线 (88,48,22) → (96,48,24)，原初态、材料、边界、0.125 ms 时长不变。',
        '两档时间步 1.953125/0.9765625 μs，共 192 步。','',
        '| 指标 | 新网格时间差 | XZ 加密变化 / 原 Y24→Y48 差异 | Y24 相对新网格差异 |',
        '|---|---:|---:|---:|']
    for k in ('F','P','v'):
        lines.append(f"| {k} | {r['time']['regions']['all']['relative'][k]:.4%} | {r['XZ_change_over_previous_Y_gap'][k]:.4f} | {r['uniform_Y24_against_joint']['regions']['all']['relative'][k]:.4%} |")
    lines+=['',f"时间检查：{r['time_passed']}；终态积分检查：{a['passed']}。",'',
        '第二列空间比值只比较变化量大小，不将不同方向误差简单相加，也不能证明新网格是真解。',
        '完整原始结果与粗细时间步下的变化量检查见 results.json。','']
    (out/'REPORT.md').write_text('\n'.join(lines))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--stage',choices=('all','run','analyze','audit','report'),default='all')
    p.add_argument('--factor',type=int,choices=(1,2),default=2);p.add_argument('--device',default='auto')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True);guard()
    if args.stage=='run':run(args.out,args.factor,args.device)
    elif args.stage=='analyze':analyze(args.out)
    elif args.stage=='audit':audit(args.out)
    elif args.stage=='report':report(args.out)
    else:
        save(args.out/'protocol.json',dict(counts=COUNTS,initial=fingerprint(initial()),duration=previous.DURATION,
            dt=[previous.DT,previous.DT/2],material_order=5,transfer='consistent_pic'))
        for factor in (1,2):
            subprocess.run([sys.executable,'-u','-m','benchmarks.aniso_xz_reference','--stage','run','--factor',str(factor),'--out',str(args.out),'--device',args.device],check=True)
        r=analyze(args.out);a=audit(args.out);report(args.out)
        if not r['time_passed'] or not a['passed']:raise RuntimeError('XZ acceptance requires attention')

if __name__=='__main__':main()
